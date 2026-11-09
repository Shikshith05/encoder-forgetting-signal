"""CPU contract tests: real autograd/AdamW, no CLIP weights or dataset downloads."""
from __future__ import annotations

import ast
import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import common, finetune


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(4, 3)
        self.token_embedding = nn.Embedding(32, 3)
        self.text_projection = nn.Linear(3, 3)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, images):
        return self.visual(images)

    def encode_text(self, tokens):
        return self.text_projection(self.token_embedding(tokens).mean(dim=1))


def tokenize(prompts):
    return torch.tensor([[ord(c) % 32 for c in p[:3].ljust(3)] for p in prompts])


@pytest.fixture
def config():
    return {"model": {"name": "ViT-B-32", "pretrained": "openai"},
            "data": {"root": "unused", "seed": 0, "max_images": 8, "num_workers": 0},
            "prompts": ["{}", "{} photo"],
            "finetune": {"optimizer": "AdamW", "lr": 1e-5, "batch_size": 64,
                         "epochs": 1, "weight_decay": 0.1}}


def tiny_loader():
    generator = torch.Generator().manual_seed(5)
    return DataLoader(TensorDataset(torch.randn(8, 4, generator=generator),
                                   torch.arange(8) % 2), batch_size=4)


@pytest.mark.parametrize("frozen", [False, True])
def test_optimizer_updates_only_allowed_towers(config, frozen):
    torch.manual_seed(0)
    model = TinyCLIP()
    before = copy.deepcopy(model.state_dict())
    result = finetune.train_model(model, tiny_loader(), tokenize, ["cat", "dog"],
                                 config["prompts"], config, "cpu", frozen)
    assert torch.equal(model.logit_scale, before["logit_scale"])
    assert not torch.equal(model.visual.weight, before["visual.weight"])
    for name in ("token_embedding.weight", "text_projection.weight", "text_projection.bias"):
        assert torch.equal(model.state_dict()[name], before[name]) == frozen
    assert result["train_images"] == 8
    assert np.isfinite(result["train_loss"])
    assert result["train_seconds"] > 0
    if frozen:
        assert result["text_frozen_verified"] is True


def test_prompt_checkpointing_matches_full_gradients():
    torch.manual_seed(7)
    model = TinyCLIP()
    tokens = tokenize(["cat", "cats", "dog", "dogs"])
    plain = finetune.prompt_features(model, tokens, 2, 2, chunk_size=4)
    (plain * torch.arange(6).reshape(2, 3)).sum().backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    checked = finetune.prompt_features(model, tokens, 2, 2, chunk_size=1, checkpoint_text=True)
    (checked * torch.arange(6).reshape(2, 3)).sum().backward()
    torch.testing.assert_close(plain, checked)
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name])


def test_dataset_protocol_disjoint_eurosat_pets_train_and_small_flowers(config, monkeypatch):
    import torchvision.datasets as datasets
    calls = []

    class FakeDataset:
        classes = ["cat", "dog"]

        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __len__(self):
            return 30

    for name in ("EuroSAT", "OxfordIIITPet", "Flowers102", "DTD"):
        monkeypatch.setattr(datasets, name, FakeDataset)
    test, test_ids, _ = common.get_dataset_loader("EuroSAT", None, config, "test")
    train, _, _ = common.get_dataset_loader("EuroSAT", None, config, "train")
    permutation = torch.randperm(30, generator=torch.Generator().manual_seed(0)).tolist()
    assert test.dataset.indices == permutation[:8]
    assert train.dataset.indices == permutation[8:16]
    assert not set(test.dataset.indices) & set(train.dataset.indices)
    assert test_ids == [f"EuroSAT_{i}" for i in permutation[:8]]
    assert common.get_dataset_loader("EuroSAT", None, config)[1] == test_ids
    common.get_dataset_loader("OxfordIIITPet", None, config, "train")
    assert calls[-1]["split"] == "trainval"
    common.get_dataset_loader("DTD", None, config)
    assert calls[-1]["split"] == "test"
    config["data"]["max_images"] = 2000
    flowers, _, classes = common.get_dataset_loader("Flowers102", None, config, "train")
    assert len(flowers.dataset) == 30  # Smaller splits are never padded/repeated.
    assert len(classes) == 102
    assert classes[37] == "great masterwort"
    assert classes[-2:] == ["trumpet creeper", "blackberry lily"]
    assert len(set(classes)) == 102


def test_evaluation_matches_shared_prompt_ensembling(config):
    torch.manual_seed(0)
    model = TinyCLIP().eval()
    classes = ["cat", "dog"]
    tokens = finetune.tokenize_classes(tokenize, classes, config["prompts"], "cpu")
    with torch.no_grad():
        actual = finetune.prompt_features(model, tokens, 2, 2).numpy()
    expected = common.extract_text_features(model, tokenize, classes, config["prompts"], "cpu")
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    metrics = finetune.evaluate_model(model, {"Pets": (tiny_loader(), [], classes)},
                                      tokenize, config["prompts"], "cpu")
    assert metrics["Pets_test_images"] == 8
    assert 0 <= metrics["Pets_accuracy_pct"] <= 100


@pytest.mark.parametrize("drop,expected_runs", [(0.5, 6), (1.0, 4), (2.0, 4)])
def test_pipeline_branches_checkpoints_fallback_and_resume(config, tmp_path, monkeypatch, drop, expected_runs):
    parents = []

    def load(ckpt=None, device="cpu", cfg=None):
        parents.append(ckpt)
        model = TinyCLIP()
        if ckpt:
            model.load_state_dict(torch.load(ckpt, weights_only=True))
        return model, None, tokenize

    def loader(name, preprocess, cfg, split):
        return tiny_loader(), [f"{name}/{split}/{i}" for i in range(8)], ["cat", "dog"]

    accuracies = iter([80, 90, 89, 90 - drop, 89, 87, 88])

    def evaluate(*args):
        acc = next(accuracies)
        return {f"{name}_accuracy_pct": acc for name in finetune.DATASETS}

    monkeypatch.setattr(finetune, "load_clip", load)
    monkeypatch.setattr(finetune, "get_dataset_loader", loader)
    monkeypatch.setattr(finetune, "evaluate_model", evaluate)
    rows = finetune.run_experiments(config, str(tmp_path), "cpu", fallback_lr=3e-5)
    assert len(rows) == expected_runs + 1  # Includes pretrained evaluation.
    assert parents[:2] == [None, None]  # Baseline and Pets load OpenAI weights.
    assert all(p == str(tmp_path / "theta_pets.pt") for p in parents[2:])
    pets = torch.load(tmp_path / "theta_pets.pt", weights_only=True)
    frozen = torch.load(tmp_path / "theta_pets_eurosat_textfrozen.pt", weights_only=True)
    for name in pets:
        if not name.startswith("visual."):
            assert torch.equal(pets[name], frozen[name])
    status = json.loads((tmp_path / "forgetting_check.json").read_text())
    assert status["fallback_triggered"] == (drop < 1)
    assert status["base_ratio_defined"] == (drop > 1)
    original_times = [r.get("train_seconds") for r in rows]
    parents.clear()
    resumed = finetune.run_experiments(config, str(tmp_path), "cpu", fallback_lr=3e-5)
    assert parents == [None]  # No training model loads after successful reuse.
    assert [r.get("train_seconds") for r in resumed] == original_times
    assert (tmp_path / "training_results.csv").exists()
    with (tmp_path / "theta_pets.pt").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(FileExistsError, match="mismatched"):
        finetune.run_experiments(config, str(tmp_path), "cpu")


def test_reject_changed_protocol(config, tmp_path, monkeypatch):
    monkeypatch.setattr(finetune, "load_clip", lambda **kw: (TinyCLIP(), None, tokenize))
    monkeypatch.setattr(finetune, "get_dataset_loader", lambda *a: (tiny_loader(), [], ["cat", "dog"]))
    (tmp_path / "protocol.json").write_text('{}')
    with pytest.raises(ValueError, match="different protocol"):
        finetune.run_experiments(config, str(tmp_path), "cpu")


@pytest.mark.parametrize("frozen", [False, True])
def test_native_openclip_autograd_and_freezing(config, frozen):
    """Exercise OpenCLIP's actual encoder/checkpoint APIs at a tiny CPU scale."""
    pytest.importorskip("open_clip")
    from open_clip.model import CLIP

    torch.manual_seed(4)
    model = CLIP(
        embed_dim=8,
        vision_cfg={"layers": 1, "width": 32, "head_width": 16,
                    "patch_size": 8, "image_size": 16},
        text_cfg={"context_length": 8, "vocab_size": 32, "width": 32,
                  "heads": 2, "layers": 1},
        quick_gelu=True,
    )
    before = copy.deepcopy(model.state_dict())
    inputs = torch.randn(4, 3, 16, 16)
    loader = DataLoader(TensorDataset(inputs, torch.tensor([0, 1, 0, 1])), batch_size=2)

    def native_tokens(prompts):
        return torch.nn.functional.pad(tokenize(prompts), (0, 5))

    result = finetune.train_model(model, loader, native_tokens, ["cat", "dog"],
                                 config["prompts"], config, "cpu", frozen)
    assert np.isfinite(result["train_loss"])
    assert not torch.equal(before["visual.conv1.weight"], model.visual.conv1.weight)
    assert torch.equal(before["token_embedding.weight"], model.token_embedding.weight) == frozen
    assert torch.equal(before["logit_scale"], model.logit_scale)


def test_notebook_embeds_current_sources_and_valid_python():
    from scripts.build_finetune_notebook import ROOT, build

    path = ROOT / "notebooks/B_finetune_Shreyas.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook == build(), "Rebuild the notebook after changing source."
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%%writefile"):
            directive, body = source.split("\n", 1)
            filename = directive.split(" ", 1)[1]
            assert body == (ROOT / filename).read_text()
            if filename.endswith(".py"):
                ast.parse(body)
        elif not source.startswith("%"):
            ast.parse(source)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_shared_loader_matches_openai_activation(config, monkeypatch, tmp_path, checkpoint):
    """The pretrained factory default must not silently substitute GELU."""
    import open_clip
    from open_clip.model import CLIP
    from open_clip.transformer import QuickGELU

    def factory(name, **kwargs):
        assert name == "ViT-B-32"
        assert kwargs["pretrained"] == "openai"
        assert kwargs["force_quick_gelu"] is True
        model = CLIP(
            embed_dim=8,
            vision_cfg={"layers": 1, "width": 32, "head_width": 16,
                        "patch_size": 8, "image_size": 16},
            text_cfg={"context_length": 8, "vocab_size": 32, "width": 32,
                      "heads": 2, "layers": 1},
            quick_gelu=kwargs["force_quick_gelu"],
        )
        return model, None, None

    monkeypatch.setattr(open_clip, "create_model_and_transforms", factory)
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda name: tokenize)
    initial, _, _ = common.load_clip(device="cpu", cfg=config)
    path = tmp_path / "theta_pets.pt"
    if checkpoint:
        torch.save(initial.state_dict(), path)
    model, _, _ = common.load_clip(str(path) if checkpoint else None, "cpu", config)
    assert isinstance(model.visual.transformer.resblocks[0].mlp.gelu, QuickGELU)
    assert isinstance(model.transformer.resblocks[0].mlp.gelu, QuickGELU)

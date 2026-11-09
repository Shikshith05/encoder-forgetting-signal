"""Part B: four small sequential CLIP fine-tuning runs, with measured results.

All ten class prompts participate in cross-entropy. Chunked text activation
checkpointing preserves gradients while fitting the Flowers prompts on Colab.
"""
from __future__ import annotations

import copy
import csv
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
from typing import Any

from src.common import get_dataset_loader, load_clip

DATASETS = ("OxfordIIITPet", "Flowers102", "EuroSAT", "DTD")
RUNS = (
    ("pets", "OxfordIIITPet", "theta_pets.pt", False),
    ("pets_to_flowers", "Flowers102", "theta_pets_flowers.pt", False),
    ("pets_to_eurosat", "EuroSAT", "theta_pets_eurosat.pt", False),
    ("pets_to_eurosat_textfrozen", "EuroSAT", "theta_pets_eurosat_textfrozen.pt", True),
)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_trainable(model: Any, text_frozen: bool) -> None:
    """For native CLIP, visual.* is the entire image tower."""
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("visual.") or not text_frozen)
    model.logit_scale.requires_grad_(False)
    if getattr(model, "logit_bias", None) is not None:
        model.logit_bias.requires_grad_(False)


def text_fingerprint(model: Any) -> str:
    """Hash text parameters AND buffers, including projection and embeddings."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith("visual.") and name not in {"logit_scale", "logit_bias"}:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prompt_features(model: Any, tokens: Any, n_classes: int, n_templates: int,
                    chunk_size: int = 32, checkpoint_text: bool = False) -> Any:
    """Normalize prompts, average per class, then normalize each class mean.

    Non-reentrant checkpointing keeps parameter gradients with integer inputs.
    Nothing is detached during training.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    encoded = []
    for chunk in tokens.split(chunk_size):
        if checkpoint_text and torch.is_grad_enabled():
            part = checkpoint(model.encode_text, chunk, use_reentrant=False)
        else:
            part = model.encode_text(chunk)
        encoded.append(F.normalize(part.float(), dim=-1))
    prompts = torch.cat(encoded).reshape(n_classes, n_templates, -1)
    return F.normalize(prompts.mean(dim=1), dim=-1)


def tokenize_classes(tokenizer: Any, classes: list[str], templates: list[str], device: str) -> Any:
    return tokenizer([t.format(c) for c in classes for t in templates]).to(device)


def _sync(device: str) -> None:
    import torch
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def train_model(model: Any, loader: Any, tokenizer: Any, classes: list[str],
                templates: list[str], cfg: dict, device: str,
                text_frozen: bool = False) -> dict:
    import torch
    import torch.nn.functional as F

    settings = cfg["finetune"]
    seed_everything(cfg["data"]["seed"])
    configure_trainable(model, text_frozen)
    model.train()
    if text_frozen:
        model.eval()
        model.visual.train()
    if hasattr(model.visual, "set_grad_checkpointing"):
        model.visual.set_grad_checkpointing(True)
    scale_before = model.logit_scale.detach().clone()
    text_before = text_fingerprint(model) if text_frozen else None
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(settings["lr"]), weight_decay=float(settings.get("weight_decay", 0.1)),
    )
    cuda = torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    tokens = tokenize_classes(tokenizer, classes, templates, device)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    cached_text = None
    if text_frozen:
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=cuda):
            cached_text = prompt_features(model, tokens, len(classes), len(templates))
    loss_sum, correct, count, skipped = 0.0, 0, 0, 0
    for epoch in range(int(settings["epochs"])):
        for step, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=cuda):
                image_vectors = F.normalize(model.encode_image(images).float(), dim=-1)
                text_vectors = cached_text if text_frozen else prompt_features(
                    model, tokens, len(classes), len(templates), checkpoint_text=True,
                )
            logits = model.logit_scale.exp().float() * image_vectors @ text_vectors.T
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; no checkpoint was saved.")
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            batch = labels.numel()
            loss_sum += loss.detach().item() * batch
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += batch
            if step == 1 or step % 10 == 0 or step == len(loader):
                print(f"  epoch {epoch + 1}, batch {step}/{len(loader)}, loss={loss.item():.4f}", flush=True)
    _sync(device)
    seconds = time.perf_counter() - started
    if not torch.equal(scale_before, model.logit_scale.detach()):
        raise AssertionError("logit_scale changed.")
    frozen_verified = text_before == text_fingerprint(model) if text_frozen else None
    if text_frozen and not frozen_verified:
        raise AssertionError("The text-frozen encoder changed.")
    model.eval()
    if hasattr(model.visual, "set_grad_checkpointing"):
        model.visual.set_grad_checkpointing(False)
    return {
        "train_seconds": seconds, "train_minutes": seconds / 60,
        "train_loss": loss_sum / count, "train_online_accuracy_pct": 100 * correct / count,
        "train_images": len(loader.dataset), "optimizer_steps_skipped": skipped,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9 if cuda else None,
        "text_frozen_verified": frozen_verified, "logit_scale_frozen_verified": True,
        "logit_scale_exp": model.logit_scale.exp().item(),
    }


def evaluate_model(model: Any, loaders: dict, tokenizer: Any, templates: list[str], device: str) -> dict:
    import torch
    import torch.nn.functional as F

    model.eval()
    results = {}
    # Float32 evaluation matches the shared feature-extraction functions.
    with torch.no_grad():
        for name, (loader, _ids, classes) in loaders.items():
            tokens = tokenize_classes(tokenizer, classes, templates, device)
            text = prompt_features(model, tokens, len(classes), len(templates))
            correct, count = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                vectors = F.normalize(model.encode_image(images).float(), dim=-1)
                correct += ((vectors @ text.T).argmax(dim=1) == labels).sum().item()
                count += labels.numel()
            results[f"{name}_accuracy_pct"] = 100 * correct / count
            results[f"{name}_test_images"] = count
            print(f"  {name}: {100 * correct / count:.2f}% ({count} test images)", flush=True)
    return results


def _json_write(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment(device: str) -> dict:
    import platform
    import torch

    versions = {}
    for name in ("torch", "torchvision", "open_clip_torch", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": platform.python_version(), "packages": versions,
            "device": device, "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None}


def run_experiments(cfg: dict, output_dir: str, device: str = "cuda", *,
                    fallback_lr: float | None = None) -> list[dict]:
    """Run/reuse four base runs; optionally add a matched EuroSAT sensitivity pair.

    Reuse requires identical configuration, source code, parent checkpoint hash,
    and checkpoint checksum. Partial/unrecognised artifacts are not overwritten.
    """
    import torch

    if cfg["model"]["name"] != "ViT-B-32" or cfg["model"]["pretrained"] != "openai":
        raise ValueError("This Phase-1 workflow supports ViT-B-32 / openai only.")
    if cfg["data"]["seed"] != 0:
        raise ValueError("The shared protocol requires seed=0.")
    if cfg["finetune"]["optimizer"] != "AdamW":
        raise ValueError("The shared protocol requires AdamW.")
    if float(cfg["finetune"]["lr"]) != 1e-5 or int(cfg["finetune"]["epochs"]) != 1 or cfg["finetune"]["batch_size"] != 64:
        raise ValueError("Base runs require lr=1e-5, epochs=1, batch_size=64.")
    if fallback_lr is not None and fallback_lr != 3e-5:
        raise ValueError("The optional sensitivity learning rate is 3e-5.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab before running training.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    environment = _environment(device)
    seed_everything(0)
    initial, preprocess, tokenizer = load_clip(device=device, cfg=cfg)
    tests = {name: get_dataset_loader(name, preprocess, cfg, "test") for name in DATASETS}
    trains = {name: get_dataset_loader(name, preprocess, cfg, "train") for name in DATASETS[:3]}
    manifests = {"test": {name: {"ids": value[1], "classes": value[2]} for name, value in tests.items()},
                 "train": {name: {"ids": value[1], "classes": value[2]} for name, value in trains.items()},
                 "eurosat_split": "seed-0 permutation: first max_images test; next max_images train",
                 "pets_train_split": "trainval", "dtd_partition": 1}
    code_hash = hashlib.sha256()
    for filename in ("common.py", "finetune.py"):
        code_hash.update(Path(__file__).with_name(filename).read_bytes())
    protocol = {"cfg": cfg, "manifests": manifests, "code_sha256": code_hash.hexdigest(),
                "packages": environment["packages"]}
    protocol_file = output / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError("Output folder contains a different protocol. Choose a new output folder.")
    _json_write(protocol_file, protocol)
    _json_write(output / "environment.json", environment)
    baseline_path = output / "baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
    else:
        print("Pretrained CLIP baseline", flush=True)
        started = time.perf_counter()
        baseline = {"run": "pretrained", **evaluate_model(initial, tests, tokenizer, cfg["prompts"], device),
                    "eval_seconds": time.perf_counter() - started}
        _json_write(baseline_path, baseline)
    initial.cpu()
    del initial
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    rows = [baseline]

    def run_one(spec: tuple, run_cfg: dict, directory: Path) -> dict:
        name, dataset, filename, frozen = spec
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = directory / filename
        metrics_path = checkpoint_path.with_suffix(".json")
        parent = None if name == "pets" else output / "theta_pets.pt"
        parent_hash = _file_hash(parent) if parent else None
        signature = {"cfg": run_cfg, "parent_sha256": parent_hash, "text_frozen": frozen}
        if checkpoint_path.exists() or metrics_path.exists():
            if checkpoint_path.exists() and metrics_path.exists():
                saved = json.loads(metrics_path.read_text())
                if saved["signature"] == signature and saved["checkpoint_sha256"] == _file_hash(checkpoint_path):
                    print(f"Reusing completed {name}", flush=True)
                    return saved["metrics"]
            raise FileExistsError(f"Incomplete or mismatched artifacts at {checkpoint_path}; use a new folder or move them aside.")
        print(f"\nTraining {name} from {parent or 'OpenAI pretrained weights'}", flush=True)
        seed_everything(0)
        model, _, _ = load_clip(str(parent) if parent else None, device=device, cfg=run_cfg)
        loader, ids, classes = trains[dataset]
        stats = train_model(model, loader, tokenizer, classes, cfg["prompts"], run_cfg, device, frozen)
        started = time.perf_counter()
        accuracy = evaluate_model(model, tests, tokenizer, cfg["prompts"], device)
        eval_seconds = time.perf_counter() - started
        metrics = {"run": name, "checkpoint": str(checkpoint_path),
                   "parent": str(parent) if parent else "openai", "dataset": dataset,
                   "lr": run_cfg["finetune"]["lr"], "epochs": run_cfg["finetune"]["epochs"],
                   "batch_size": run_cfg["finetune"]["batch_size"], "text_frozen": frozen,
                   "gpu": environment["gpu"], "device": device, **stats,
                   "eval_seconds": eval_seconds, **accuracy}
        pets_reference = rows[1]["OxfordIIITPet_accuracy_pct"] if name != "pets" else accuracy["OxfordIIITPet_accuracy_pct"]
        metrics["pets_forgetting_pp"] = pets_reference - accuracy["OxfordIIITPet_accuracy_pct"]
        model.cpu()
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(model.state_dict(), temporary)
        temporary.replace(checkpoint_path)
        _json_write(metrics_path, {"signature": signature, "checkpoint_sha256": _file_hash(checkpoint_path),
                                   "metrics": metrics, "train_ids": ids})
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return metrics

    def save_table() -> None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        temporary = output / "training_results.csv.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / "training_results.csv")

    save_table()
    for spec in RUNS:
        rows.append(run_one(spec, cfg, output))
        save_table()
    drop = rows[3]["pets_forgetting_pp"]
    status = {"base_pets_forgetting_pp": drop, "fallback_triggered": drop < 1.0,
              "base_ratio_defined": drop > 1.0,
              "note": "Sensitivity runs respond to a test metric; report separately as exploratory."}
    if drop < 1.0:
        print(f"Pets forgetting is {drop:.2f} pp (<1). A matched EuroSAT sensitivity pair is needed.", flush=True)
        if fallback_lr is not None:
            stronger = copy.deepcopy(cfg)
            stronger["finetune"]["lr"] = fallback_lr
            for spec in RUNS[2:]:
                rows.append(run_one(spec, stronger, output / "sensitivity_lr3e-5"))
                save_table()
            status["sensitivity_pets_forgetting_pp"] = rows[-2]["pets_forgetting_pp"]
            status["sensitivity_ratio_defined"] = rows[-2]["pets_forgetting_pp"] > 1.0
        else:
            print("Rerun with fallback_lr=3e-5 to add those runs; completed base runs will be reused.", flush=True)
    _json_write(output / "forgetting_check.json", status)
    return rows

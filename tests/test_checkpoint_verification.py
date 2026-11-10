"""Exercise binary integrity and frozen-encoder checks using small CPU tensors."""
from __future__ import annotations

import copy
import csv
import json

import pytest
import torch

from scripts import verify_finetune_checkpoints as verifier
from src.common import load_config
from src.finetune import RUNS


def json_write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    def build(fallback=False):
        folder = tmp_path / "export"
        folder.mkdir()
        cfg = load_config()
        manifests = {name: {"ids": [f"{name}_train_0", f"{name}_train_1"]}
                     for name in ("OxfordIIITPet", "Flowers102", "EuroSAT")}
        json_write(folder / "protocol.json", {"code_sha256": verifier.source_hash(),
                    "cfg": cfg, "packages": {}, "manifests": {"train": manifests}})
        json_write(folder / "environment.json", {"packages": {}})
        json_write(folder / "forgetting_check.json", {"fallback_triggered": fallback})
        json_write(folder / "baseline.json", {"run": "pretrained"})
        parent = {"visual.weight": torch.zeros(2, 2),
                  "token_embedding.weight": torch.zeros(2, 2),
                  "logit_scale": torch.tensor(4.6051702)}
        monkeypatch.setattr(verifier, "model_schema", lambda: {
            name: (tuple(value.shape), str(value.dtype)) for name, value in parent.items()
        })
        specs = [(name, ds, filename, frozen, 1e-5) for name, ds, filename, frozen in RUNS]
        if fallback:
            specs.extend((name, ds, "sensitivity_lr3e-5/" + filename, frozen, 3e-5)
                         for name, ds, filename, frozen in RUNS[2:])
        rows = []
        for i, (name, ds, filename, frozen, lr) in enumerate(specs):
            path = folder / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            state = {key: value.clone() for key, value in parent.items()}
            if i:
                state["visual.weight"] += i * 0.01
                if not frozen:
                    state["token_embedding.weight"] += i * 0.01
            torch.save(state, path)
            settings = copy.deepcopy(cfg)
            settings["finetune"]["lr"] = lr
            metrics = {"run": name, "dataset": ds, "text_frozen": frozen, "lr": lr,
                       "checkpoint": "/kaggle/working/checkpoints/" + filename,
                       "logit_scale_frozen_verified": True,
                       "text_frozen_verified": True if frozen else None,
                       "logit_scale_exp": state["logit_scale"].exp().item(),
                       "train_seconds": 1.0}
            json_write(path.with_suffix(".json"), {
                "checkpoint_sha256": verifier.file_hash(path),
                "signature": {"cfg": settings, "text_frozen": frozen,
                    "parent_sha256": None if i == 0 else verifier.file_hash(folder / "theta_pets.pt")},
                "train_ids": manifests[ds]["ids"], "metrics": metrics,
            })
            rows.append(metrics)
        with (folder / "training_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerow({"run": "pretrained"})
            writer.writerows(rows)
        return folder
    return build


@pytest.mark.parametrize("fallback,expected_count", [(False, 4), (True, 6)])
def test_verify_complete_base_and_sensitivity_bundles(bundle, fallback, expected_count):
    result = verifier.verify(bundle(fallback))
    assert result["status"] == "passed"
    assert result["checkpoint_count"] == expected_count
    frozen = [row for row in result["checkpoints"] if row["text_frozen"]]
    assert all(row["frozen_text_bitwise_verified"] for row in frozen)
    assert all(row["text_tensors_changed_from_pets"] == 0 for row in frozen)
    assert all(row["visual_tensors_changed_from_pets"] > 0 for row in frozen)


def test_reject_corrupted_checkpoint_bytes(bundle):
    folder = bundle()
    with (folder / "theta_pets_eurosat.pt").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verifier.verify(folder)


@pytest.mark.parametrize("problem,message", [
    ("text", "Frozen text weights changed"),
    ("shape", "Unexpected tensor shape"),
    ("nonfinite", "Non-finite model weights"),
])
def test_inspect_tensor_values_even_when_checksum_is_updated(bundle, problem, message):
    folder = bundle()
    path = folder / "theta_pets_eurosat_textfrozen.pt"
    state = torch.load(path, weights_only=True)
    if problem == "text":
        state["token_embedding.weight"] += 0.1
    elif problem == "shape":
        state["visual.weight"] = torch.zeros(3, 3)
    else:
        state["visual.weight"][0, 0] = float("nan")
    torch.save(state, path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["checkpoint_sha256"] = verifier.file_hash(path)
    json_write(path.with_suffix(".json"), metadata)
    with pytest.raises(ValueError, match=message):
        verifier.verify(folder)


def test_reject_wrong_parent_hash(bundle):
    folder = bundle()
    path = folder / "theta_pets_flowers.json"
    metadata = json.loads(path.read_text())
    metadata["signature"]["parent_sha256"] = "incorrect parent"
    json_write(path, metadata)
    with pytest.raises(ValueError, match="Parent/configuration signature mismatch"):
        verifier.verify(folder)


def test_reject_mismatched_csv_timing(bundle):
    folder = bundle()
    path = folder / "theta_pets_flowers.json"
    metadata = json.loads(path.read_text())
    metadata["metrics"]["train_seconds"] = 99.0
    json_write(path, metadata)
    with pytest.raises(ValueError, match="CSV and checkpoint sidecar differ"):
        verifier.verify(folder)


def test_reject_mismatched_baseline_record(bundle):
    folder = bundle()
    json_write(folder / "baseline.json", {"run": "pretrained", "eval_seconds": 123})
    with pytest.raises(ValueError, match="Baseline JSON and CSV differ"):
        verifier.verify(folder)

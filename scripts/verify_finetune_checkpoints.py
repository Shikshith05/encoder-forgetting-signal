"""Verify a complete Part B checkpoint export on CPU, without downloading weights.

Checks binary SHA-256 against each JSON sidecar, parent hashes, the CLIP schema,
finite tensors, frozen scale/text weights and consistency with the run's CSV.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.finetune import RUNS


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash() -> str:
    return hashlib.sha256(
        (REPO_ROOT / "src/common.py").read_bytes()
        + (REPO_ROOT / "src/finetune.py").read_bytes()
    ).hexdigest()


def model_schema() -> dict:
    import open_clip

    # The meta device creates shapes, not allocated parameter storage. No
    # pretrained model is downloaded, and there is no GPU execution.
    model = open_clip.create_model(
        "ViT-B-32", pretrained=None, force_quick_gelu=True, device="meta",
    )
    return {name: (tuple(value.shape), str(value.dtype))
            for name, value in model.state_dict().items()}


def _same_csv_value(saved, cell: str) -> bool:
    if saved is None:
        return cell == ""
    if isinstance(saved, bool):
        return cell == str(saved)
    if isinstance(saved, (float, int)):
        return math.isclose(float(saved), float(cell), rel_tol=1e-12, abs_tol=1e-10)
    return str(saved) == cell


def verify(folder: Path) -> dict:
    import torch

    folder = Path(folder)
    protocol = json.loads((folder / "protocol.json").read_text())
    environment = json.loads((folder / "environment.json").read_text())
    status = json.loads((folder / "forgetting_check.json").read_text())
    if protocol["code_sha256"] != source_hash():
        raise ValueError("Training source fingerprint differs from this repository's corrected code.")
    if protocol["packages"] != environment["packages"]:
        raise ValueError("Environment packages differ from the protocol.")
    cfg = protocol["cfg"]
    if cfg["model"]["name"] != "ViT-B-32" or cfg["model"]["pretrained"] != "openai":
        raise ValueError("Expected OpenAI ViT-B-32 checkpoints.")
    with (folder / "training_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    baseline = json.loads((folder / "baseline.json").read_text())
    baseline_rows = [row for row in rows if row["run"] == "pretrained"]
    if len(baseline_rows) != 1 or any(
        field not in baseline_rows[0] or not _same_csv_value(value, baseline_rows[0][field])
        for field, value in baseline.items()
    ):
        raise ValueError("Baseline JSON and CSV differ.")
    pets_row = next(row for row in rows if row["run"] == "pets")
    recorded_root = PurePosixPath(pets_row["checkpoint"]).parent
    rows_by_file = {}
    for row in rows:
        if row["checkpoint"]:
            relative = PurePosixPath(row["checkpoint"]).relative_to(recorded_root).as_posix()
            if relative in rows_by_file:
                raise ValueError(f"Duplicate CSV checkpoint: {relative}")
            rows_by_file[relative] = row
    specs = [(name, dataset, filename, frozen, float(cfg["finetune"]["lr"]))
             for name, dataset, filename, frozen in RUNS]
    if status["fallback_triggered"]:
        specs.extend((name, dataset, f"sensitivity_lr3e-5/{filename}", frozen, 3e-5)
                     for name, dataset, filename, frozen in RUNS[2:])
    expected_files = {spec[2] for spec in specs}
    found_files = {p.relative_to(folder).as_posix() for p in folder.rglob("*.pt")}
    if expected_files != found_files or expected_files != set(rows_by_file):
        raise ValueError("Checkpoint files, CSV and expected base/sensitivity runs differ.")
    if len(rows) != len(expected_files) + 1:
        raise ValueError("Unexpected CSV row count.")

    schema = model_schema()
    parent_path = folder / "theta_pets.pt"
    parent_sha = file_hash(parent_path)
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    checks = []
    for run, dataset, relative, frozen, lr in specs:
        path = folder / relative
        print(f"Verifying {relative}", flush=True)
        metadata = json.loads(path.with_suffix(".json").read_text())
        checksum = parent_sha if relative == "theta_pets.pt" else file_hash(path)
        if checksum != metadata["checkpoint_sha256"]:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        expected_cfg = copy.deepcopy(cfg)
        expected_cfg["finetune"]["lr"] = lr
        expected_signature = {"cfg": expected_cfg,
                              "parent_sha256": None if relative == "theta_pets.pt" else parent_sha,
                              "text_frozen": frozen}
        if metadata["signature"] != expected_signature:
            raise ValueError(f"Parent/configuration signature mismatch: {relative}")
        metrics = metadata["metrics"]
        row = rows_by_file[relative]
        if (metrics["run"], metrics["dataset"], metrics["text_frozen"], float(metrics["lr"])) != (run, dataset, frozen, lr):
            raise ValueError(f"Run metadata mismatch: {relative}")
        for field, value in metrics.items():
            if field not in row or not _same_csv_value(value, row[field]):
                raise ValueError(f"CSV and checkpoint sidecar differ at {relative}: {field}")
        if metadata["train_ids"] != protocol["manifests"]["train"][dataset]["ids"]:
            raise ValueError(f"Training image IDs differ: {relative}")
        if not metrics["logit_scale_frozen_verified"] or (frozen and not metrics["text_frozen_verified"]):
            raise ValueError(f"Reported freeze checks failed: {relative}")

        state = parent if relative == "theta_pets.pt" else torch.load(
            path, map_location="cpu", weights_only=True,
        )
        if not isinstance(state, dict) or set(state) != set(schema):
            raise ValueError(f"Unexpected model state keys: {relative}")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or (tuple(value.shape), str(value.dtype)) != schema[name]:
                raise ValueError(f"Unexpected tensor shape/dtype at {relative}: {name}")
            if value.is_floating_point() and not torch.isfinite(value).all().item():
                raise ValueError(f"Non-finite model weights at {relative}: {name}")
        if not torch.equal(state["logit_scale"], parent["logit_scale"]):
            raise ValueError(f"Scale differs from Pets parent: {relative}")
        if not math.isclose(state["logit_scale"].exp().item(), metrics["logit_scale_exp"], abs_tol=1e-4):
            raise ValueError(f"Scale does not match reported value: {relative}")
        visual_changed = [name for name in state if name.startswith("visual.")
                          and not torch.equal(state[name], parent[name])]
        text_changed = [name for name in state
                        if not name.startswith("visual.") and name not in {"logit_scale", "logit_bias"}
                        and not torch.equal(state[name], parent[name])]
        if relative != "theta_pets.pt" and not visual_changed:
            raise ValueError(f"Vision tower did not update: {relative}")
        if frozen and text_changed:
            raise ValueError(f"Frozen text weights changed: {relative}")
        if relative != "theta_pets.pt" and not frozen and not text_changed:
            raise ValueError(f"Full update left the text tower unchanged: {relative}")
        checks.append({
            "file": relative, "bytes": path.stat().st_size, "sha256": checksum,
            "sidecar_sha256": file_hash(path.with_suffix(".json")),
            "parent_sha256": expected_signature["parent_sha256"],
            "tensor_count": len(state), "schema_verified": True, "finite_weights_verified": True,
            "logit_scale_matches_pets": True, "logit_scale_exp": state["logit_scale"].exp().item(),
            "visual_tensors_changed_from_pets": len(visual_changed),
            "text_tensors_changed_from_pets": len(text_changed),
            "text_frozen": frozen, "frozen_text_bitwise_verified": True if frozen else None,
            "csv_and_sidecar_consistent": True, "training_ids_consistent": True,
        })
        del state
        gc.collect()
    return {
        "status": "passed", "verification_device": "cpu",
        "training_source_sha256": protocol["code_sha256"],
        "model_schema": "ViT-B-32 with QuickGELU in both towers; parameter shapes and dtypes checked",
        "checkpoint_count": len(checks), "checkpoint_bytes": sum(c["bytes"] for c in checks),
        "metadata_file_sha256": {name: file_hash(folder / name) for name in (
            "training_results.csv", "protocol.json", "environment.json", "forgetting_check.json", "baseline.json",
        )},
        "checkpoints": checks,
        "limits": [
            "No dataset was downloaded and no test accuracy was recomputed.",
            "Original pretrained weights were not downloaded; Pets is the reference for frozen-weight comparisons.",
            "Activations are not in state dictionaries; the recorded source and QuickGELU schema establish intended loading.",
            "Encoder-swap contributions require image/text feature evaluation in Part C.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.checkpoint_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.report)
    print(f"Verified {result['checkpoint_count']} checkpoints; report: {args.report}")


if __name__ == "__main__":
    main()

"""Exercise platform setup and Kaggle exports without a GPU or hosted runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import zipfile

import pytest

from scripts.build_finetune_notebook import build
from src.finetune import RUNS


def code_cell(marker):
    return next("".join(c["source"]) for c in build()["cells"]
                if c["cell_type"] == "code" and marker in "".join(c["source"]))


@pytest.mark.parametrize("platform", ["Kaggle", "Colab", "Local Jupyter"])
def test_platform_setup_selects_writable_paths_and_imports(platform, monkeypatch):
    created, changed = [], []
    root = Path.cwd()
    monkeypatch.setattr(Path, "is_dir", lambda path: (
        str(path) == "/kaggle/working" if platform == "Kaggle" else
        str(path) == "/content" if platform == "Colab" else False
    ))
    monkeypatch.setattr(Path, "mkdir", lambda path, **kwargs: created.append(path))
    monkeypatch.setattr(Path, "touch", lambda path: None)
    monkeypatch.setattr(os, "chdir", lambda path: changed.append(path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {}
    exec(code_cell("USE_GOOGLE_DRIVE = False"), namespace)
    assert namespace["PLATFORM"] == platform
    assert namespace["WORK_DIR"] in created
    assert changed == [namespace["WORK_DIR"]]
    assert str(namespace["WORK_DIR"]) in sys.path
    assert namespace["OUTPUT_DIR"].name == "checkpoints_quickgelu"
    if platform == "Kaggle":
        assert namespace["OUTPUT_DIR"].is_relative_to("/kaggle/working")
        assert not namespace["DATA_DIR"].is_relative_to("/kaggle/working")
        assert namespace["EXPORT_DIR"] == Path("/kaggle/working")
    elif platform == "Colab":
        assert namespace["OUTPUT_DIR"].is_relative_to("/content")
        assert namespace["DATA_DIR"] == Path("/content/encoder-forgetting-data")
    else:
        assert namespace["WORK_DIR"] == root / ".notebook_partb"


def completed_outputs(folder, fallback):
    folder.mkdir()
    for _, _, name, frozen in RUNS:
        path = folder / name
        path.write_bytes(b"test checkpoint")
        path.with_suffix(".json").write_text(json.dumps({"metrics": {
            "logit_scale_frozen_verified": True, "text_frozen_verified": frozen,
        }}))
    (folder / "forgetting_check.json").write_text(json.dumps({"fallback_triggered": fallback}))
    (folder / "protocol.json").write_text(json.dumps({"code_sha256": "test source fingerprint"}))
    (folder / "environment.json").write_text('{}')
    (folder / "training_results.csv").write_text('run\npets\n')
    (folder / "interrupted.pt.tmp").write_bytes(b"partial")
    if fallback:
        sensitivity = folder / "sensitivity_lr3e-5"
        sensitivity.mkdir()
        for name in ("theta_pets_eurosat.pt", "theta_pets_eurosat_textfrozen.pt"):
            (sensitivity / name).write_bytes(b"sensitivity checkpoint")


@pytest.mark.parametrize("fallback", [False, True])
def test_kaggle_export_contains_checkpoints_and_metadata_without_colab(tmp_path, monkeypatch, fallback):
    folder = tmp_path / "checkpoints_quickgelu"
    completed_outputs(folder, fallback)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    namespace = {"Path": Path, "OUTPUT_DIR": folder, "IS_KAGGLE": True, "IS_COLAB": False,
                 "EXPORT_DIR": tmp_path}
    exec(code_cell("DOWNLOAD_ARTIFACTS = IS_KAGGLE"), namespace)
    with zipfile.ZipFile(tmp_path / "part_b_artifacts.zip") as archive:
        names = archive.namelist()
        assert all(f"checkpoints/{name}" in names for _, _, name, _ in RUNS)
        assert "checkpoints/protocol.json" in names
        assert not any(name.endswith(".tmp") for name in names)
        assert any("sensitivity_lr3e-5/" in name for name in names) == fallback


def test_export_stops_if_training_is_incomplete(tmp_path):
    folder = tmp_path / "checkpoints_quickgelu"
    completed_outputs(folder, False)
    (folder / "theta_pets_eurosat.pt").unlink()
    namespace = {"Path": Path, "OUTPUT_DIR": folder, "IS_KAGGLE": True, "IS_COLAB": False,
                 "EXPORT_DIR": tmp_path}
    with pytest.raises(AssertionError, match="Missing checkpoint"):
        exec(code_cell("DOWNLOAD_ARTIFACTS = IS_KAGGLE"), namespace)
    assert not (tmp_path / "part_b_artifacts.zip").exists()

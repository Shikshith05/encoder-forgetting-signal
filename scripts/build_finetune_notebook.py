"""Bundle local Part B source into one self-contained Colab notebook (stdlib only)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build() -> dict:
    cells = []

    def add(kind: str, source: str) -> None:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
                "id": f"partb-{len(cells):02d}"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """# Part B — CLIP fine-tuning · Shreyas K M

Run this notebook in **Google Colab → Runtime → Change runtime type → GPU**.
The team's shared location is the GitHub repository. All training code is embedded
below; cloning the repository is unnecessary. Parts A/C/D can import the same
`src/common.py` after the corrected version is shared.

**QuickGELU correction:** this version explicitly matches the OpenAI weights'
activation function in both encoders. If an earlier run printed a QuickGELU
mismatch warning, preserve its files as preliminary artifacts and restart the
baseline and all four experiments with this notebook. Do not load those older
checkpoints into the corrected architecture. The default output folder below is
new so earlier artifacts are retained. Parts A/C/D must use this corrected loader.

**Scope:** CLIP ViT-B/32 (`open_clip`, `pretrained='openai'`), four base training runs:
Pets; Pets→Flowers; Pets→EuroSAT; Pets→EuroSAT with text frozen. The three branches
each start from `theta_pets.pt`. DTD is evaluation only. The 56-pair grid is Phase II.

**Fixed protocol:** AdamW, lr `1e-5`, batch 64 (final batch may be smaller), 1 epoch,
at most 2000 train/test images, seed 0. Flowers has only **1020 training images**;
use all of them without mixing validation or repeating images. Weight decay is 0.1
from the shared config. No scheduler or added classifier. Use the checkpoint's
deterministic CLIP resize/crop preprocessing for both training and testing.

The objective is cross-entropy between each image and **all class text prototypes**.
For every class, normalize its ten prompt embeddings, average, and normalize again.
Both towers receive gradients, except in the text-frozen control. `logit_scale`
retains its loaded value throughout (it is never set to a new constant).

**Split details:** Pets `train` maps to torchvision `trainval`; test is `test`.
EuroSAT has no official split: generate one seed-0 permutation, take its first
2000 indices for test and next 2000 for train. This preserves the team's existing
EuroSAT evaluation subset and prevents leakage. Flowers/DTD use official splits;
DTD uses partition 1. Each official split is independently shuffled with
`torch.Generator().manual_seed(0)` before taking at most 2000 indices.
The complete selection and class order are saved to `protocol.json`.

The 2000-image cap limits processing, **not download size**: torchvision downloads
the full dataset archives. Initial installs/downloads are outside training time.
""")
    add("code", """%pip -q install 'open_clip_torch==3.3.0' 'timm==1.0.22' pyyaml pandas scipy tqdm
""")
    add("code", """from pathlib import Path
import os
WORK_DIR = Path('/content/encoder-forgetting-signal-partb')
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
for directory in ('src', 'configs'):
    Path(directory).mkdir(exist_ok=True)
Path('src/__init__.py').touch()

# No Drive folder is required. Download results before the Colab runtime expires.
# For persistence across runtime resets, optionally enable Drive.
USE_GOOGLE_DRIVE = False
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_DIR = Path('/content/drive/MyDrive/PW26_VK_06/checkpoints_quickgelu')
else:
    OUTPUT_DIR = WORK_DIR / 'artifacts' / 'checkpoints_quickgelu'
print('Checkpoints and measured results:', OUTPUT_DIR)
""")
    add("markdown", """## Shared implementation

These cells materialize the embedded source under `/content`, keeping dataset I/O
on the Colab disk. Checkpoints, metrics, and the protocol use `OUTPUT_DIR`.
Expand these cells to inspect the exact training and split implementation.
""")
    for filename in ("configs/config.yaml", "src/common.py", "src/finetune.py"):
        add("code", f"%%writefile {filename}\n" + (ROOT / filename).read_text())
    add("code", """import torch
from src.common import load_config
from src.finetune import run_experiments

assert torch.cuda.is_available(), 'Select a GPU runtime, then run again.'
print('GPU:', torch.cuda.get_device_name(0))
cfg = load_config()
cfg['data']['root'] = '/content/encoder-forgetting-data'
cfg['data']['num_workers'] = 2
print('Base training settings:', cfg['finetune'])
print('Prompt templates:', cfg['prompts'])
""")
    add("markdown", """## Run the four experiments

This first evaluates pretrained CLIP, then trains and evaluates each checkpoint
on the same four test subsets. Each run prints progress and writes its checkpoint,
JSON metrics and `training_results.csv`. Float32 test evaluation matches Parts A/C.
Training uses mixed precision plus activation checkpointing to reduce GPU memory.

`train_minutes` is **GPU-synchronized elapsed training time**: it includes batch
loading, forward/backward, optimizer updates and the frozen control's prompt cache.
It excludes downloads, model loading, test evaluation, checksum verification and
checkpoint writes. It is a feasibility measurement on the named GPU, not
pure kernel time. `eval_seconds` and peak allocated GPU memory are also recorded.

Local Colab files survive a browser disconnect only while the same runtime lives;
they do not survive a runtime reset. Download artifacts when done, or enable the
optional Drive destination before training if persistence is needed.

Rerunning this cell reuses completed runs only when the protocol, installed package
versions, parent hash and checkpoint checksum match. An interrupted training run
restarts from its parent. Partial checkpoint files must be moved aside if reported.
For a new protocol, choose a new output directory. Keep the same runtime/package
versions when resuming. Do not rerun source-writing cells after importing modules;
restart the runtime after changing source.
""")
    add("code", """rows = run_experiments(cfg, str(OUTPUT_DIR), device='cuda')
""")
    add("markdown", """## Measured table and low-forgetting check

Accuracy columns are percentages; `pets_forgetting_pp` is Pets accuracy at
`theta_pets.pt` minus Pets accuracy after the branch, in **percentage points**.
The pretrained row is evaluation only and has no training time.

If the full EuroSAT branch drops by **less than 1 point**, the next cell runs a
matched EuroSAT pair (both encoders / frozen text) at `3e-5`, each initialized from
the original Pets checkpoint. It preserves the four base checkpoints and writes
the additional two into `sensitivity_lr3e-5/`, logging the changed learning rate.
Otherwise it reuses the four existing runs. This is a conditional extra pair,
not part of the four-run budget. Report it as exploratory because a test metric
triggered it. Greater forgetting is not guaranteed; report small/negative drops
honestly. The Part C dominance ratio is undefined for a drop ≤1 point.
""")
    add("code", """rows = run_experiments(cfg, str(OUTPUT_DIR), device='cuda', fallback_lr=3e-5)

import json
import pandas as pd
table = pd.read_csv(OUTPUT_DIR / 'training_results.csv')
columns = ['run', 'lr', 'epochs', 'train_images', 'train_minutes',
           'OxfordIIITPet_accuracy_pct', 'Flowers102_accuracy_pct',
           'EuroSAT_accuracy_pct', 'DTD_accuracy_pct', 'pets_forgetting_pp']
display(table.reindex(columns=columns).round(3))
print(json.dumps(json.loads((OUTPUT_DIR / 'forgetting_check.json').read_text()), indent=2))
print('Base checkpoints:')
for file in sorted(OUTPUT_DIR.glob('theta_*.pt')):
    print(file.name, f'{file.stat().st_size / 1e6:.1f} MB')
""")
    add("markdown", """## Download the artifacts

The source notebook belongs in GitHub. Model checkpoints are large and are ignored
by this repository's `.gitignore`; share the exported archive separately with the
team. With local Colab storage, download it before ending/resetting the runtime.
The archive includes checkpoints, tables and reproducibility metadata. Enable the
switch below when ready; the archive can be several GB. If Drive is enabled,
artifacts already persist at `OUTPUT_DIR`.
""")
    add("code", """DOWNLOAD_ARTIFACTS = False  # Set True after training, then run this cell.
if DOWNLOAD_ARTIFACTS:
    import zipfile
    from google.colab import files
    archive = WORK_DIR / 'part_b_artifacts.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as bundle:
        for artifact in sorted(OUTPUT_DIR.rglob('*')):
            if artifact.is_file() and not artifact.name.endswith('.tmp'):
                bundle.write(artifact, Path('checkpoints') / artifact.relative_to(OUTPUT_DIR))
    files.download(str(archive))
else:
    print('Artifacts are at', OUTPUT_DIR)
    print('Enable DOWNLOAD_ARTIFACTS to export them before this runtime expires.')
""")
    add("markdown", """## Demo, slides, report and viva

**3–4 minute screen recording (Thursday; Shikshith narrates):**

- 0:00–0:35: show GPU, CLIP model, four-run scope, prompts and split rule.
- 0:35–1:20: show the objective, AdamW settings and frozen `logit_scale`.
- 1:20–2:05: show a live run's progress, checkpoint saving and measured time.
- 2:05–3:05: show the completed table, compare Pets→Flowers and Pets→EuroSAT,
  then explain the frozen-text control and low-forgetting check.
- 3:05–3:40: show saved artifacts and the Phase-II timeline. Training may take
  longer than the video; prepare completed results before recording and show
  live progress separately. Do not present reused runs as newly measured runs.

**Implementation slide:** the two branches from Pets, loss/settings, notebook
progress screenshot, results-table screenshot, artifact location and video link.

**Timeline / Phase-II slide:** Phase I implements two pairs plus a text-frozen
control; Phase II expands to the promised 56-pair grid, repeated seeds and
uncertainty analysis. The four current datasets provide 12 ordered pairs; a
56-pair grid requires eight datasets. Confirm the remaining four in Phase II.
Leave dates tied to the team's agreed schedule rather than inventing them.

**Feasibility box:** public torchvision datasets, public OpenAI CLIP weights and
public project code; GPU model and **measured** minutes per run from the table.
The signal uses forward passes only, with no optimization. Put “minutes” on the
slide only after Part A measures its runtime; this notebook does not measure it.

**Report — Implementation:** describe CLIP, train/test manifests, prompt ensemble,
cross-entropy, AdamW, precision, frozen scale, branches and the control. Describe
EuroSAT's custom holdout explicitly. Include versions from `environment.json`.

**Report — Results:** paste the measured table. State Pets accuracy before and
after each shift and the difference in points; report target-domain accuracy and
GPU time. Separate sensitivity runs from base runs. The close/far designations
are experimental hypotheses, not results. Do not infer encoder contributions
from accuracy alone; Part C supplies the swap/Shapley measurements.

**Viva:** “We optimize image-to-all-class cross-entropy through the image and text
encoders using ten-prompt class prototypes. AdamW uses lr 1e-5, batch 64 and one
epoch; the learned temperature stays frozen. Each branch reloads the Pets
checkpoint. The frozen-text run holds language parameters fixed, so a change in
Pets predictions must come from vision; Part C should give zero text contribution
up to numerical tolerance. Training took **[measured minutes]** on **[GPU]**.”

Sources: [Pets splits](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html),
[EuroSAT API](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html),
[Flowers label order](https://github.com/pytorch/vision/blob/main/torchvision/datasets/flowers102.py),
[OpenCLIP](https://github.com/mlfoundations/open_clip).
""")
    return {"cells": cells, "metadata": {
        "colab": {"name": "B_finetune_Shreyas.ipynb", "provenance": []},
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    target = ROOT / "notebooks/B_finetune_Shreyas.ipynb"
    target.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(target)

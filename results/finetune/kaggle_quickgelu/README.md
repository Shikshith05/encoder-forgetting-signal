# Corrected Kaggle Part B results — Shreyas

Reviewed on 10 November 2026 from the four user-supplied files preserved here.
This run completed **four base training runs and two triggered sensitivity runs**
on one Tesla T4. The additional GPU allocated by Kaggle was not used by the code.
These results supersede the earlier Colab run with the QuickGELU mismatch.

## Implementation and provenance

CLIP ViT-B/32 uses OpenAI pretrained weights with QuickGELU explicitly enabled
in both encoders. AdamW uses weight decay 0.1, batch 64, one epoch and seed 0.
Base learning rate is 1e-5. Each image is classified against all class prototypes,
constructed by averaging the ten normalized prompt embeddings and normalizing
again. Both encoders are optimized except in the frozen-text controls. The loaded
logit scale remains fixed at 100. Training uses mixed precision and activation
checkpointing; evaluation is float32 with deterministic CLIP preprocessing.

Pets uses 2000 trainval images; Flowers uses its entire 1020-image official train
split; EuroSAT uses 2000 training images disjoint from its 2000-image test holdout.
Each dataset's test selection is fixed across evaluations: Pets/Flowers/EuroSAT
have 2000 images each, and DTD has 1880. The exact IDs, prompt templates, class
order and EuroSAT convention are in [protocol.json](protocol.json).

The recorded SHA-256 fingerprint of `src/common.py` followed by `src/finetune.py`
matches the corrected repository source exactly:

```text
6731dd556992fe5296b2199fbaaf9dba8373b55e8de53d378dd272f1992fdbb2
```

[environment.json](environment.json) records Python 3.13.16, PyTorch
2.11.0+cu130, torchvision 0.26.0+cu130, OpenCLIP 3.3.0 and NumPy 2.1.3.
The complete original table is [training_results.csv](training_results.csv).

## Base results: lr 1e-5

All accuracy values are percentages. Forgetting is the Pets checkpoint's
86.05% Pets accuracy minus the branch's Pets accuracy, in percentage points (pp).
Negative forgetting denotes an accuracy improvement.

| Checkpoint | Pets % | Flowers % | EuroSAT % | DTD % | Pets forgetting, pp | Training, s |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained (evaluation only) | 85.60 | 64.80 | 35.30 | 43.24 | — | — |
| Pets | 86.05 | 51.25 | 23.70 | 22.07 | 0.00 | 42.98 |
| Pets → Flowers | 83.65 | 74.40 | 18.55 | 29.41 | 2.40 | 51.88 |
| Pets → EuroSAT | 85.35 | 50.25 | 91.10 | 34.89 | 0.70 | 19.01 |
| Pets → EuroSAT, text frozen | 86.60 | 43.55 | 94.05 | 17.29 | -0.55 | 7.91 |

Pets fine-tuning improved Pets accuracy from 85.60% to 86.05%. From that checkpoint,
Flowers accuracy increased from 51.25% to 74.40%, with 2.40 pp Pets forgetting.
EuroSAT accuracy increased from 23.70% to 91.10%, with only 0.70 pp Pets forgetting.
The frozen-text EuroSAT control reached 94.05% EuroSAT accuracy while Pets improved
by 0.55 pp. Thus the proposed close/far shift labels did not imply greater
forgetting for the far shift in this base run.

The base EuroSAT drop is below the agreed 1 pp trigger. Accordingly,
[forgetting_check.json](forgetting_check.json) records `fallback_triggered=true`
and `base_ratio_defined=false`. Part C may still compute the raw contributions,
but the normalized dominance ratio is undefined under its ≤1 pp rule.

## Exploratory sensitivity check: lr 3e-5

Both sensitivity runs start from the original `theta_pets.pt`, use a fresh AdamW
optimizer and retain the one-epoch, batch-64 protocol. The frozen control uses the
same increased learning rate. Their files live under `sensitivity_lr3e-5/` and
are distinct from the retained base checkpoints.

| Checkpoint | Pets % | Flowers % | EuroSAT % | DTD % | Pets forgetting, pp | Training, s |
|---|---:|---:|---:|---:|---:|---:|
| Pets → EuroSAT | 32.30 | 8.00 | 73.55 | 7.02 | 53.75 | 18.78 |
| Pets → EuroSAT, text frozen | 59.25 | 11.60 | 85.35 | 5.59 | 26.80 | 7.91 |

The stronger full update induced severe forgetting, but its EuroSAT accuracy was
17.55 pp below the base full update. The stronger frozen control also retained
less Pets accuracy and had 8.70 pp lower EuroSAT accuracy than its base control.
These are observed outcomes of stronger adaptation, not evidence that a larger
learning rate is a better training choice. The full sensitivity drop exceeds
1 pp, so the ratio's threshold condition is met; its value still requires the
four encoder-swap accuracies from Part C.

This protocol change was triggered by the Pets test metric. Report it separately
as exploratory, alongside the base results. The difference between the full and
frozen runs is a control comparison; encoder Shapley attribution requires Part C.

## Measured feasibility and optimizer updates

| Measurement | Recorded time |
|---|---:|
| Four base training runs | 121.77 s (2 min 1.77 s) |
| Two sensitivity training runs | 26.69 s |
| All six training runs | 148.47 s (2 min 28.47 s) |
| Seven evaluations, including pretrained baseline | 273.38 s (4 min 33.38 s) |
| Sum of recorded training and evaluation times | 421.85 s (7 min 1.85 s) |

Training time is GPU-synchronized elapsed time, including batch loading and
optimizer work; it is not pure kernel time. Package installation, dataset/weight
downloads, queueing, model loading, file checksums and artifact/checkpoint writes
are outside the recorded training/evaluation totals. This table is not a complete
session wall-clock measurement. Part A's forward-only signal runtime remains a
separate measurement, not established by these fine-tuning results.

Peak allocated GPU memory was 3.12 GB for the full runs and 2.08 GB for frozen
controls (rounded decimal GB). Mixed precision recorded skipped optimizer updates:

| Run | LR | Attempted batches | Skipped updates | Applied updates |
|---|---:|---:|---:|---:|
| Pets | 1e-5 | 32 | 3 | 29 |
| Pets → Flowers | 1e-5 | 16 | 4 | 12 |
| Pets → EuroSAT | 1e-5 | 32 | 4 | 28 |
| Pets → EuroSAT, frozen text | 1e-5 | 32 | 3 | 29 |
| Pets → EuroSAT, sensitivity | 3e-5 | 32 | 4 | 28 |
| Pets → EuroSAT, sensitivity, frozen text | 3e-5 | 32 | 3 | 29 |

The epoch describes a pass through the selected images; the AMP skip counts mean
not every attempted batch applied an optimizer update. Retain this detail when
reporting the precision/training protocol rather than describing all batches as
successful updates. No further hyperparameter changes were made during review.

## Verification scope

[verification.json](verification.json) records the checks and input file hashes.
Checks cover source fingerprint, environment/package agreement, fixed settings,
unique image IDs, class order/counts, EuroSAT train/test disjointness, row and
sample counts, accuracy arithmetic, forgetting/timing consistency, initialization
paths, the fallback condition and reported scale/text freeze flags.

This review used metadata and aggregate accuracies. Checkpoint binaries, their
sidecars/checksums and per-image predictions were not supplied for independent
inspection. The flags report that the training implementation verified the
frozen text tower. Part C should additionally verify unchanged text weights from
the actual checkpoints and match its Pets before/after accuracies to this table.
This is one seed with finite test subsets; no uncertainty intervals, significance
claims or 56-pair validation are established by these results.

## Part C checkpoint handoff

Keep the complete exported `checkpoints/` folder, including its
`sensitivity_lr3e-5/` subfolder, in the team's artifact storage. Share the current
repository branch with Shikshith. Part C's `_load_clip` now delegates to the same
QuickGELU-corrected loader used by Part B. Updating only the checkpoint filenames
in an old copy of Part C would retain its former architecture mismatch.
The existing Part C notebook clones the default branch; select
`shreyas/phase1-finetuning` in that checkout and install `requirements.txt` before
running it, until the team merges these changes into `main`.

After extracting the archive into a GPU environment and setting `CKPT_DIR` to
the extracted checkpoint folder, run the base analysis with:

```bash
python scripts/run_swap.py --ckpt-dir "$CKPT_DIR" --out-dir results/swap/base --device cuda
```

The base analysis should reproduce `A_pp=86.05%` for each pair and the relevant
`A_ff` values above, allowing for numerical tolerance. Its base EuroSAT dominance
ratio must remain undefined. Flowers has a 2.40 pp drop and satisfies the ratio
threshold. Frozen-text controls should give zero text contribution and interaction
up to numerical tolerance.

Run the sensitivity pair separately with the original Pets parent:

```python
from src.common import load_config
from src.swap_shapley import run_pairs

cfg = load_config()
CKPT_DIR = "/path/to/extracted/checkpoints"  # Set to the actual GPU-environment path.
run_pairs(
    [
        ("theta_pets.pt", "sensitivity_lr3e-5/theta_pets_eurosat.pt",
         "Pets->EuroSAT (lr3e-5 exploratory)"),
        ("theta_pets.pt", "sensitivity_lr3e-5/theta_pets_eurosat_textfrozen.pt",
         "Pets->EuroSAT (lr3e-5 exploratory, text frozen)"),
    ],
    CKPT_DIR, "results/swap/sensitivity_lr3e-5", cfg, "cuda",
)
```

Expected sensitivity `A_ff` values are 32.30% and 59.25%, with the same 86.05%
Pets parent accuracy. Save base and sensitivity tables separately. Final slides
and report should include both protocols, the actual six-run cost and Part C's
measured contributions once available. The checkpoint binaries stay outside Git;
only these small result/provenance files are versioned here.

# Shreyas — Phase-I implementation and handoff

Use `notebooks/B_finetune_Shreyas.ipynb` as the single Kaggle/Colab entry point. It embeds
the code, fixed config, demo outline, slide notes, report guidance and viva answers.
It runs without cloning GitHub. This repository has no measured GPU results yet.

For Kaggle, follow [the complete run and handoff guide](KAGGLE_PART_B.md). The same
notebook detects Kaggle, stores outputs under `/kaggle/working`, keeps dataset
downloads in temporary storage, and exports without importing `google.colab`.
Only one GPU is used even when the accelerator allocation contains two GPUs.

The corrected loader explicitly sets `force_quick_gelu=True` for OpenAI weights.
OpenCLIP 3.3 otherwise warns of an activation mismatch; CPU tests of the training
loop alone did not catch the original factory default. Earlier Colab runs showing
that warning must remain preliminary. Restart the baseline and all four runs;
do not reuse their checkpoints with the corrected activation. The notebook now
defaults to `checkpoints_quickgelu` to preserve earlier artifacts. Share the
corrected loader with Parts A/C/D and recompute their affected features/results.

1. Keep the notebook in the shared GitHub repository, then open it in Colab.
2. Select a GPU. Artifacts default to local Colab storage; Drive is optional via
   `USE_GOOGLE_DRIVE`. No shared Drive folder has been created or is required.
3. Run all cells. Downloading all four torchvision archives is required even
   though processing is capped at 2000 images per split. Keep data on `/content`.
4. Retain the four base `.pt` files, their `.json` sidecars, `protocol.json`,
   `baseline.json`, `environment.json`, `forgetting_check.json` and
   `training_results.csv` together in `OUTPUT_DIR`. Set `DOWNLOAD_ARTIFACTS=True`
   in the export cell to download them before resetting/ending a local runtime.
5. Give Shikshith that checkpoint folder and the matching shared data loader.
   Plain model state dicts load with `src.common.load_clip` and existing Part C;
   set Part C's checkpoint path to the extracted folder. These large artifacts
   remain ignored by Git; share the exported archive separately from source.

| Training | Initialization | Output |
|---|---|---|
| Pets | OpenAI ViT-B/32 | `theta_pets.pt` |
| Pets → Flowers | `theta_pets.pt` | `theta_pets_flowers.pt` |
| Pets → EuroSAT | `theta_pets.pt` | `theta_pets_eurosat.pt` |
| Pets → EuroSAT, text frozen | `theta_pets.pt` | `theta_pets_eurosat_textfrozen.pt` |

Base runs use AdamW (weight decay 0.1), lr 1e-5, batch 64, one epoch and seed 0.
Each class prototype averages ten normalized prompt embeddings and is normalized
again; CE uses all class prototypes. Text embeddings are recomputed with gradients
every step unless the tower is frozen. The scale stays at its checkpoint value.
Mixed precision and activation checkpointing reduce memory; evaluation is float32.

The table records GPU-synchronized elapsed training time, including batch loading
and optimizer steps. It excludes model loading, downloads, evaluation and checkpoint
writes. `gpu`, `train_minutes`, `eval_seconds`, `peak_gpu_memory_gb` and package
versions make those measurements interpretable. Online training accuracy is
labelled separately from final held-out test accuracy. Timings are measurements,
not hardware-independent guarantees. Non-finite losses abort; skipped AMP updates
are recorded rather than hidden.

The original shared loader needed two split fixes and a label correction:

- [Pets](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html)
  exposes `trainval`, not `train`. No official training image enters the test split.
- [EuroSAT](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html)
  has no official train/test split. One seed-0 permutation supplies the first 2000
  indices for test and the next 2000 for training. Test IDs keep their previous
  format for Parts A/C/D. Everyone must use this convention and the same cap.
- [Flowers102](https://github.com/pytorch/vision/blob/main/torchvision/datasets/flowers102.py)
  needs the correct 102-name label order. The old list contained substitutions and
  duplicates. Recompute prior Flowers features/results made using that old list.
  Its official train set has 1020 images; use all without validation/test leakage.

The custom EuroSAT holdout and preprocessing choices are recorded conventions
that the team should include in the report. DTD uses its default official
partition 1. All four test subsets are fixed across all evaluations.

The low-forgetting check compares Pets accuracy at `theta_pets.pt` against the
full EuroSAT branch. Below 1 percentage point, the notebook adds two exploratory
lr-3e-5 runs in `sensitivity_lr3e-5/`, both reinitialized from the original Pets
checkpoint. It retains the four original checkpoints and reports all run times.
The frozen control uses the same altered learning rate. This exceeds four runs
only when the stated fallback is triggered. It does not guarantee greater
forgetting. The dominance ratio in Part C is undefined for total forgetting ≤1 pp.

If using sensitivity checkpoints with Part C, explicitly pair those two files with
the base Pets checkpoint; Part C's default paths point to the base runs. Report
the sensitivity protocol change separately rather than replacing the base data.

Completed runs can be reused after a disconnect only with matching protocol,
source, packages, parent and file hashes. GPU times from the original runs remain
intact. An incomplete training run restarts from its parent, not mid-epoch. A
checkpoint without matching sidecar is rejected instead of overwritten; move an
incomplete artifact aside or choose a new output folder. Keep the same environment
to resume. The notebook saves each completed run and table to `OUTPUT_DIR` incrementally.
Local Colab artifacts can be reused only while that runtime exists; a runtime
reset removes them. Optional Drive persistence supports reuse across runtime resets.

For slides, use actual table values; the forward-only signal time belongs to Part A
and remains unmeasured here. The notebook includes a 3–4 minute recording outline
for Thursday with Shikshith narrating. Record the real Colab session after obtaining
results. Phase II's 56 ordered pairs require eight datasets; these four provide
12 ordered pairs. The two Phase-I training pairs remain the agreed scope.

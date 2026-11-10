# Shreyas: run Part B on Kaggle and finish the handoff

Use the notebook on branch **`shreyas/phase1-finetuning`**, not an older download
from `main`. The same `B_finetune_Shreyas.ipynb` now supports Kaggle and Colab.
Training settings and the QuickGELU correction are identical across platforms.
No Colab Pro subscription, local NVIDIA GPU, Google Drive folder or GitHub token
is required inside Kaggle. Kaggle GPU access remains subject to your account's
available quota and hardware availability; check the current settings panel.

## 1. Get the current code

In the repository's VS Code terminal:

```bash
git fetch origin
git switch shreyas/phase1-finetuning
git pull --ff-only
```

Alternatively download `notebooks/B_finetune_Shreyas.ipynb` from this branch on
GitHub. You do not need to run local installation or training commands to use
the hosted notebook. The notebook already contains the required source and config.

## 2. Import and configure Kaggle

1. Sign in to Kaggle and create a new Python notebook.
2. Use **File → Import Notebook** to upload `B_finetune_Shreyas.ipynb`.
3. In the notebook's **Settings / Session options**, select an available **GPU**,
   for example **T4 ×2**. The code deliberately uses GPU 0 only. Leave the batch
   size, learning rate, image cap and epoch count unchanged. TPU is unsupported.
4. Turn **Internet on**. Installation, CLIP weights and the four torchvision
   dataset downloads require it. Follow any account-verification prompt Kaggle
   presents if GPU/Internet settings are unavailable; check quota in that panel.
5. Leave `USE_GOOGLE_DRIVE=False` and the Kaggle export default unchanged.

Kaggle downloads the full dataset archives despite the processing cap. No manually
attached Kaggle datasets are required. The environment's CUDA-enabled PyTorch is
used as installed; the first cell pins OpenCLIP and timm without replacing PyTorch.

## 3. Run a saved version

For the final run, use **Save Version → Save & Run All**. This executes a fresh
copy from top to bottom and produces a saved version with output files. Wait for
that version to finish successfully and inspect its logs. A code-only Quick Save
is insufficient evidence that checkpoint files have been saved.

You may first execute the setup/GPU-check cells interactively. They should print:

```text
Platform: Kaggle
Checkpoints and measured results: /kaggle/working/encoder-forgetting-signal-partb/artifacts/checkpoints_quickgelu
GPU: <the GPU assigned to your session>
```

The final saved run has its own execution: do not assume interactive files or
imports carry over. Each complete execution evaluates pretrained CLIP, trains
the four base runs, evaluates all four datasets, applies the low-forgetting check,
and exports artifacts. The second training cell reuses the just-finished base
runs rather than repeating them. If Pets forgetting after full EuroSAT is below
1 percentage point, it adds the matched lr-3e-5 EuroSAT pair in a separate folder.

Training/evaluation times are measured on Kaggle's assigned GPU; earlier Colab
timings do not establish Kaggle duration. Installs, downloads, queueing and export
add time outside the measured training loop. If installation asks for a kernel
restart, restart before training and rerun the notebook from the beginning.

## 4. Check completion and download

The final export cell checks the four checkpoint files, metric sidecars, scale
freeze flags and frozen-text flag. If the fallback was triggered, it also checks
for both additional sensitivity checkpoints. A failure should be resolved before
calling the run complete.

Open the successfully saved version's **Output** tab. Locate and download:

```text
part_b_artifacts.zip
```

It contains a `checkpoints/` folder with:

```text
theta_pets.pt
theta_pets_flowers.pt
theta_pets_eurosat.pt
theta_pets_eurosat_textfrozen.pt
theta_pets.json
theta_pets_flowers.json
theta_pets_eurosat.json
theta_pets_eurosat_textfrozen.json
training_results.csv
forgetting_check.json
protocol.json
environment.json
baseline.json
```

If triggered, `sensitivity_lr3e-5/` contains the extra pair and sidecars. The ZIP
can be several GB. Confirm the download finishes and the archive opens before
discarding the session. Dataset archives and temporary files are excluded.
If you already use the Kaggle CLI, saved output can also be downloaded with:

```bash
kaggle kernels output YOUR_USERNAME/YOUR_NOTEBOOK_SLUG -p downloads/kaggle-part-b
```

Save a local copy of the executed `.ipynb` with cell outputs and the saved Kaggle
version's URL. Stop the interactive GPU session when you are done.

## 5. Verify the results together

Send these small files here; the checkpoint binaries are unnecessary for review:

- `training_results.csv`
- `forgetting_check.json`
- `protocol.json`
- `environment.json`

We will confirm the corrected source fingerprint, matching settings/splits,
accuracy changes, feasibility times, freeze checks, skipped AMP updates and any
sensitivity runs. Keep the earlier QuickGELU-mismatched Colab results separate as
preliminary records. The fallback cannot guarantee a drop greater than one point;
small or negative forgetting must be reported honestly.

## 6. Complete the team deliverables

- Give Shikshith the extracted checkpoint folder and the current shared loader
  for Part C. Set Part C's checkpoint directory to this actual folder; its old
  Colab/Drive default is not the Kaggle export path. Explicitly select the
  sensitivity pair if the team uses those exploratory results.
- Parts A/C/D must use the corrected `src/common.py`, including QuickGELU, class
  names, prompt ensemble and EuroSAT holdout. Recompute incompatible old features.
- Put the verified accuracy/time table in Results and the measured GPU time in
  the feasibility box. Encoder contributions come from Part C's swap/Shapley
  results; Part A must measure its own forward-only signal time.
- Prepare Implementation and Phase-II timeline slides, then record the 3–4 minute
  demo with Shikshith narrating using the notebook's outline. Save the video/link.

## 7. Push the verified small results to GitHub

The Kaggle-support code is committed and pushed by Codex as part of preparing
this guide. After the GPU results are reviewed, keep the binaries in the exported
artifact location and copy only the verified small reports into the repository:

```text
results/finetune/kaggle_quickgelu/training_results.csv
results/finetune/kaggle_quickgelu/forgetting_check.json
results/finetune/kaggle_quickgelu/protocol.json
results/finetune/kaggle_quickgelu/environment.json
```

Then review and commit those specific files on the same branch:

```bash
git switch shreyas/phase1-finetuning
git status --short
git add results/finetune/kaggle_quickgelu/training_results.csv results/finetune/kaggle_quickgelu/forgetting_check.json results/finetune/kaggle_quickgelu/protocol.json results/finetune/kaggle_quickgelu/environment.json
git diff --cached --stat
git commit -m "Record verified Kaggle Phase-1 fine-tuning results"
git push
```

You can instead give me the verified files and ask me to commit/push them. The
checkpoint binaries stay outside Git. Open a pull request from this branch into
`main` for team review after the handoff; opening the PR does not merge it.

References: [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks),
[NVIDIA's Kaggle GPU setup](https://docs.nvidia.com/datascience/deployment/latest/platforms/kaggle/),
[official Kaggle CLI: versions and outputs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).

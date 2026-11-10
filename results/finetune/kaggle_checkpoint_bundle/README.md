# Uploaded Kaggle checkpoint bundle — binary verification

The extracted `checkpoints(kaggle)/` folder supplied by Shreyas was inspected on
10 November 2026. **All six checkpoint binaries passed verification.** Original
weights remain in that local folder and are ignored by Git. This directory
contains only the verification reports and byte-preserved small metadata files.

## What was checked

The CPU verification command was:

```bash
python scripts/verify_finetune_checkpoints.py "checkpoints(kaggle)" --report results/finetune/kaggle_checkpoint_bundle/checkpoint_verification.json
```

It did not download model weights/datasets, retrain models or use a GPU. It checked:

- Each checkpoint's actual SHA-256 against its accompanying JSON sidecar.
- Each branch's parent SHA-256 against the supplied original `theta_pets.pt`.
- All 302 state tensors per checkpoint against the expected ViT-B/32 shapes and
  float32 dtypes, using a model schema on the allocation-free `meta` device.
- All floating-point weights are finite.
- All saved logit-scale tensors are bitwise identical to the Pets reference.
- Both frozen-text controls retain every text-side tensor bitwise, including
  token embeddings, transformer parameters, normalization and projection.
- Every branch updated its vision tower; full updates also changed text weights.
- Sidecar configuration, learning rate, text-freeze setting, training image IDs,
  and metrics agree with this bundle's protocol and CSV. Baseline JSON agrees
  with the pretrained CSV row.
- The recorded training-source fingerprint matches the QuickGELU-corrected source.

[checkpoint_verification.json](checkpoint_verification.json) contains all binary
and sidecar hashes, parent links, sizes, tensor-change counts and verification
limits. The six binaries total **3,631,357,626 bytes (about 3.38 GiB)**.

| Checkpoint relative to the supplied folder | Changed vision tensors vs Pets | Changed text tensors vs Pets | Result |
|---|---:|---:|---|
| `theta_pets.pt` | 0 (reference) | 0 (reference) | Passed |
| `theta_pets_flowers.pt` | 152 | 149 | Passed |
| `theta_pets_eurosat.pt` | 152 | 149 | Passed |
| `theta_pets_eurosat_textfrozen.pt` | 152 | 0 | Passed; text bitwise identical |
| `sensitivity_lr3e-5/theta_pets_eurosat.pt` | 152 | 149 | Passed |
| `sensitivity_lr3e-5/theta_pets_eurosat_textfrozen.pt` | 152 | 0 | Passed; text bitwise identical |

The saved scale's CPU exponential is approximately 100.0000076, consistent with
the GPU-reported 100.0 within floating-point evaluation tolerance. The scale
parameters themselves are bitwise identical across the checkpoints.

Activations are not stored in state dictionaries; correct QuickGELU loading is
established by the recorded training source and shared loader. Original pretrained
weights were not downloaded for comparison, so Pets is the reference for the
binary freeze checks. No test accuracies or Shapley contributions were recomputed.

## This bundle's timing record

The supplied CSV has exactly the same non-timing values as the previously
[reviewed Kaggle CSV](../kaggle_quickgelu/training_results.csv), including all
accuracies, losses, skipped updates, sample counts, paths and freeze flags.
Its protocol, environment and forgetting-check files are byte-identical to those
previously supplied. However, the CSV and checkpoint sidecars record different
training and evaluation durations. Each record is therefore preserved separately.

[execution_comparison.json](execution_comparison.json) lists every changed field.
These are the same seed/configuration, not independent seed replications. Do not
combine one execution's sidecars with the other execution's timing CSV.

| Measurement | Earlier reviewed CSV | Uploaded checkpoint bundle |
|---|---:|---:|
| Four base training runs | 121.77 s | 117.63 s |
| Two sensitivity training runs | 26.69 s | 25.65 s |
| All six training runs | 148.47 s | 143.29 s |
| All seven evaluations | 273.38 s | 260.26 s |
| Recorded training + evaluation | 421.85 s | 403.55 s |

Both timing records exclude installation, downloads, queueing, model loading and
artifact writes. Label which execution supplies the feasibility numbers used in
slides/report. The earlier report remains a record of the earlier supplied CSV;
this bundle's own values are in [training_results.csv](training_results.csv).

## Results and next analysis

The accuracies and interpretation remain unchanged:

- Pets reference accuracy: **86.05%**.
- Base Flowers: Pets **83.65%**, forgetting **2.40 pp**.
- Base EuroSAT: Pets **85.35%**, forgetting **0.70 pp**; its normalized dominance
  ratio is undefined under the agreed threshold.
- Base frozen-text EuroSAT: Pets **86.60%**, an improvement of **0.55 pp**.
- Exploratory lr-3e-5 EuroSAT: Pets **32.30%**, forgetting **53.75 pp**.
- Exploratory lr-3e-5 frozen-text EuroSAT: Pets **59.25%**, forgetting **26.80 pp**.

There is no checkpoint-integrity reason to retrain. The next task is Part C's
forward-pass encoder-swap/Shapley evaluation using these exact files. It requires
Pets test images and a suitable execution environment. This local verification
host had neither CUDA nor an available MPS backend; it performed CPU integrity
checks only.

Shikshith should use branch `shreyas/phase1-finetuning`, whose Part C loader
delegates to the corrected shared loader. In a CUDA environment where the folder
is available under the same relative name, the base command is:

```bash
python scripts/run_swap.py --ckpt-dir "checkpoints(kaggle)" --out-dir results/swap/base --device cuda
```

Use the separate [sensitivity-pair invocation](../kaggle_quickgelu/README.md#part-c-checkpoint-handoff)
with `CKPT_DIR` set to this actual folder. It pairs the original Pets checkpoint
with the two files under `sensitivity_lr3e-5/`. Base and sensitivity outputs should
remain separate.

Expected checks: `A_pp=86.05%` for every pair, `A_ff` equal to the relevant Pets
accuracy above within numerical tolerance, frozen-text `F_L` and interaction zero
within tolerance, and `F_V + F_L = A_pp - A_ff`. Actual dominance values require
the encoder-swap predictions and are not inferred from the control comparison.

The large checkpoint files must still be accessible to the person/environment
running Part C. Keeping this verification report in GitHub does not upload the
weights or mount the local folder in Kaggle/Colab.

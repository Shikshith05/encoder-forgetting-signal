# Experiment: per-encoder gradient interference (exploratory)

**Purpose.** This experiment tests whether a cheap signal computed *before* fine-tuning can predict which encoder dominates forgetting when a CLIP ViT-B/32 model trained on an old domain O is fine-tuned on a new domain N. The signal is first-order gradient interference, computed separately for the vision tower and the text tower. For each grid12 source checkpoint theta_O we compute the training-loss gradient on O's and on N's TRAIN split. Interference is then I_E = -dot(g_O_E, sign(g_N_E)) (the AdamW first step; this is the primary signal) and I_E = -dot(g_O_E, g_N_E) (SGD), for E in {V, L}. We compare it with the true per-encoder forgetting (F_V, F_L, D) already measured by the grid12 swap/Shapley experiment, and with the gap_v - gap_l baseline. No fine-tuning is done here, and no ground truth is recomputed.

## How to run

Use the CUDA environment (`D:\CAPSTONE\grid12\venv`):

```
cd experiments/gradient_interference
D:\CAPSTONE\grid12\venv\Scripts\python grad_signal.py              # 12 pairs, resumable
D:\CAPSTONE\grid12\venv\Scripts\python grad_signal.py --sanity --batch 64   # determinism check
D:\CAPSTONE\grid12\venv\Scripts\python analyse_grad.py             # CPU only, ~1 min
```

`grad_signal.py` first tests whether a float32 gradient set at batch 64 fits on the GPU and falls back to batch 32 if it does not. Pass `--batch` to skip this test. The script writes `grad_signals.csv` after every pair and skips pairs that are already there.

## Inputs

These live outside the repo and are not committed:

- `D:\CAPSTONE\grid12\ckpt\theta_<O>.pt` for O in OxfordIIITPet, Flowers102, EuroSAT, DTD (grid12 source checkpoints).
- The repo's `data/` folder (TRAIN splits only, via `src.common.get_dataset_loader`, seed 0).
- `ground_truth_grid12.csv` in this folder, copied unchanged from `D:\CAPSTONE\grid12\results\grid12_results.csv`.

## Outputs (this folder)

| file | content |
|---|---|
| `grad_signals.csv` | one row per (O, N): I_sgd_V/L, I_adam_V/L, cos_V/L, normN_V/L, normO_V/L, batch, K, losses, timings |
| `sanity.json` | max abs difference between two reruns of the theta_EuroSAT-on-EuroSAT gradient set |
| `analysis.md` | method, 12-row table, comparison with the gap baseline, leave-one-source-out results, interpretation, caveats |
| `analysis_table.csv`, `analysis_summary.json` | the numbers behind `analysis.md`, in machine-readable form |
| `runtime.json`, `run_pairs.log`, `run_sanity.log` | measured runtimes and logs |

## Runtime measured

Measured on an NVIDIA T1200 Laptop GPU (4 GB) with torch 2.6.0+cu124, float32, batch 64, K = 4:

- 12 pairs: **25.1 min** (1508 s), including the batch-size test. That is 16 gradient sets of 26 to 217 s each; the Flowers102 sets (1020 prompts) are the slowest. Peak GPU memory was about 3.0 GB.
- Sanity check: 1.7 min.
- Analysis: under 1 min on the CPU.

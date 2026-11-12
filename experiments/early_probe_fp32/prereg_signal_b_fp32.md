# Pre-registration: signal B2-fp32, early probe without skipped steps (PW26_VK_06)

Written Thu 12 Nov 2026, BEFORE any B2-fp32 run. Copied unchanged into
experiments/early_probe_fp32/ (branch exp/early-probe-fp32).

## Why this signal

In signal B (exp/early-probe, prereg_signals_v2.md section 4) the fp16 GradScaler skipped
3-5 of the 8 optimizer steps in every pair, so the probe made only 3-5 real updates.
B2-fp32 is the same probe trained in fp32, so all 8 updates happen. Everything not listed
under "Changes from B" is identical to B.

## Changes from B (the only ones)

1. Training precision: fp32, no autocast, no GradScaler.
2. Training loop: the repo `train_model` always uses autocast + GradScaler and src/ is not
   modified, so the experiment folder has its own loop, `train_fp32`. It copies train_model's
   logic exactly (seed_everything(cfg seed = 0); repo `configure_trainable(model, False)`, so both
   encoders are trainable and logit_scale is frozen; model.train(); visual grad checkpointing on;
   AdamW over the trainable params, lr 3e-5, weight_decay 0.1; zero_grad -> encode_image ->
   normalise -> prompt_features(checkpoint_text=True) -> logits = logit_scale.exp() * img @ txt.T
   -> cross-entropy -> non-finite check -> backward -> step; logit_scale-unchanged check;
   model.eval() and grad checkpointing off at the end). Only autocast and GradScaler are removed.
3. Memory fallback: the run first tries batch 64 on the first pair (OxfordIIITPet -> Flowers102,
   which has the largest prompt set, 1020 prompts, so it is the worst case for memory). If that
   runs out of GPU memory, ALL 12 pairs use gradient accumulation 4 x 16. Each micro-batch loss
   is divided by 4 and the optimizer steps once per 64 images, which is the same update as
   batch 64 up to float rounding. The mode used is recorded in every row
   (`mode` = "batch64" or "accum4x16"). Estimate, not measured: fp32 AdamW needs about
   2.4 GB for weights, gradients and Adam states, and the earlier fp32 gradient sets peaked at
   3.0 GB without Adam states. So 4 x 16 is likely on the 4 GB T1200.
4. Update count: the number of real optimizer updates is logged per pair and must be 8.
   If a pair has fewer than 8 (e.g. a non-finite loss stops it), its D is undefined and it
   counts as WRONG.

## Unchanged from B

- Start: theta_O (D:\CAPSTONE\grid12\ckpt\theta_<O>.pt), loaded with repo `load_clip`.
- Training data: the first 8 batches of 64 (first 512 images) of N's TRAIN split, same order as
  B (repo `get_dataset_loader(N, ..., "train")`, seed 0, shuffle=False, Subset(range(512))).
- Probe set: the first 512 images of O's TRAIN split (batch 64). Features from the repo
  `extract_image_features` / `extract_text_features`, for theta_O and for the probe model.
  O's TEST split is never used.
- Swap: repo `swap_from_features(I_p, I_f, T_p, T_f, labels, ..., scale=100, threshold=0.9)`.
- Results are saved after every pair (resumable) and the runtime is measured.

## Scores

- PRIMARY: D_B2fp32 = (M_F_V - M_F_L) / (|M_F_V| + |M_F_L|), from the linear-margin Shapley split.
  Prediction: "vision" if D > 0. A score that is 0 or NaN counts as wrong.
- Secondary (reported only, cannot rescue a failed primary):
  - accuracy-based probe D = (F_V - F_L) / (|F_V| + |F_L|), from the same swap;
  - Spearman of probe M_total vs true total. NOTE: this relation was already seen in B on these
    same 12 pairs (rho +0.71, p = 0.014), so it is NOT independent evidence.

## Evaluation (same signal_eval.py as A, B, C)

Ground truth: grid12_results.csv (12 pairs, lr 3e-5, seed 0). Metrics: sign accuracy /12, AUROC
for D > 0, Spearman vs true D with two-sided permutation p (10,000 permutations, seed 0),
leave-one-source-out sign accuracy.

DEV PASS = primary sign accuracy >= 11/12 OR primary AUROC > 0.833.
A dev pass is "promising on dev only"; proposing it as working needs a held-out run, which
needs the user's OK first.

Comparison (already measured):
| signal | sign acc | AUROC | Spearman vs D (perm p) |
|---|---|---|---|
| gap_v - gap_l (threshold tuned on same data) | 10/12 | 0.833 | +0.501 (p = 0.10) |
| gradient interference (Adam) | 7/12 | 0.486 | +0.018 (p = 0.96) |
| A Fisher x consistency | 7/12 | 0.778 | +0.573 (p = 0.056) |
| B early probe, fp16 (3-5 real updates) | 8/12 | 0.667 | +0.259 (p = 0.41) |
| C fragility | 6/12 | 0.167 | -0.626 (p = 0.036) |

Caveat stated in advance: B2-fp32 is a second attempt at the probe idea on the SAME 12
development pairs, after B's result was seen. This raises the chance of a lucky pass, so even a
dev pass is weaker evidence than a first attempt.

## Run

One background job: the probe, then analyse.py. Branch exp/early-probe-fp32, files only in
experiments/early_probe_fp32/. No commit until the user confirms.
Estimated GPU time: 90 to 150 minutes (estimate; fp32 and possible accumulation are slower than B's
74 min).

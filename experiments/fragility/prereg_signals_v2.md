# Pre-registration: second-order and probe signals (PW26_VK_06)

Draft written Wed 11 Nov 2026, 22:55 IST. FINAL version written Wed 11 Nov 2026, BEFORE any of
these signals were computed (approved to run by Shikshith: "run A, B, C in that order").
This file is copied unchanged into each experiment folder.

Change from the draft (made before any run): signal B uses 8 training steps, not 4. Reason:
the grid12 source runs logged 3-5 GradScaler-skipped optimizer steps per epoch (fp16 loss-scale
calibration, almost certainly at the start), so 4 steps would leave about 1 real update.

## 1. What is being predicted

Ground truth: `D:\CAPSTONE\grid12\results\grid12_results.csv` (copied as ground_truth_grid12.csv).
12 ordered pairs O->N of {OxfordIIITPet, Flowers102, EuroSAT, DTD}, lr 3e-5, seed 0.
Target: sign of true D (D > 0 = vision encoder forgets more). 6 of 12 pairs have D > 0.

Baselines (already measured; reported next to every signal):
| baseline | sign acc | AUROC | Spearman vs D (perm p) |
|---|---|---|---|
| gap_v - gap_l | 10/12 (threshold tuned on the same data) | 0.833 | +0.501 (p = 0.10) |
| gradient interference (Adam) | 7/12 (threshold 0) | 0.486 | +0.018 (p = 0.96) |

## 2. Rules common to A, B, C

- Model: CLIP ViT-B-32, "openai", QuickGELU (repo `load_clip`), logit scale fixed at 100, frozen.
- Data: repo `get_dataset_loader(D, preprocess, cfg, "train")`, seed 0, shuffle=False, so
  "first n images" = first n of the seed-0 permutation of the TRAIN split. O's TEST split is
  never used to compute any signal.
- Loss (where gradients are needed) = the training loss: image vs ALL class prototypes
  (10 config templates, `prompt_features(..., checkpoint_text=True)`), logits = 100 * img @ txt.T,
  cross-entropy. `seed_everything(0)` before every gradient set or run.
- Encoders: V = parameters named `visual.*` (87,849,216); L = all other parameters except
  logit_scale (63,428,096).
- Each signal has ONE primary score. Only the primary decides pass/fail. Secondary scores are
  reported but cannot rescue a failed primary.
- Prediction: "vision" if primary score > 0. No tuned threshold. A score that is exactly 0 or
  undefined (NaN) counts as a WRONG prediction, and enters the AUROC as 0.
- Reported for every score: sign accuracy /12; AUROC for (D > 0); Spearman rho of score vs true D
  with two-sided permutation p (10,000 permutations, numpy seed 0); leave-one-source-out
  (sign accuracy on the 3 pairs of each held-out O; nothing is fitted, so no refitting needed).
  Extra Spearman (reported, not used for pass): per-encoder score vs F_V and vs F_L.
- DEV PASS = primary sign accuracy >= 11/12 OR primary AUROC > 0.833.
- Dev pass => "promising on dev only". PROPOSED as working only after a held-out run with new
  old-domains under a rule written before that run. Held-out runs need the user's OK first.
- Results are saved after every pair (resumable). One GPU job at a time. Runtimes measured.

## 3. Signal A: Fisher x Adam-consistency (no training). Branch exp/fisher-consistency

- Fisher of the old task at theta_O (float32, no autocast):
  F_i = mean over the first 16 mini-batches of size 8 (first 128 images of O's TRAIN split)
  of (mini-batch-mean gradient_i)^2. (Mini-batch approximation of the diagonal empirical Fisher.)
- Consistency of the new task at theta_O (float32): over the first K = 8 batches of size 64
  (first 512 images of N's TRAIN split), c_i = |mean over the 8 batches of sign(g_N,i,batch)|.
  c_i is in {0, 1/8, ..., 1}.
- S_E = sum over params i in encoder E of F_i * c_i^2.
- PRIMARY: D_A = (S_V - S_L) / (S_V + S_L).
- Secondary (reported only):
  A2 parameter-count normalised: s_E = S_E / n_E; D = (s_V - s_L) / (s_V + s_L).
  A3 Fisher only: sum_E F_i  (same for all 3 pairs of an O).
  A4 consistency only: sum_E c_i^2.
- Estimated GPU time: 45 to 80 minutes (estimate).

## 4. Signal B: early probe (8 training steps). Branch exp/early-probe

- Start from theta_O; train on N with the unchanged repo `train_model` (AdamW, lr 3e-5,
  wd 0.1, batch 64, autocast + GradScaler, cfg seed 0) on a loader that contains only the first
  8 batches (first 512 images) of N's TRAIN loader in the same order, i.e. exactly the first
  8 of the ~32 steps of the real run (25% of the epoch). Skipped steps are recorded.
- Probe set: first 512 images of O's TRAIN split (batch 64). Features with the repo
  `extract_image_features` / `extract_text_features` (float32), for theta_O and the probe model.
- Swap: repo `swap_from_features(I_p, I_f, T_p, T_f, labels, ..., scale=100, threshold=0.9)`.
- PRIMARY: D_B = (M_F_V - M_F_L) / (|M_F_V| + |M_F_L|), the linear-margin Shapley split.
- Secondary (reported only): B2 accuracy-based probe D_acc = (F_V - F_L)/(|F_V| + |F_L|)
  from the same swap (NaN if both are 0).
- If all 8 steps are skipped (no update), D_B is undefined -> counted wrong.
- B uses a little training: it is an "early warning", not strictly pre-fine-tuning.
- Estimated GPU time: 45 to 90 minutes (estimate).

## 5. Signal C: old-domain fragility (forward only). Branch exp/fragility

- theta_O, first 512 images of O's TRAIN split. Image features I (512 x 512, unit norm) and
  class prototypes T (C x 512, unit norm), repo extract functions.
- Noise: x' = normalize(x + e), e ~ N(0, (sigma^2 / 512) I), so ||e|| is about sigma.
  20 draws; a fresh numpy default_rng(0) for each (O, sigma); in each draw the image noise is
  drawn first, then the text noise. Accuracy = argmax of I @ T.T vs labels.
- drop_V = clean acc - mean acc with only I perturbed; drop_L = clean acc - mean acc with only
  T perturbed.
- PRIMARY: sigma = 1.0, D_C = (drop_V - drop_L) / (|drop_V| + |drop_L|).
- Secondary (reported only): sigma = 0.25 and sigma = 0.5.
- C depends only on O: same score for the 3 pairs of a source -> sign ceiling 10/12 here, so C
  can pass only via AUROC (max 0.917). Known geometric bias, stated in advance: perturbing each
  of the C prototypes independently probably hurts more than perturbing the image, so D_C may be
  negative for every O (sign accuracy would then be 6/12). AUROC is unaffected by this offset.
- Estimated GPU time: about 10 minutes (estimate).

## 6. Held-out confirmation (only for a signal that passes dev; asks user first)

New old-domains (candidates FGVCAircraft, Food101): new source models + new ground-truth pairs.
Rough estimate 3 to 4 hours of laptop GPU. Exact pairs, rule and runtime are written and shown
to the user before any launch.

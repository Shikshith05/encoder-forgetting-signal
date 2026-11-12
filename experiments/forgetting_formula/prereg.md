# Pre-registration: forgetting formula (PW26_VK_06)

Written and committed BEFORE any new data was downloaded, any model trained, or any signal computed
for CIFAR-100 / GTSRB. Model: CLIP ViT-B/32 (OpenAI, QuickGELU), repo `load_clip`.

## Sets
- **Development set:** the existing 12 pairs in `D:\CAPSTONE\grid12\results\grid12_results.csv`
  (A, B in {OxfordIIITPet, Flowers102, EuroSAT, DTD}).
- **Test set:** the 18 new ordered pairs that involve CIFAR-100 or GTSRB as A or B (6 datasets, 30 pairs total).
- **All constants are fitted on the development set only.**
- Target: `total` = accuracy drop (points) on A's TEST split, theta_A -> theta_{A->B} (grid12 protocol:
  `swap_from_features`, source models lr 1e-5, pairs lr 3e-5, batch 64, 1 epoch, logit scale frozen, seed 0).
- Fallback: if GTSRB cannot be obtained, SVHN replaces it (reported under DEVIATIONS).

## Definitions (no signal reads A's TEST split)
- `U_B = 1 - thA_zs_acc_B_train / 100` (zero-shot accuracy of theta_A on B's train images).
- `Acc_A = thA_zs_acc_A_train` (theta_A accuracy on A's train images, %).
- `S_intra_A = thA_S_intra_A`, `margin_A = thA_margin_A` (theta_A, A's train images).
- `steps` = number of optimiser steps on B = `len(train loader)` at the batch actually used.
- Everything else follows `parameter_dictionary.md` (branch cross-modal-implementation):
  `pre_*` = pretrained CLIP, `thA_*` = theta_A; train splits, <= 2000 images, seed-0 permutation.

## Formulas (prediction of `total`, in accuracy points)
| id | formula | fitted constants |
|---|---|---|
| F0 | mean of dev `total` | 1 |
| F1 IIMM (paper) | a + b * pre_IIMM_B | OLS |
| F2 | a + b * pre_zs_acc_B_train | OLS |
| F3 weighted IIMM | a + b1 * pre_S_inter_B + b2 * pre_S_intra_B | OLS |
| F4 Sekeh & Wisell Thm 4.6 | b1 * thA_Con_IT + b2 * thA_Con_TI + a | OLS |
| **F5 PRIMARY** | Acc_A * (1 - exp(-k * U_B * S_intra_A)) | k |
| F6 | Acc_A * (1 - exp(-k * U_B / margin_A)) | k |
| F7 | Acc_A * (1 - exp(-k * U_B * S_intra_A * steps / 32)) | k |
| F8 | a + b1 * thA_IIMM_B + b2 * thA_CMS | OLS |

k (F5-F7): minimise the sum of squared errors on dev over the log grid
`k = 10**linspace(-3, 3, 6001)` (no bounded scalar optimiser). The chosen k is the grid argmin; if it lies
on a grid edge this is reported.

## Primary test
F5 vs F1 on the 18 test pairs. Report MAE, R^2 (1 - SSE/SST around the test mean), Spearman rho, and a
paired bootstrap 95% CI (percentile) of MAE(F1) - MAE(F5): 10,000 resamples of the 18 pairs with
replacement, `numpy.random.default_rng(0)`.

**WIN = F5's test MAE < F1's test MAE AND the CI excludes 0.** Otherwise NO-WIN.

## Secondary (reported only, no verdict)
- All formulas F0-F8 on the 18 test pairs (MAE, R^2, Spearman).
- Leave-one-dataset-out over all 30 pairs: for each of the 6 datasets, fit on pairs not involving it,
  predict the pairs involving it (as A or B); report each fold's MAE and the mean of the 6 fold MAEs.

## Caveat
Single model (CLIP ViT-B/32), single seed, 6 datasets, pairs share datasets (not independent).

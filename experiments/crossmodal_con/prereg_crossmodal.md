# Pre-registration: cross-modal contribution (Con / CMS) and IIMM signals (PW26_VK_06)

Written Thu 12 Nov 2026, BEFORE any correlation, regression or CV error was computed.
Steps 1-2 (feature bank, parameter table) may already have run; they only compute and save
values and test nothing. Status: APPROVED 12 Nov 2026 by PI before any test was run: (1) the three
chat-stated baselines are recomputed on the same folds and only recomputed values are used; (2) "better"
= lower MAE than both baselines in BOTH CV schemes; (3) cross_inter / Con_TI tested once.

Caveat (word for word): "Exploratory: 12 non-independent pairs, single seed, a dev set reused by
earlier signals. Nothing here is claimed as a working predictor; the result only decides what is
pre-registered for Phase-II (30+ pairs)."

## 1. Data and target

- Ground truth: `D:\CAPSTONE\grid12\results\grid12_results.csv`, 12 ordered pairs A -> B
  (A = old = O, B = new = N) of {OxfordIIITPet, Flowers102, EuroSAT, DTD}.
- Target: `total` (accuracy drop on A's test split after training on B, pp).
- Signals: `experiments/crossmodal_con/pair_parameters.csv` (definitions in
  `parameter_dictionary.md`). Models: pretrained CLIP ViT-B/32 OpenAI QuickGELU (`pre_`) and
  theta_A = `D:\CAPSTONE\grid12\ckpt\theta_<A>.pt` (`thA_`). Image-based signals use TRAIN images
  (repo loader, seed 0, <= 2000); only `zs_acc_B_test` uses B's TEST split. A's TEST split never
  feeds a signal.

## 2. Primary tests (Bonferroni alpha = 0.05 / 3 = 0.0167)

| id | test | statistic |
|---|---|---|
| P1 | theory score | Spearman(thA_CMS, total), two-sided, 10,000-permutation p, numpy seed 0 |
| P2 | IIMM extension | Spearman(thA_IIMM_B, total), same test |
| P3 | IIMM + cross-modal (main objective) | OLS with intercept: total ~ thA_IIMM_B + thA_CMS |

P1/P2 are "significant" if p < 0.0167. Sign is reported but not pre-specified (two-sided).

P3 protocol:
- Two cross-validation schemes: leave-one-pair-out (LOPO, 12 folds) and
  leave-one-old-domain-out (LOODO, 4 folds of 3 pairs sharing A). In each fold, the model is fitted
  on the training pairs only.
- Held-out MAE = mean |prediction - total| over the 12 held-out predictions; held-out
  R^2 = 1 - sum(err^2) / sum((total - mean(total))^2) over the same 12 predictions.
- Baselines, computed with the same code and folds: (i) predict the training-fold mean of total;
  (ii) OLS total ~ zero-shot accuracy on B (`grid12_zs_N`, the already-measured pretrained zero-shot
  accuracy on B's test split).
- The values you gave (mean predictor 15.2 pp, zero-shot alone 13.8 pp held-out MAE) are shown next
  to them. **No file in the repo, its branches, or D:\CAPSTONE contains these two numbers**, so
  the comparison uses the recomputed values, and any difference is reported.
- "Better" = held-out MAE lower than BOTH baselines, in BOTH LOPO and LOODO.

## 3. Already-measured baselines (shown, not recomputed as signals)

| signal | rho vs total | perm p | source |
|---|---|---|---|
| zero-shot accuracy on B (pretrained) | -0.583 | 0.0553 | grid12/results/analysis.md, test 5c |
| IIMM on B (pretrained) | -0.086 | 0.8119 | grid12/results/analysis.md, test 5d |
| Gap_V | -0.184 | 0.5824 | grid12/results/analysis.md, test 5a |
| Gap_L | -0.353 | 0.2659 | grid12/results/analysis.md, test 5b |

Note: grid12.py (`step_signals`) computed gap_v/gap_l/zs_N/iimm_N from TEST features, so the
grid12 Gap_V baseline used A's TEST images. It is shown for reference only. This experiment's
`pre_Gap_V` / `thA_Gap_V` use TRAIN images.

The zero-shot within-old-domain p = 0.013 you quoted is also in no file. It is recomputed in
Step 4 with the within-A permutation defined below, and only the recomputed value is reported.

## 4. Exploratory analysis (everything else)

- Parameters: every `pre_*` and `thA_*` column of pair_parameters.csv. Dropped before testing:
  columns constant across the 12 pairs, and columns whose 12 values exactly equal those of an
  earlier column (by construction `cross_inter` == `Con_TI`, and `pre_n_classes_B` ==
  `thA_n_classes_B`). The dropped columns are listed in analysis.md.
- Targets: total, F_V, F_L, D.
- For each (parameter, target): Spearman rho; two-sided permutation p (10,000 permutations of the
  target, numpy seed 0); within-old-domain permutation p (the target permuted only within the 3
  pairs that share A, 10,000, seed 0). A parameter that only depends on A (e.g. pre_IIMM_A) gets
  within-A p = 1 by construction.
- Benjamini-Hochberg adjustment of the global permutation p over the WHOLE exploratory list
  (all parameters x 4 targets). The table is sorted by adjusted p.
- P1/P2 parameters also appear in this list, marked; their primary verdict comes only from section 2.

## 5. Output

`experiments/crossmodal_con/analysis.md`: primary table, exploratory table (sorted, BH-adjusted),
per-pair table of the main parameters, measured runtimes, and a 6-line summary of P1-P3.

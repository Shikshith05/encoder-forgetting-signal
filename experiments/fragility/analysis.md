# Signal C: old-domain fragility (forward only)

Pre-registration: `prereg_signals_v2.md` (this folder; written before the run).
Prediction: vision if score > 0 (0/NaN = wrong). DEV PASS = primary sign accuracy >= 11/12 OR primary AUROC > 0.833.

## Per-pair table

| O | N | drop_V | drop_L | D_C | true F_V | true F_L | true D | correct? |
|---|---|---:|---:|---:|---:|---:|---:|:-:|
| OxfordIIITPet | Flowers102 | 11.77 | 34.01 | -0.486 | 4.50 | 23.30 | -0.676 | yes |
| OxfordIIITPet | EuroSAT | 11.77 | 34.01 | -0.486 | 16.25 | 38.40 | -0.405 | yes |
| OxfordIIITPet | DTD | 11.77 | 34.01 | -0.486 | 11.87 | 36.23 | -0.506 | yes |
| Flowers102 | OxfordIIITPet | 23.38 | 44 | -0.306 | 18.25 | 43.90 | -0.413 | yes |
| Flowers102 | EuroSAT | 23.38 | 44 | -0.306 | 44.57 | 35.33 | +0.116 | no |
| Flowers102 | DTD | 23.38 | 44 | -0.306 | 23.65 | 46.25 | -0.323 | yes |
| EuroSAT | OxfordIIITPet | 2.744 | 28.26 | -0.823 | 36.10 | 9.70 | +0.576 | no |
| EuroSAT | Flowers102 | 2.744 | 28.26 | -0.823 | 25.42 | 0.12 | +0.990 | no |
| EuroSAT | DTD | 2.744 | 28.26 | -0.823 | 62.38 | 3.52 | +0.893 | no |
| DTD | OxfordIIITPet | 13.05 | 47.36 | -0.568 | 9.10 | 32.50 | -0.563 | yes |
| DTD | Flowers102 | 13.05 | 47.36 | -0.568 | 30.74 | 10.16 | +0.503 | no |
| DTD | EuroSAT | 13.05 | 47.36 | -0.568 | 41.06 | 23.09 | +0.280 | no |

## Secondary scores per pair (reported only)

| O | N | D_C_s0.25 | correct? | D_C_s0.5 | correct? |
|---|---|---:|:-:|---:|:-:|
| OxfordIIITPet | Flowers102 | -0.418 | yes | -0.542 | yes |
| OxfordIIITPet | EuroSAT | -0.418 | yes | -0.542 | yes |
| OxfordIIITPet | DTD | -0.418 | yes | -0.542 | yes |
| Flowers102 | OxfordIIITPet | -0.621 | yes | -0.498 | yes |
| Flowers102 | EuroSAT | -0.621 | no | -0.498 | no |
| Flowers102 | DTD | -0.621 | yes | -0.498 | yes |
| EuroSAT | OxfordIIITPet | -0.605 | no | -0.844 | no |
| EuroSAT | Flowers102 | -0.605 | no | -0.844 | no |
| EuroSAT | DTD | -0.605 | no | -0.844 | no |
| DTD | OxfordIIITPet | -0.923 | yes | -0.759 | yes |
| DTD | Flowers102 | -0.923 | no | -0.759 | no |
| DTD | EuroSAT | -0.923 | no | -0.759 | no |

## Metrics vs baselines

| score | sign acc | AUROC (D>0) | Spearman vs D (perm p) | dev pass? |
|---|:-:|:-:|:-:|:-:|
| D_C (PRIMARY) | 6/12 | 0.167 | -0.626 (p=0.0357) | no |
| D_C_s0.25 (secondary) | 6/12 | 0.333 | -0.151 (p=0.6548) | - |
| D_C_s0.5 (secondary) | 6/12 | 0.167 | -0.626 (p=0.0357) | - |
| baseline: gap_v - gap_l (threshold tuned on same data) | 10/12 | 0.833 | +0.501 (p=0.10) | - |
| baseline: gradient interference, Adam (exp/gradient-interference) | 7/12 | 0.486 | +0.018 (p=0.96) | - |

Spearman p: two-sided permutation, 10,000 permutations, seed 0.

## Leave-one-source-out (sign accuracy on the 3 pairs of each held-out O; nothing fitted)

| held-out O | D_C | D_C_s0.25 | D_C_s0.5 |
|---|:-:|:-:|:-:|
| OxfordIIITPet | 3/3 | 3/3 | 3/3 |
| Flowers102 | 2/3 | 2/3 | 2/3 |
| EuroSAT | 0/3 | 0/3 | 0/3 |
| DTD | 1/3 | 1/3 | 1/3 |

## Extra Spearman (reported only, not used for pass)

| signal column | ground truth | Spearman (perm p) |
|---|---|:-:|
| drop_V | F_V | -0.065 (p=0.8609) |
| drop_L | F_L | +0.389 (p=0.2177) |

## Verdict

**DEV FAIL: negative result** (primary D_C: 6/12, AUROC 0.167).

## Caveats

> Exploratory: 12 pairs from 4 datasets (not independent), single seed. These 12 pairs are a development set already used by the gap and gradient-interference signals. A dev pass is 'promising on dev only', not a confirmation.

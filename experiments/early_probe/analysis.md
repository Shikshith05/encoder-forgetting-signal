# Signal B: early probe (8 training steps)

Pre-registration: `prereg_signals_v2.md` (this folder; written before the run).
Prediction: vision if score > 0 (0/NaN = wrong). DEV PASS = primary sign accuracy >= 11/12 OR primary AUROC > 0.833.

## Per-pair table

| O | N | M_F_V | M_F_L | effective_updates | D_B | true F_V | true F_L | true D | correct? |
|---|---|---:|---:|---:|---:|---:|---:|---:|:-:|
| OxfordIIITPet | Flowers102 | 0.02687 | 0.03499 | 3 | -0.131 | 4.50 | 23.30 | -0.676 | yes |
| OxfordIIITPet | EuroSAT | 0.02134 | 0.09053 | 4 | -0.619 | 16.25 | 38.40 | -0.405 | yes |
| OxfordIIITPet | DTD | 0.02684 | 0.06712 | 5 | -0.429 | 11.87 | 36.23 | -0.506 | yes |
| Flowers102 | OxfordIIITPet | 0.0303 | 0.03895 | 4 | -0.125 | 18.25 | 43.90 | -0.413 | yes |
| Flowers102 | EuroSAT | 0.04521 | 0.09267 | 3 | -0.344 | 44.57 | 35.33 | +0.116 | no |
| Flowers102 | DTD | 0.02915 | 0.07155 | 4 | -0.421 | 23.65 | 46.25 | -0.323 | yes |
| EuroSAT | OxfordIIITPet | 0.01033 | 0.02311 | 4 | -0.382 | 36.10 | 9.70 | +0.576 | no |
| EuroSAT | Flowers102 | 0.02383 | 0.004652 | 3 | +0.673 | 25.42 | 0.12 | +0.990 | yes |
| EuroSAT | DTD | 0.04619 | 0.01505 | 4 | +0.509 | 62.38 | 3.52 | +0.893 | yes |
| DTD | OxfordIIITPet | 0.01569 | 0.01227 | 4 | +0.122 | 9.10 | 32.50 | -0.563 | no |
| DTD | Flowers102 | 0.01351 | 0.04385 | 4 | -0.529 | 30.74 | 10.16 | +0.503 | no |
| DTD | EuroSAT | 0.03966 | 0.03835 | 3 | +0.017 | 41.06 | 23.09 | +0.280 | yes |

## Secondary scores per pair (reported only)

| O | N | D_B2_acc | correct? |
|---|---|---:|:-:|
| OxfordIIITPet | Flowers102 | -0.282 | yes |
| OxfordIIITPet | EuroSAT | -0.511 | yes |
| OxfordIIITPet | DTD | +0.287 | no |
| Flowers102 | OxfordIIITPet | +0.411 | no |
| Flowers102 | EuroSAT | -0.098 | no |
| Flowers102 | DTD | -0.065 | yes |
| EuroSAT | OxfordIIITPet | -0.337 | no |
| EuroSAT | Flowers102 | +0.946 | yes |
| EuroSAT | DTD | +0.828 | yes |
| DTD | OxfordIIITPet | -0.112 | yes |
| DTD | Flowers102 | +0.075 | yes |
| DTD | EuroSAT | +0.688 | yes |

## Metrics vs baselines

| score | sign acc | AUROC (D>0) | Spearman vs D (perm p) | dev pass? |
|---|:-:|:-:|:-:|:-:|
| D_B (PRIMARY) | 8/12 | 0.667 | +0.259 (p=0.4125) | no |
| D_B2_acc (secondary) | 8/12 | 0.722 | +0.476 (p=0.1248) | - |
| baseline: gap_v - gap_l (threshold tuned on same data) | 10/12 | 0.833 | +0.501 (p=0.10) | - |
| baseline: gradient interference, Adam (exp/gradient-interference) | 7/12 | 0.486 | +0.018 (p=0.96) | - |

Spearman p: two-sided permutation, 10,000 permutations, seed 0.

## Leave-one-source-out (sign accuracy on the 3 pairs of each held-out O; nothing fitted)

| held-out O | D_B | D_B2_acc |
|---|:-:|:-:|
| OxfordIIITPet | 3/3 | 2/3 |
| Flowers102 | 2/3 | 1/3 |
| EuroSAT | 2/3 | 2/3 |
| DTD | 1/3 | 3/3 |

## Extra Spearman (reported only, not used for pass)

| signal column | ground truth | Spearman (perm p) |
|---|---|:-:|
| M_F_V | F_V | +0.406 (p=0.1900) |
| M_F_L | F_L | +0.713 (p=0.0123) |
| M_total | total | +0.706 (p=0.0140) |
| probe_total | total | +0.650 (p=0.0241) |

## Verdict

**DEV FAIL: negative result** (primary D_B: 8/12, AUROC 0.667).

## Caveats

> Exploratory: 12 pairs from 4 datasets (not independent), single seed. These 12 pairs are a development set already used by the gap and gradient-interference signals. A dev pass is 'promising on dev only', not a confirmation.

# Gradient interference vs grid12 ground truth

## Method

For each source checkpoint theta_O (grid12, `D:\CAPSTONE\grid12\ckpt`), the training loss (image vs all class prototypes of the dataset, 10 config templates, logits = 100 * img @ txt.T, cross-entropy; float32, logit_scale frozen) is differentiated on the TRAIN split. A gradient set is the mean gradient over the first K=4 batches (batch 64, seed 0, no shuffling). Gradients are split into V = `visual.*` and L = all other parameters except logit_scale/logit_bias. No fine-tuning is performed.

- I_sgd_E  = -dot(g_O_E, g_N_E)
- I_adam_E = -dot(g_O_E, sign(g_N_E))  (primary; AdamW first step)
- cos_E = cosine(g_O_E, g_N_E)
- Dp = (I_V - I_L) / (|I_V| + |I_L|), for adam and sgd versions
- Prediction: Dp > 0 means vision-dominated forgetting (true D > 0). Threshold 0, no tuning.
- Gap baseline: gap_v - gap_l with its median as threshold (**data-tuned** threshold).

## Per-pair table (Adam signal, primary)

| O | N | I_adam_V | I_adam_L | Dp_adam | true F_V | true F_L | true D | correct? |
|---|---|---:|---:|---:|---:|---:|---:|:-:|
| OxfordIIITPet | Flowers102 | -3001 | -1916 | -0.221 | 4.50 | 23.30 | -0.676 | yes |
| OxfordIIITPet | EuroSAT | -3056 | -1004 | -0.506 | 16.25 | 38.40 | -0.405 | yes |
| OxfordIIITPet | DTD | -1136 | -552.9 | -0.345 | 11.87 | 36.23 | -0.506 | yes |
| Flowers102 | OxfordIIITPet | -1284 | -1416 | +0.049 | 18.25 | 43.90 | -0.413 | no |
| Flowers102 | EuroSAT | 1374 | -590.8 | +1.000 | 44.57 | 35.33 | +0.116 | yes |
| Flowers102 | DTD | 1159 | -4221 | +1.000 | 23.65 | 46.25 | -0.323 | no |
| EuroSAT | OxfordIIITPet | 672.9 | 293.7 | +0.392 | 36.10 | 9.70 | +0.576 | yes |
| EuroSAT | Flowers102 | -302.4 | -529.8 | +0.273 | 25.42 | 0.12 | +0.990 | yes |
| EuroSAT | DTD | -6844 | -186 | -0.947 | 62.38 | 3.52 | +0.893 | no |
| DTD | OxfordIIITPet | -5266 | -1531 | -0.550 | 9.10 | 32.50 | -0.563 | yes |
| DTD | Flowers102 | -1.134e+04 | -656 | -0.891 | 30.74 | 10.16 | +0.503 | no |
| DTD | EuroSAT | -5125 | -1687 | -0.505 | 41.06 | 23.09 | +0.280 | no |

## Comparison vs gap baseline

| metric | Adam Dp | SGD Dp | gap_v - gap_l |
|---|---:|---:|---:|
| sign accuracy (/12) | 7 | 3 | 10 (median thr +0.1570, data-tuned) |
| AUROC for D > 0 | 0.486 | 0.194 | 0.833 |
| Spearman score vs D | +0.018 (p=0.9604) | -0.531 (p=0.0794) | +0.501 (p=0.1023) |
| Spearman I_V vs F_V (gap: gap_v vs F_V) | +0.049 (p=0.8880) | -0.119 (p=0.7177) | -0.077 (p=0.8146) |
| Spearman I_L vs F_L (gap: gap_l vs F_L) | -0.587 (p=0.0480) | -0.671 (p=0.0204) | +0.258 (p=0.4140) |
| Spearman I_V+I_L vs total (gap: gap_v+gap_l vs total) | +0.056 (p=0.8698) | -0.245 (p=0.4445) | -0.298 (p=0.3454) |

p-values: two-sided permutation test, 10,000 permutations, seed 0.

## Leave-one-source-out (sign accuracy on the 3 pairs of each held-out O; no fitting)

| held-out O | Adam | SGD | gap (median thr from all 12) |
|---|:-:|:-:|:-:|
| OxfordIIITPet | 3/3 | 1/3 | 2/3 |
| Flowers102 | 1/3 | 1/3 | 3/3 |
| EuroSAT | 2/3 | 0/3 | 3/3 |
| DTD | 1/3 | 1/3 | 2/3 |

The Adam/SGD signals use a fixed threshold of 0, so leave-one-source-out needs no refitting; the gap column reuses the median of all 12 pairs and is therefore not a true held-out number.

## Interpretation

Rule: "promising" if Adam sign accuracy >= 10/12 AND AUROC > 0.83 (gap baseline) AND Dp-vs-D p < 0.05; "comparable" if within ±1 correct pair of the gap baseline; otherwise "not better".

**Applies: not better** (Adam 7/12, AUROC 0.486, Dp-vs-D p = 0.9604; gap baseline 10/12).

## Caveats

> Exploratory: 12 pairs from 4 datasets (not independent), single seed, K=4 gradient batches,
>  first-order (one-step) approximation of a full epoch of training. Not a confirmation.

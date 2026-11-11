# Part C results: 2x2 encoder swap + Shapley split

## What was run

- Code: `src/swap_shapley.py` via `scripts/run_swap.py`, run on Shreyas's Part B checkpoints (`part_b_artifacts.zip`, `checkpoints/`).
- 5 pairs, each with `theta_pets.pt` as "before" (p): Pets->EuroSAT, Pets->Flowers, Pets->EuroSAT (text frozen), Pets->EuroSAT (lr 3e-5), Pets->EuroSAT text frozen (lr 3e-5).
- Old-domain evaluation: 2000 Oxford-IIIT Pets test images (seeded randperm, `data.seed = 0`, `data.max_images = 2000` from `configs/config.yaml`).
- Device: CPU (torch 2.15.0+cpu, open_clip 4.2.1). Total wall-clock time: 1134 s (18.9 min).
- Pets data: the original `images.tar.gz` / `annotations.tar.gz` (MD5 matches torchvision's `OxfordIIITPet` checksums).

Hybrid naming: letter 1 = vision checkpoint, letter 2 = text checkpoint (p = before, f = after).
Accuracies (A_*), F_V, F_L, total, interaction and CW_* are in percentage points; M_* are linear-margin values.

## Results

| pair | A_pp | A_pf | A_fp | A_ff | total | F_V | F_L | interaction | D | M_F_V | M_F_L | CW_pp | CW_ff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pets->EuroSAT | 86.05 | 85.25 | 85.85 | 85.35 | 0.70 | 0.050 | 0.650 | -0.30 | NaN | 0.0027 | 0.0173 | 1.90 | 1.05 |
| Pets->Flowers | 86.05 | 84.45 | 85.80 | 83.65 | 2.40 | 0.525 | 1.875 | 0.55 | -0.562 | 0.0010 | -0.0019 | 1.90 | 4.15 |
| Pets->EuroSAT (text frozen) | 86.05 | 86.05 | 86.60 | 86.60 | -0.55 | -0.550 | 0.000 | 0.00 | NaN | -0.0009 | 0.0000 | 1.90 | 1.95 |
| Pets->EuroSAT (lr 3e-5) | 86.05 | 43.85 | 69.30 | 32.30 | 53.75 | 14.150 | 39.600 | -5.20 | -0.473 | 0.0269 | 0.1002 | 1.90 | 0.00 |
| Pets->EuroSAT text frozen (lr 3e-5) | 86.05 | 86.05 | 59.25 | 59.25 | 26.80 | 26.800 | 0.000 | 0.00 | 1.000 | 0.0664 | 0.0000 | 1.90 | 2.40 |

Full columns (incl. M_pp..M_ff, M_total, M_interaction, CW_pf, CW_fp, text_frozen) are in `swap_results.csv`.

## Checks (all passed)

| check | result |
|---|---|
| a) A_pp = 86.05 for every pair (tol 0.1) | PASS |
| b) A_ff matches Shreyas's table (85.35 / 83.65 / 86.60 / 32.30 / 59.25, tol 0.1) | PASS |
| c) Both text-frozen pairs: text_frozen = True, F_L = 0, interaction = 0, M_F_L = 0 | PASS |
| d) F_V + F_L = total for every pair | PASS |
| e) predictions.csv: 40000 rows (5 pairs x 4 hybrids x 2000), columns pair, hybrid, image_id, true, pred, conf | PASS |

## Larger share per pair

- Pets->EuroSAT: forgetting below 1 pt: split not interpreted.
- Pets->Flowers: text encoder has the larger share (F_L 1.875 vs F_V 0.525).
- Pets->EuroSAT (text frozen): forgetting below 1 pt: split not interpreted.
- Pets->EuroSAT (lr 3e-5): text encoder has the larger share (F_L 39.60 vs F_V 14.15).
- Pets->EuroSAT text frozen (lr 3e-5): vision encoder has the larger share (F_V 26.80 vs F_L 0.00).

## Comparison: total forgetting with text frozen (lr 3e-5) vs F_V of the full lr 3e-5 run

| | value (pts) |
|---|---|
| total forgetting, Pets->EuroSAT text frozen (lr 3e-5) | 26.80 |
| F_V, Pets->EuroSAT (lr 3e-5) | 14.15 |

## Note for Mythri (Part D)

`predictions.csv` is the input for Part D (ECE). `image_id` is a string (e.g. `havanese_8`); read it with
`pd.read_csv("results/swap/predictions.csv", dtype={"image_id": str})`.

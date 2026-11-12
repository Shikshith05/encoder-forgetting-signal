# Parameter dictionary: pair_parameters.csv

One row per ordered pair A -> B (A = old = `O`, B = new = `N`). Every signal column exists twice:
`pre_<name>` = computed with pretrained CLIP ViT-B/32 (OpenAI, QuickGELU, repo `load_clip(None)`);
`thA_<name>` = computed with theta_A (`D:\CAPSTONE\grid12\ckpt\theta_<A>.pt`), both A and B
features taken from that same model. Features come from `D:\CAPSTONE\grid12\cache\features\`
(build_features.py): unit-normalised image embeddings; prototypes = repo `extract_text_features`
(mean of the 10 unit-normalised config-template embeddings, re-normalised). TRAIN = repo
`get_dataset_loader(..., "train")`, seed 0, <= 2000 images (Pets train = trainval; EuroSAT
train = the next seed-0 block after the test block). cos = dot product of unit vectors. Softmax
uses scale 100. A's TEST split is never used by any column below.

| column | formula | data |
|---|---|---|
| O, N | old task A, new task B | - |
| S_inter_B | mean over B images of mean cos to the WRONG B prototypes (`compute_iimm_score` inter) | B train |
| S_intra_B | mean cos over distinct pairs of B images (`compute_iimm_score` intra) | B train |
| IIMM_B | (S_inter_B + S_intra_B) / 2 | B train |
| zs_acc_B_train | zero-shot top-1 % with B prototypes (`compute_zeroshot_accuracy`) | B train |
| zs_acc_B_test | zero-shot top-1 % with B prototypes | B TEST |
| margin_B | mean over B images of cos(true proto) - max cos(wrong protos) | B train |
| maxprob_B | mean over B images of max softmax(100 cos) | B train |
| entropy_B | mean over B images of -sum p log p (nats) of softmax(100 cos) | B train |
| text_intra_B | mean cos over distinct pairs of B prototypes | B prototypes |
| modality_gap_B | L2 norm of mean(B image emb) - mean(B prototypes) | B train + prototypes |
| n_classes_B | number of B classes | - |
| S_inter_A, S_intra_A, IIMM_A | same as the B versions, on A | A train |
| zs_acc_A_train | zero-shot top-1 % with A prototypes | A train |
| margin_A | same as margin_B, on A | A train |
| Con_II | mean cos(A image, B image) over all pairs (Sekeh & Wisell Def 4.1-4.2) | A train, B train |
| Con_TT | mean cos(A prototype, B prototype) | prototypes |
| Con_IT | mean cos(A image, B prototype) | A train, B prototypes |
| Con_TI | mean cos(A prototype, B image) | A prototypes, B train |
| CMS | (Con_IT + Con_TI) / 2 | as above |
| CMS4 | (Con_II + Con_TT + Con_IT + Con_TI) / 4 | as above |
| cross_inter | mean cos of B images to ALL A prototypes (identical to Con_TI by definition) | B train, A prototypes |
| nearest_old_label | mean over B images of max cos to A prototypes | B train, A prototypes |
| text_overlap | mean over A prototypes of max cos to B prototypes | prototypes |
| Gap_V | `compute_gap` (Frechet, Ledoit-Wolf) between A and B image embeddings | A train, B train |
| Gap_L | `compute_gap` between A and B prototypes | prototypes |

Ground-truth columns, copied unchanged from `D:\CAPSTONE\grid12\results\grid12_results.csv`
(`total`, `A_pp`, `A_pf`, `A_fp`, `A_ff`, `F_V`, `F_L`, `interaction`, `D`, `M_*`, `CW_*`, training
metadata). The four old pretrained signals in that file are renamed `grid12_gap_v`, `grid12_gap_l`,
`grid12_zs_N`, `grid12_iimm_N` (computed earlier by grid12.py on TEST features; kept for reference).

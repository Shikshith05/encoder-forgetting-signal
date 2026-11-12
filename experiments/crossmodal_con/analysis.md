# Cross-modal contribution and IIMM signals vs grid12 forgetting

Pre-registration: `D:\CAPSTONE\prereg_crossmodal.md` (copy: prereg_crossmodal.md).

> Exploratory: 12 non-independent pairs, single seed, a dev set reused by earlier signals. Nothing here is claimed as a working predictor; the result only decides what is pre-registered for Phase-II (30+ pairs).

## Primary tests (Bonferroni alpha = 0.0167)

| id | test | rho | perm p | within-A perm p | significant? |
|---|---|:-:|:-:|:-:|:-:|
| P1 | Spearman(thA_CMS, total) | -0.133 | 0.6810 | 0.9760 | no |
| P2 | Spearman(thA_IIMM_B, total) | -0.308 | 0.3294 | 0.3969 | no |

P3 held-out error (same folds for every model):

| model | scheme | MAE (pp) | R^2 |
|---|---|:-:|:-:|
| P3: total ~ thA_IIMM_B + thA_CMS | LOPO | 17.89 | -0.563 |
| P3: total ~ thA_IIMM_B + thA_CMS | LOODO | 18.88 | -0.985 |
| baseline: training-fold mean | LOPO | 15.17 | -0.190 |
| baseline: training-fold mean | LOODO | 16.03 | -0.350 |
| baseline: total ~ grid12_zs_N | LOPO | 13.81 | -0.035 |
| baseline: total ~ grid12_zs_N | LOODO | 14.16 | -0.111 |

Stated in chat, no file: mean predictor 15.2 pp, zero-shot alone 13.8 pp (held-out MAE). Only the recomputed values above are used for the verdict.

P3 better than both baselines: LOPO no, LOODO no -> **NOT better**

## Already-measured baselines

| signal | rho | perm p | within-A p | source |
|---|:-:|:-:|:-:|---|
| zero-shot on B (grid12_zs_N) | -0.583 | 0.0553 | 0.013 (stated in chat, no file) | grid12 analysis.md 5c |
| zero-shot on B, recomputed here | -0.583 | 0.0553 | 0.0159 | this script |
| IIMM pretrained (grid12_iimm_N) | -0.086 | 0.8119 | - | grid12 analysis.md 5d |
| Gap_V (A TEST images) | -0.184 | 0.5824 | - | grid12 analysis.md 5a |
| Gap_L | -0.353 | 0.2659 | - | grid12 analysis.md 5b |

Note: the old grid12 Gap_V / Gap_L (and zs_N, iimm_N) baselines were computed by grid12.py `step_signals` on TEST features, i.e. Gap_V used A's TEST images. This experiment's pre_/thA_ Gap_V and Gap_L use TRAIN images only; A's TEST split feeds no signal here.

## Exploratory (51 parameters x 4 targets = 204 tests; BH over all; sorted by adjusted p)

Dropped before testing: `pre_cross_inter` (= pre_Con_TI by definition, max float diff 5.6e-17); `thA_n_classes_B` (identical to pre_n_classes_B); `thA_cross_inter` (= thA_Con_TI by definition, max float diff 2.8e-17)

| parameter | target | rho | perm p | within-A p | BH p |
|---|---|:-:|:-:|:-:|:-:|
| thA_S_intra_A | F_L | +0.907 | 0.0001 | 1.0000 | 0.0068 |
| thA_IIMM_A | F_L | +0.907 | 0.0001 | 1.0000 | 0.0068 |
| thA_text_overlap | F_L | -0.916 | 0.0001 | 0.0850 | 0.0068 |
| thA_S_inter_B | F_L | -0.874 | 0.0003 | 0.0111 | 0.0153 |
| pre_Con_TT | F_V | +0.848 | 0.0006 | 0.0092 | 0.0245 |
| thA_Con_TT | F_L | -0.832 | 0.0011 | 0.5562 | 0.0321 |
| thA_Gap_L | F_L | +0.846 | 0.0011 | 0.2960 | 0.0321 |
| pre_text_overlap | F_L | -0.783 | 0.0036 | 0.1268 | 0.0686 |
| pre_S_inter_A | D | +0.799 | 0.0037 | 1.0000 | 0.0686 |
| pre_zs_acc_A_train | D | -0.799 | 0.0037 | 1.0000 | 0.0686 |
| pre_margin_A | D | -0.799 | 0.0037 | 1.0000 | 0.0686 |
| pre_Con_TT | D | +0.792 | 0.0046 | 0.0106 | 0.0782 |
| pre_S_inter_A | F_L | -0.777 | 0.0067 | 1.0000 | 0.0911 |
| pre_zs_acc_A_train | F_L | +0.777 | 0.0067 | 1.0000 | 0.0911 |
| pre_margin_A | F_L | +0.777 | 0.0067 | 1.0000 | 0.0911 |
| thA_CMS4 | F_L | -0.748 | 0.0073 | 0.4391 | 0.0931 |
| thA_CMS | F_L | -0.720 | 0.0116 | 0.6254 | 0.1392 |
| pre_text_overlap | D | +0.706 | 0.0133 | 0.0715 | 0.1507 |
| thA_Con_TI | F_L | -0.699 | 0.0147 | 0.5595 | 0.1578 |
| thA_text_overlap | D | +0.650 | 0.0244 | 0.1373 | 0.2489 |
| thA_Con_TT | D | +0.643 | 0.0269 | 0.0937 | 0.2513 |
| pre_text_overlap | F_V | +0.643 | 0.0271 | 0.0375 | 0.2513 |
| thA_S_intra_A | D | -0.626 | 0.0357 | 1.0000 | 0.2735 |
| thA_IIMM_A | D | -0.626 | 0.0357 | 1.0000 | 0.2735 |
| pre_S_inter_A | F_V | +0.626 | 0.0362 | 1.0000 | 0.2735 |
| pre_zs_acc_A_train | F_V | -0.626 | 0.0362 | 1.0000 | 0.2735 |
| pre_margin_A | F_V | -0.626 | 0.0362 | 1.0000 | 0.2735 |
| thA_Con_IT | total | +0.615 | 0.0393 | 0.0131 | 0.2799 |
| thA_CMS | D | +0.608 | 0.0398 | 0.0799 | 0.2799 |
| pre_Con_IT | total | +0.594 | 0.0429 | 0.0127 | 0.2917 |
| pre_Gap_L | F_V | -0.586 | 0.0463 | 0.1218 | 0.2965 |
| pre_n_classes_B | total | -0.605 | 0.0475 | 0.0388 | 0.2965 |
| pre_S_inter_B | total | +0.583 | 0.0553 | 0.0159 | 0.2965 |
| pre_zs_acc_B_train | total | -0.583 | 0.0553 | 0.0159 | 0.2965 |
| pre_margin_B | total | -0.583 | 0.0553 | 0.0159 | 0.2965 |
| pre_zs_acc_B_test | total | -0.583 | 0.0553 | 0.0159 | 0.2965 |
| pre_text_intra_B | total | +0.583 | 0.0553 | 0.0159 | 0.2965 |
| pre_modality_gap_B | total | +0.583 | 0.0553 | 0.0159 | 0.2965 |
| thA_CMS4 | D | +0.573 | 0.0576 | 0.1840 | 0.2965 |
| thA_Gap_L | D | -0.566 | 0.0588 | 0.5987 | 0.2965 |
| thA_nearest_old_label | F_L | -0.566 | 0.0596 | 0.3147 | 0.2965 |
| pre_maxprob_B | total | -0.561 | 0.0679 | 0.0278 | 0.3185 |
| pre_entropy_B | total | +0.561 | 0.0679 | 0.0278 | 0.3185 |
| thA_S_inter_A | F_L | -0.561 | 0.0687 | 1.0000 | 0.3185 |
| thA_nearest_old_label | D | +0.531 | 0.0813 | 0.1869 | 0.3589 |
| pre_S_intra_A | D | +0.540 | 0.0827 | 1.0000 | 0.3589 |
| pre_IIMM_A | D | +0.540 | 0.0827 | 1.0000 | 0.3589 |
| thA_nearest_old_label | F_V | +0.517 | 0.0886 | 0.1293 | 0.3692 |
| thA_S_inter_B | D | +0.517 | 0.0887 | 0.1665 | 0.3692 |
| thA_S_inter_B | total | -0.510 | 0.0905 | 0.2701 | 0.3692 |
| thA_modality_gap_B | total | +0.503 | 0.0987 | 0.0101 | 0.3923 |
| pre_CMS4 | F_V | +0.501 | 0.1000 | 0.0640 | 0.3923 |
| thA_S_inter_A | D | +0.497 | 0.1112 | 1.0000 | 0.4160 |
| thA_Con_TI | D | +0.483 | 0.1139 | 0.1320 | 0.4160 |
| thA_entropy_B | D | +0.490 | 0.1141 | 0.0383 | 0.4160 |
| pre_Con_TI | F_L | -0.483 | 0.1142 | 0.7316 | 0.4160 |
| thA_CMS | F_V | +0.476 | 0.1199 | 0.0480 | 0.4291 |
| pre_Con_TT | F_L | -0.481 | 0.1222 | 0.2643 | 0.4298 |
| thA_S_intra_A | total | +0.475 | 0.1302 | 1.0000 | 0.4426 |
| thA_IIMM_A | total | +0.475 | 0.1302 | 1.0000 | 0.4426 |
| thA_text_intra_B | total | +0.455 | 0.1391 | 0.1048 | 0.4631 |
| thA_margin_B | total | -0.448 | 0.1471 | 0.0249 | 0.4631 |
| pre_Gap_L | D | -0.441 | 0.1490 | 0.1600 | 0.4631 |
| thA_S_inter_A | F_V | +0.453 | 0.1493 | 1.0000 | 0.4631 |
| thA_Con_TT | F_V | +0.441 | 0.1500 | 0.0383 | 0.4631 |
| thA_zs_acc_B_train | total | -0.441 | 0.1519 | 0.0578 | 0.4631 |
| thA_zs_acc_B_test | total | -0.441 | 0.1521 | 0.0439 | 0.4631 |
| thA_text_overlap | F_V | +0.434 | 0.1549 | 0.0410 | 0.4647 |
| pre_Con_IT | F_V | +0.434 | 0.1606 | 0.0904 | 0.4748 |
| thA_maxprob_B | D | -0.427 | 0.1746 | 0.0211 | 0.5088 |
| thA_Con_TI | F_V | +0.413 | 0.1816 | 0.0236 | 0.5217 |
| pre_CMS4 | D | +0.406 | 0.1847 | 0.0624 | 0.5233 |
| pre_S_intra_A | F_V | +0.410 | 0.1954 | 1.0000 | 0.5361 |
| pre_IIMM_A | F_V | +0.410 | 0.1954 | 1.0000 | 0.5361 |
| thA_CMS4 | F_V | +0.399 | 0.1971 | 0.2026 | 0.5361 |
| pre_Con_TT | total | +0.396 | 0.2105 | 0.0735 | 0.5650 |
| thA_zs_acc_A_train | F_L | -0.389 | 0.2177 | 1.0000 | 0.5693 |
| thA_margin_A | F_L | -0.389 | 0.2177 | 1.0000 | 0.5693 |
| thA_Gap_L | F_V | -0.378 | 0.2219 | 0.4312 | 0.5730 |
| pre_Con_TI | D | +0.378 | 0.2336 | 0.0775 | 0.5956 |
| pre_CMS | F_V | +0.368 | 0.2462 | 0.1103 | 0.6052 |
| pre_n_classes_B | F_L | -0.367 | 0.2496 | 0.2134 | 0.6052 |
| thA_zs_acc_A_train | D | +0.367 | 0.2503 | 1.0000 | 0.6052 |
| thA_margin_A | D | +0.367 | 0.2503 | 1.0000 | 0.6052 |
| thA_entropy_B | F_L | -0.357 | 0.2522 | 0.3658 | 0.6052 |
| pre_Gap_L | total | -0.349 | 0.2642 | 0.2948 | 0.6071 |
| thA_text_overlap | total | -0.343 | 0.2745 | 0.8971 | 0.6071 |
| pre_S_inter_B | F_V | +0.345 | 0.2768 | 0.3270 | 0.6071 |
| pre_zs_acc_B_train | F_V | -0.345 | 0.2768 | 0.3270 | 0.6071 |
| pre_margin_B | F_V | -0.345 | 0.2768 | 0.3270 | 0.6071 |
| pre_zs_acc_B_test | F_V | -0.345 | 0.2768 | 0.3270 | 0.6071 |
| pre_text_intra_B | F_V | +0.345 | 0.2768 | 0.3270 | 0.6071 |
| pre_modality_gap_B | F_V | +0.345 | 0.2768 | 0.3270 | 0.6071 |
| thA_margin_B | F_L | -0.336 | 0.2837 | 0.6223 | 0.6115 |
| thA_Gap_L | total | +0.336 | 0.2848 | 0.6721 | 0.6115 |
| thA_maxprob_B | F_V | -0.336 | 0.2879 | 0.2305 | 0.6117 |
| thA_text_intra_B | F_V | +0.329 | 0.3014 | 0.2381 | 0.6338 |
| thA_S_intra_A | F_V | -0.324 | 0.3091 | 1.0000 | 0.6369 |
| thA_IIMM_A | F_V | -0.324 | 0.3091 | 1.0000 | 0.6369 |
| thA_IIMM_B | F_L | -0.315 | 0.3152 | 0.1270 | 0.6429 |
| thA_S_inter_B | F_V | +0.308 | 0.3214 | 0.2207 | 0.6475 |
| pre_CMS4 | total | +0.310 | 0.3238 | 0.2366 | 0.6475 |
| thA_IIMM_B (P) | total | -0.308 | 0.3294 | 0.3969 | 0.6523 |
| thA_modality_gap_B | F_V | +0.308 | 0.3326 | 0.3425 | 0.6523 |
| pre_Con_TI | F_V | +0.301 | 0.3379 | 0.0519 | 0.6564 |
| pre_CMS | total | +0.283 | 0.3861 | 0.3543 | 0.7430 |
| pre_CMS4 | F_L | -0.268 | 0.3929 | 0.5929 | 0.7443 |
| thA_maxprob_B | F_L | +0.266 | 0.3975 | 0.3952 | 0.7443 |
| thA_entropy_B | F_V | +0.266 | 0.4042 | 0.2821 | 0.7443 |
| pre_Con_II | total | +0.269 | 0.4078 | 0.4135 | 0.7443 |
| thA_Con_TT | total | -0.259 | 0.4127 | 0.9791 | 0.7443 |
| pre_Gap_L | F_L | +0.258 | 0.4140 | 0.0621 | 0.7443 |
| pre_maxprob_B | F_V | -0.259 | 0.4251 | 0.3949 | 0.7443 |
| pre_entropy_B | F_V | +0.259 | 0.4251 | 0.3949 | 0.7443 |
| thA_Con_IT | F_L | +0.252 | 0.4267 | 0.4277 | 0.7443 |
| thA_Con_IT | F_V | +0.252 | 0.4289 | 0.3030 | 0.7443 |
| pre_S_intra_A | F_L | -0.259 | 0.4339 | 1.0000 | 0.7443 |
| pre_IIMM_A | F_L | -0.259 | 0.4339 | 1.0000 | 0.7443 |
| pre_CMS | D | +0.254 | 0.4342 | 0.2979 | 0.7443 |
| thA_text_intra_B | D | +0.245 | 0.4403 | 0.2004 | 0.7484 |
| pre_Con_IT | D | +0.238 | 0.4528 | 0.3221 | 0.7579 |
| pre_CMS | F_L | -0.240 | 0.4586 | 0.7064 | 0.7579 |
| thA_CMS4 | total | -0.231 | 0.4599 | 0.8779 | 0.7579 |
| pre_Con_II | F_V | +0.240 | 0.4607 | 0.3316 | 0.7579 |
| pre_n_classes_B | F_V | -0.238 | 0.4724 | 0.4021 | 0.7683 |
| thA_zs_acc_B_train | F_V | -0.231 | 0.4746 | 0.5185 | 0.7683 |
| thA_IIMM_B | D | +0.217 | 0.5035 | 0.2524 | 0.8089 |
| thA_Gap_V | total | -0.210 | 0.5127 | 0.3497 | 0.8164 |
| thA_Con_II | total | +0.196 | 0.5378 | 0.4404 | 0.8164 |
| pre_nearest_old_label | total | -0.189 | 0.5505 | 0.3802 | 0.8164 |
| thA_S_intra_B | F_L | -0.189 | 0.5538 | 0.1891 | 0.8164 |
| pre_maxprob_B | F_L | -0.194 | 0.5559 | 0.4946 | 0.8164 |
| pre_entropy_B | F_L | +0.194 | 0.5559 | 0.4946 | 0.8164 |
| pre_S_inter_B | F_L | +0.194 | 0.5562 | 0.6844 | 0.8164 |
| pre_zs_acc_B_train | F_L | -0.194 | 0.5562 | 0.6844 | 0.8164 |
| pre_margin_B | F_L | -0.194 | 0.5562 | 0.6844 | 0.8164 |
| pre_zs_acc_B_test | F_L | -0.194 | 0.5562 | 0.6844 | 0.8164 |
| pre_text_intra_B | F_L | +0.194 | 0.5562 | 0.6844 | 0.8164 |
| pre_modality_gap_B | F_L | +0.194 | 0.5562 | 0.6844 | 0.8164 |
| thA_zs_acc_B_test | F_V | -0.182 | 0.5826 | 0.6076 | 0.8431 |
| thA_zs_acc_B_test | F_L | -0.175 | 0.5895 | 0.6017 | 0.8431 |
| pre_Gap_V | total | -0.168 | 0.5989 | 0.5219 | 0.8431 |
| thA_modality_gap_B | D | +0.168 | 0.6009 | 0.4744 | 0.8431 |
| thA_zs_acc_A_train | total | -0.173 | 0.6066 | 1.0000 | 0.8431 |
| thA_margin_A | total | -0.173 | 0.6066 | 1.0000 | 0.8431 |
| pre_S_intra_A | total | +0.173 | 0.6102 | 1.0000 | 0.8431 |
| pre_IIMM_A | total | +0.173 | 0.6102 | 1.0000 | 0.8431 |
| thA_Con_II | D | -0.161 | 0.6185 | 0.6800 | 0.8431 |
| thA_S_intra_B | total | -0.161 | 0.6195 | 0.5804 | 0.8431 |
| thA_Con_TI | total | -0.161 | 0.6199 | 0.9923 | 0.8431 |
| thA_maxprob_B | total | -0.147 | 0.6472 | 0.7061 | 0.8565 |
| thA_modality_gap_B | F_L | +0.147 | 0.6473 | 0.4534 | 0.8565 |
| thA_zs_acc_B_train | F_L | -0.147 | 0.6494 | 0.5497 | 0.8565 |
| pre_nearest_old_label | F_V | -0.147 | 0.6513 | 0.4910 | 0.8565 |
| thA_CMS (P) | total | -0.133 | 0.6810 | 0.9760 | 0.8565 |
| thA_Gap_V | D | +0.133 | 0.6819 | 0.4712 | 0.8565 |
| pre_Con_TI | total | -0.126 | 0.6903 | 0.9563 | 0.8565 |
| thA_Con_II | F_L | +0.126 | 0.6991 | 0.7528 | 0.8565 |
| thA_margin_B | D | +0.126 | 0.7024 | 0.9632 | 0.8565 |
| thA_S_intra_B | F_V | +0.126 | 0.7027 | 0.6754 | 0.8565 |
| pre_S_inter_A | total | -0.130 | 0.7045 | 1.0000 | 0.8565 |
| pre_zs_acc_A_train | total | +0.130 | 0.7045 | 1.0000 | 0.8565 |
| pre_margin_A | total | +0.130 | 0.7045 | 1.0000 | 0.8565 |
| pre_S_intra_B | F_V | +0.130 | 0.7090 | 0.6850 | 0.8565 |
| pre_IIMM_B | F_V | +0.130 | 0.7090 | 0.6850 | 0.8565 |
| thA_S_inter_A | total | -0.130 | 0.7170 | 1.0000 | 0.8565 |
| thA_IIMM_B | F_V | +0.119 | 0.7206 | 0.6376 | 0.8565 |
| thA_Gap_V | F_L | -0.105 | 0.7479 | 0.5665 | 0.8565 |
| pre_maxprob_B | D | -0.108 | 0.7525 | 0.7068 | 0.8565 |
| pre_entropy_B | D | +0.108 | 0.7525 | 0.7068 | 0.8565 |
| pre_S_intra_B | F_L | -0.108 | 0.7542 | 0.5867 | 0.8565 |
| pre_IIMM_B | F_L | -0.108 | 0.7542 | 0.5867 | 0.8565 |
| pre_S_inter_B | D | +0.108 | 0.7559 | 0.7839 | 0.8565 |
| pre_zs_acc_B_train | D | -0.108 | 0.7559 | 0.7839 | 0.8565 |
| pre_margin_B | D | -0.108 | 0.7559 | 0.7839 | 0.8565 |
| pre_zs_acc_B_test | D | -0.108 | 0.7559 | 0.7839 | 0.8565 |
| pre_text_intra_B | D | +0.108 | 0.7559 | 0.7839 | 0.8565 |
| pre_modality_gap_B | D | +0.108 | 0.7559 | 0.7839 | 0.8565 |
| pre_n_classes_B | D | +0.108 | 0.7563 | 0.7533 | 0.8565 |
| thA_zs_acc_A_train | F_V | +0.108 | 0.7599 | 1.0000 | 0.8565 |
| thA_margin_A | F_V | +0.108 | 0.7599 | 1.0000 | 0.8565 |
| thA_zs_acc_B_train | D | -0.091 | 0.7826 | 0.7721 | 0.8740 |
| thA_S_intra_B | D | +0.091 | 0.7840 | 0.7226 | 0.8740 |
| thA_Con_IT | D | +0.084 | 0.8017 | 0.6987 | 0.8857 |
| pre_Gap_V | F_L | -0.077 | 0.8040 | 0.6567 | 0.8857 |
| pre_S_intra_B | total | -0.086 | 0.8119 | 0.7610 | 0.8857 |
| pre_IIMM_B | total | -0.086 | 0.8119 | 0.7610 | 0.8857 |
| pre_Gap_V | F_V | -0.053 | 0.8679 | 0.8210 | 0.9203 |
| pre_nearest_old_label | D | -0.056 | 0.8697 | 0.5978 | 0.9203 |
| thA_margin_B | F_V | -0.056 | 0.8697 | 0.9073 | 0.9203 |
| pre_S_intra_B | D | +0.065 | 0.8717 | 0.8504 | 0.9203 |
| pre_IIMM_B | D | +0.065 | 0.8717 | 0.8504 | 0.9203 |
| pre_Con_II | F_L | -0.057 | 0.8739 | 0.6828 | 0.9203 |
| pre_Con_II | D | +0.057 | 0.8769 | 0.7693 | 0.9203 |
| pre_nearest_old_label | F_L | +0.049 | 0.8825 | 0.6738 | 0.9203 |
| thA_Gap_V | F_V | -0.049 | 0.8858 | 0.8050 | 0.9203 |
| thA_Con_II | F_V | +0.049 | 0.8887 | 0.8555 | 0.9203 |
| thA_nearest_old_label | total | -0.042 | 0.9091 | 0.9013 | 0.9367 |
| pre_text_overlap | total | -0.035 | 0.9189 | 0.9440 | 0.9420 |
| pre_Gap_V | D | +0.032 | 0.9273 | 0.8523 | 0.9459 |
| pre_Con_IT | F_L | -0.021 | 0.9557 | 0.9438 | 0.9700 |
| thA_entropy_B | total | -0.014 | 0.9730 | 0.9785 | 0.9826 |
| thA_zs_acc_B_test | D | +0.000 | 1.0000 | 1.0000 | 1.0000 |
| thA_text_intra_B | F_L | +0.000 | 1.0000 | 1.0000 | 1.0000 |

## Per pair (main parameters)

| O | N | thA_CMS | thA_IIMM_B | pre_CMS | pre_IIMM_B | thA_Con_II | thA_Con_TT | thA_Con_IT | thA_Con_TI | grid12_zs_N | total | pred_P3_LOPO | pred_P3_LOODO |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| OxfordIIITPet | Flowers102 | 0.1662 | 0.5059 | 0.1747 | 0.4938 | 0.5442 | 0.5880 | 0.1983 | 0.1340 | 64.80 | 27.80 | 53.39 | 52.41 |
| OxfordIIITPet | EuroSAT | 0.1879 | 0.5196 | 0.1894 | 0.5249 | 0.4850 | 0.6027 | 0.2193 | 0.1564 | 35.30 | 54.65 | 45.09 | 50.58 |
| OxfordIIITPet | DTD | 0.1876 | 0.4524 | 0.2074 | 0.4444 | 0.5154 | 0.6298 | 0.2246 | 0.1506 | 43.24 | 48.10 | 59.36 | 59.67 |
| Flowers102 | OxfordIIITPet | 0.1532 | 0.4548 | 0.1747 | 0.4627 | 0.5352 | 0.4332 | 0.1847 | 0.1218 | 85.60 | 62.15 | 41.03 | 24.26 |
| Flowers102 | EuroSAT | 0.1859 | 0.5060 | 0.1945 | 0.5249 | 0.5128 | 0.5168 | 0.2336 | 0.1382 | 35.30 | 79.90 | 44.14 | 39.39 |
| Flowers102 | DTD | 0.1842 | 0.4324 | 0.2013 | 0.4444 | 0.5718 | 0.5285 | 0.2349 | 0.1336 | 43.24 | 69.90 | 55.37 | 44.98 |
| EuroSAT | OxfordIIITPet | 0.1893 | 0.4779 | 0.1894 | 0.4627 | 0.4563 | 0.7713 | 0.1968 | 0.1819 | 85.60 | 45.80 | 54.41 | 56.15 |
| EuroSAT | Flowers102 | 0.1995 | 0.5165 | 0.1945 | 0.4938 | 0.4820 | 0.8113 | 0.2088 | 0.1902 | 64.80 | 25.55 | 53.54 | 54.21 |
| EuroSAT | DTD | 0.2135 | 0.4509 | 0.2186 | 0.4444 | 0.5362 | 0.8664 | 0.2241 | 0.2030 | 43.24 | 65.90 | 59.33 | 63.92 |
| DTD | OxfordIIITPet | 0.1988 | 0.4764 | 0.2074 | 0.4627 | 0.5273 | 0.6841 | 0.1955 | 0.2021 | 85.60 | 41.60 | 57.60 | 54.93 |
| DTD | Flowers102 | 0.1946 | 0.5163 | 0.2013 | 0.4938 | 0.5919 | 0.7079 | 0.1934 | 0.1957 | 64.80 | 40.90 | 49.76 | 44.86 |
| DTD | EuroSAT | 0.2212 | 0.5453 | 0.2186 | 0.5249 | 0.6628 | 0.8239 | 0.2115 | 0.2309 | 35.30 | 64.15 | 35.33 | 39.39 |

## Measured runtimes

- Feature bank: 40 files, 1884 s summed encode time (features_manifest.csv; excludes model loading).
- feature_bank_encode_seconds_sum: 1884.3 s
- compute_params_seconds: 25.9 s
- analyse_seconds: 1142.3 s

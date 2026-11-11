"""Compare the gradient-interference signal with grid12 ground truth.

Inputs (this folder): grad_signals.csv, ground_truth_grid12.csv
Outputs (this folder): analysis.md, analysis_table.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
N_PERM = 10_000
CAVEAT = ("Exploratory: 12 pairs from 4 datasets (not independent), single seed, K=4 gradient batches,\n"
          " first-order (one-step) approximation of a full epoch of training. Not a confirmation.")


def spearman_perm(x, y, n_perm=N_PERM, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n_perm)])
    p = (np.sum(np.abs(null) >= abs(rho) - 1e-12) + 1) / (n_perm + 1)
    return float(rho), float(p)


def dominance(iv, il):
    return (iv - il) / (np.abs(iv) + np.abs(il))


def main() -> None:
    g = pd.read_csv(HERE / "grad_signals.csv")
    t = pd.read_csv(HERE / "ground_truth_grid12.csv")
    df = g.merge(t[["O", "N", "F_V", "F_L", "total", "D", "gap_v", "gap_l"]], on=["O", "N"], how="inner")
    assert len(df) == 12, f"expected 12 merged pairs, got {len(df)}"
    df["Dp_adam"] = dominance(df.I_adam_V, df.I_adam_L)
    df["Dp_sgd"] = dominance(df.I_sgd_V, df.I_sgd_L)
    df["gap_diff"] = df.gap_v - df.gap_l
    gap_thr = float(df.gap_diff.median())
    truth = df.D > 0

    preds = {
        "adam": (df.Dp_adam, 0.0, df.I_adam_V, df.I_adam_L),
        "sgd": (df.Dp_sgd, 0.0, df.I_sgd_V, df.I_sgd_L),
        "gap": (df.gap_diff, gap_thr, None, None),
    }
    summary, loso = {}, {}
    for name, (score, thr, iv, il) in preds.items():
        correct = (score > thr) == truth
        df[f"correct_{name}"] = correct
        s = {"sign_correct": int(correct.sum()), "threshold": thr,
             "auroc": float(roc_auc_score(truth, score))}
        s["rho_Dp_D"], s["p_Dp_D"] = spearman_perm(score, df.D)
        if iv is not None:
            s["rho_IV_FV"], s["p_IV_FV"] = spearman_perm(iv, df.F_V)
            s["rho_IL_FL"], s["p_IL_FL"] = spearman_perm(il, df.F_L)
            s["rho_sum_total"], s["p_sum_total"] = spearman_perm(iv + il, df.total)
        else:
            s["rho_IV_FV"], s["p_IV_FV"] = spearman_perm(df.gap_v, df.F_V)
            s["rho_IL_FL"], s["p_IL_FL"] = spearman_perm(df.gap_l, df.F_L)
            s["rho_sum_total"], s["p_sum_total"] = spearman_perm(df.gap_v + df.gap_l, df.total)
        summary[name] = s
        loso[name] = {o: int(correct[df.O == o].sum()) for o in df.O.unique()}

    a, gp = summary["adam"], summary["gap"]
    if a["sign_correct"] >= 10 and a["auroc"] > 0.83 and a["p_Dp_D"] < 0.05:
        verdict = "promising"
    elif abs(a["sign_correct"] - gp["sign_correct"]) <= 1:
        verdict = "comparable"
    else:
        verdict = "not better"

    cols = ["O", "N", "I_adam_V", "I_adam_L", "Dp_adam", "F_V", "F_L", "D", "correct_adam",
            "I_sgd_V", "I_sgd_L", "Dp_sgd", "correct_sgd", "cos_V", "cos_L", "gap_diff", "correct_gap"]
    df[cols].to_csv(HERE / "analysis_table.csv", index=False)
    (HERE / "analysis_summary.json").write_text(
        json.dumps({"summary": summary, "loso": loso, "verdict": verdict}, indent=2) + "\n")

    batch, k = int(df.batch.iloc[0]), int(df.K.iloc[0])
    L = []
    L.append("# Gradient interference vs grid12 ground truth\n")
    L.append("## Method\n")
    L.append(f"For each source checkpoint theta_O (grid12, `D:\\CAPSTONE\\grid12\\ckpt`), the training loss "
             "(image vs all class prototypes of the dataset, 10 config templates, logits = 100 * img @ txt.T, "
             "cross-entropy; float32, logit_scale frozen) is differentiated on the TRAIN split. A gradient set "
             f"is the mean gradient over the first K={k} batches (batch {batch}, seed 0, no shuffling). "
             "Gradients are split into V = `visual.*` and L = all other parameters except logit_scale/logit_bias. "
             "No fine-tuning is performed.\n")
    L.append("- I_sgd_E  = -dot(g_O_E, g_N_E)")
    L.append("- I_adam_E = -dot(g_O_E, sign(g_N_E))  (primary; AdamW first step)")
    L.append("- cos_E = cosine(g_O_E, g_N_E)")
    L.append("- Dp = (I_V - I_L) / (|I_V| + |I_L|), for adam and sgd versions")
    L.append("- Prediction: Dp > 0 means vision-dominated forgetting (true D > 0). Threshold 0, no tuning.")
    L.append("- Gap baseline: gap_v - gap_l with its median as threshold (**data-tuned** threshold).\n")
    L.append("## Per-pair table (Adam signal, primary)\n")
    L.append("| O | N | I_adam_V | I_adam_L | Dp_adam | true F_V | true F_L | true D | correct? |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|:-:|")
    for _, r in df.iterrows():
        L.append(f"| {r.O} | {r.N} | {r.I_adam_V:.4g} | {r.I_adam_L:.4g} | {r.Dp_adam:+.3f} | "
                 f"{r.F_V:.2f} | {r.F_L:.2f} | {r.D:+.3f} | {'yes' if r.correct_adam else 'no'} |")
    L.append("")
    L.append("## Comparison vs gap baseline\n")
    L.append("| metric | Adam Dp | SGD Dp | gap_v - gap_l |")
    L.append("|---|---:|---:|---:|")
    sa, ss = summary["adam"], summary["sgd"]
    L.append(f"| sign accuracy (/12) | {sa['sign_correct']} | {ss['sign_correct']} | "
             f"{gp['sign_correct']} (median thr {gap_thr:+.4f}, data-tuned) |")
    L.append(f"| AUROC for D > 0 | {sa['auroc']:.3f} | {ss['auroc']:.3f} | {gp['auroc']:.3f} |")
    for key, lab in [("Dp_D", "Spearman score vs D"), ("IV_FV", "Spearman I_V vs F_V (gap: gap_v vs F_V)"),
                     ("IL_FL", "Spearman I_L vs F_L (gap: gap_l vs F_L)"),
                     ("sum_total", "Spearman I_V+I_L vs total (gap: gap_v+gap_l vs total)")]:
        L.append(f"| {lab} | {sa['rho_' + key]:+.3f} (p={sa['p_' + key]:.4f}) | "
                 f"{ss['rho_' + key]:+.3f} (p={ss['p_' + key]:.4f}) | "
                 f"{gp['rho_' + key]:+.3f} (p={gp['p_' + key]:.4f}) |")
    L.append(f"\np-values: two-sided permutation test, {N_PERM:,} permutations, seed 0.\n")
    L.append("## Leave-one-source-out (sign accuracy on the 3 pairs of each held-out O; no fitting)\n")
    L.append("| held-out O | Adam | SGD | gap (median thr from all 12) |")
    L.append("|---|:-:|:-:|:-:|")
    for o in df.O.unique():
        L.append(f"| {o} | {loso['adam'][o]}/3 | {loso['sgd'][o]}/3 | {loso['gap'][o]}/3 |")
    L.append("\nThe Adam/SGD signals use a fixed threshold of 0, so leave-one-source-out needs no refitting; "
             "the gap column reuses the median of all 12 pairs and is therefore not a true held-out number.\n")
    L.append("## Interpretation\n")
    L.append('Rule: "promising" if Adam sign accuracy >= 10/12 AND AUROC > 0.83 (gap baseline) AND Dp-vs-D '
             'p < 0.05; "comparable" if within ±1 correct pair of the gap baseline; otherwise "not better".\n')
    L.append(f"**Applies: {verdict}** (Adam {sa['sign_correct']}/12, AUROC {sa['auroc']:.3f}, "
             f"Dp-vs-D p = {sa['p_Dp_D']:.4f}; gap baseline {gp['sign_correct']}/12).\n")
    L.append("## Caveats\n")
    L.append("> " + CAVEAT.replace("\n", "\n> ") + "\n")
    (HERE / "analysis.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

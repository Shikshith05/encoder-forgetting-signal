"""Step 4: pre-registered analysis (D:\\CAPSTONE\\prereg_crossmodal.md). CPU only.

Reads pair_parameters.csv; writes analysis.md, primary_results.csv, exploratory_results.csv.
Spearman permutation p uses signal_eval.spearman_perm (copied unchanged from exp/fisher-consistency).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from signal_eval import spearman_perm, N_PERM  # noqa: E402

TARGETS = ["total", "F_V", "F_L", "D"]
ALPHA = 0.05 / 3


def within_group_perm(x, y, groups, n_perm=N_PERM, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(groups == g) for g in pd.unique(groups)]
    null = np.empty(n_perm)
    for k in range(n_perm):
        yp = y.copy()
        for ix in idx:
            yp[ix] = y[rng.permutation(ix)]
        null[k] = spearmanr(x, yp).statistic
    return float((np.sum(np.abs(null) >= abs(rho) - 1e-12) + 1) / (n_perm + 1))


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def cv_predict(df, cols, folds):
    y = df["total"].to_numpy(float)
    pred = np.empty(len(df))
    for test in folds:
        train = np.setdiff1d(np.arange(len(df)), test)
        if not cols:
            pred[test] = y[train].mean()
            continue
        X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
        beta, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        pred[test] = X[test] @ beta
    err = pred - y
    return float(np.abs(err).mean()), float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()), pred


def main() -> None:
    t0 = time.perf_counter()
    df = pd.read_csv(HERE / "pair_parameters.csv")
    groups = df["O"].to_numpy()
    L = ["# Cross-modal contribution and IIMM signals vs grid12 forgetting", "",
         "Pre-registration: `D:\\CAPSTONE\\prereg_crossmodal.md` (copy: prereg_crossmodal.md).", "",
         "> Exploratory: 12 non-independent pairs, single seed, a dev set reused by earlier signals. "
         "Nothing here is claimed as a working predictor; the result only decides what is "
         "pre-registered for Phase-II (30+ pairs).", ""]

    # ---------------- primary
    prim = []
    for pid, col in (("P1", "thA_CMS"), ("P2", "thA_IIMM_B")):
        rho, p = spearman_perm(df[col], df["total"])
        pw = within_group_perm(df[col], df["total"], groups)
        prim.append({"id": pid, "test": f"Spearman({col}, total)", "rho": rho, "p": p,
                     "p_within_A": pw, "significant": p < ALPHA})
    folds = {"LOPO": [np.array([i]) for i in range(12)],
             "LOODO": [np.flatnonzero(groups == g) for g in pd.unique(groups)]}
    models = {"P3: total ~ thA_IIMM_B + thA_CMS": ["thA_IIMM_B", "thA_CMS"],
              "baseline: training-fold mean": [],
              "baseline: total ~ grid12_zs_N": ["grid12_zs_N"]}
    cv = []
    for name, cols in models.items():
        for scheme, f in folds.items():
            mae, r2, pred = cv_predict(df, cols, f)
            cv.append({"model": name, "scheme": scheme, "MAE": mae, "R2": r2})
            if name.startswith("P3"):
                df[f"pred_P3_{scheme}"] = pred
    cvd = pd.DataFrame(cv)
    better = {s: all(cvd.query("scheme==@s and model.str.startswith('P3')", engine="python")["MAE"].iloc[0]
                     < cvd.query("scheme==@s and model.str.startswith('baseline')", engine="python")["MAE"])
              for s in folds}
    p3_better = all(better.values())
    pd.DataFrame(prim).to_csv(HERE / "primary_results.csv", index=False)
    cvd.to_csv(HERE / "p3_cv_results.csv", index=False)

    L += ["## Primary tests (Bonferroni alpha = 0.0167)", "",
          "| id | test | rho | perm p | within-A perm p | significant? |", "|---|---|:-:|:-:|:-:|:-:|"]
    for r in prim:
        L.append(f"| {r['id']} | {r['test']} | {r['rho']:+.3f} | {r['p']:.4f} | {r['p_within_A']:.4f} | "
                 f"{'yes' if r['significant'] else 'no'} |")
    L += ["", "P3 held-out error (same folds for every model):", "",
          "| model | scheme | MAE (pp) | R^2 |", "|---|---|:-:|:-:|"]
    for r in cv:
        L.append(f"| {r['model']} | {r['scheme']} | {r['MAE']:.2f} | {r['R2']:+.3f} |")
    L += ["", "Stated in chat, no file: mean predictor 15.2 pp, zero-shot alone 13.8 pp "
          "(held-out MAE). Only the recomputed values above are used for the verdict.",
          f"", f"P3 better than both baselines: LOPO {'yes' if better['LOPO'] else 'no'}, "
          f"LOODO {'yes' if better['LOODO'] else 'no'} -> **{'BETTER' if p3_better else 'NOT better'}**", ""]

    # baselines recomputed with this code (zs within-A p was stated without a source file)
    zr, zp = spearman_perm(df["grid12_zs_N"], df["total"])
    zw = within_group_perm(df["grid12_zs_N"], df["total"], groups)
    L += ["## Already-measured baselines", "",
          "| signal | rho | perm p | within-A p | source |", "|---|:-:|:-:|:-:|---|",
          f"| zero-shot on B (grid12_zs_N) | -0.583 | 0.0553 | 0.013 (stated in chat, no file) | grid12 analysis.md 5c |",
          f"| zero-shot on B, recomputed here | {zr:+.3f} | {zp:.4f} | {zw:.4f} | this script |",
          "| IIMM pretrained (grid12_iimm_N) | -0.086 | 0.8119 | - | grid12 analysis.md 5d |",
          "| Gap_V (A TEST images) | -0.184 | 0.5824 | - | grid12 analysis.md 5a |",
          "| Gap_L | -0.353 | 0.2659 | - | grid12 analysis.md 5b |", "",
          "Note: the old grid12 Gap_V / Gap_L (and zs_N, iimm_N) baselines were computed by grid12.py "
          "`step_signals` on TEST features, i.e. Gap_V used A's TEST images. This experiment's "
          "pre_/thA_ Gap_V and Gap_L use TRAIN images only; A's TEST split feeds no signal here.", ""]

    # ---------------- exploratory
    params = [c for c in df.columns if c.startswith(("pre_", "thA_")) and not c.startswith("pred_")]
    kept, dropped = [], []
    for c in params:
        v = df[c].to_numpy(float)
        if c.endswith("cross_inter"):  # == Con_TI by definition (approved: test once)
            diff = np.abs(v - df[c.replace("cross_inter", "Con_TI")].to_numpy(float)).max()
            dropped.append((c, f"= {c.replace('cross_inter', 'Con_TI')} by definition, "
                               f"max float diff {diff:.1e}"))
            continue
        if np.all(v == v[0]):
            dropped.append((c, "constant"))
            continue
        dup = next((k for k in kept if np.array_equal(df[k].to_numpy(float), v)), None)
        if dup:
            dropped.append((c, f"identical to {dup}"))
            continue
        kept.append(c)
    ex = []
    for c in kept:
        for t in TARGETS:
            rho, p = spearman_perm(df[c], df[t])
            ex.append({"parameter": c, "target": t, "rho": rho, "p": p,
                       "p_within_A": within_group_perm(df[c], df[t], groups),
                       "primary": c in ("thA_CMS", "thA_IIMM_B") and t == "total"})
    exd = pd.DataFrame(ex)
    exd["p_BH"] = bh(exd["p"])
    exd = exd.sort_values(["p_BH", "p"]).reset_index(drop=True)
    exd.to_csv(HERE / "exploratory_results.csv", index=False)
    L += [f"## Exploratory ({len(kept)} parameters x {len(TARGETS)} targets = {len(exd)} tests; "
          "BH over all; sorted by adjusted p)", "",
          "Dropped before testing: " + "; ".join(f"`{c}` ({why})" for c, why in dropped), "",
          "| parameter | target | rho | perm p | within-A p | BH p |", "|---|---|:-:|:-:|:-:|:-:|"]
    for _, r in exd.iterrows():
        mark = " (P)" if r["primary"] else ""
        L.append(f"| {r['parameter']}{mark} | {r['target']} | {r['rho']:+.3f} | {r['p']:.4f} | "
                 f"{r['p_within_A']:.4f} | {r['p_BH']:.4f} |")

    # ---------------- per pair
    main_cols = ["thA_CMS", "thA_IIMM_B", "pre_CMS", "pre_IIMM_B", "thA_Con_II", "thA_Con_TT",
                 "thA_Con_IT", "thA_Con_TI", "grid12_zs_N", "total", "pred_P3_LOPO", "pred_P3_LOODO"]
    L += ["", "## Per pair (main parameters)", "",
          "| O | N | " + " | ".join(main_cols) + " |", "|---|---|" + ":-:|" * len(main_cols)]
    for _, r in df.iterrows():
        L.append(f"| {r['O']} | {r['N']} | " + " | ".join(
            f"{r[c]:.2f}" if abs(r[c]) >= 10 else f"{r[c]:.4f}" for c in main_cols) + " |")

    # ---------------- runtimes
    man = pd.read_csv(HERE / "features_manifest.csv")
    rt = json.loads((HERE / "runtime.json").read_text()) if (HERE / "runtime.json").exists() else {}
    rt["analyse_seconds"] = round(time.perf_counter() - t0, 1)
    (HERE / "runtime.json").write_text(json.dumps(rt, indent=2))
    L += ["", "## Measured runtimes", "",
          f"- Feature bank: {len(man)} files, {man['seconds'].sum():.0f} s summed encode time "
          f"(features_manifest.csv; excludes model loading)."]
    L += [f"- {k}: {v} s" for k, v in rt.items()]
    (HERE / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()

"""Shared evaluation for pre-registered signals (prereg_signals_v2.md, section 2).

Rules: predict "vision" if score > 0; 0 or NaN counts as wrong and enters AUROC as 0.
Metrics: sign accuracy /12, AUROC for (D > 0), Spearman vs D with two-sided permutation p
(10,000 permutations, numpy seed 0), leave-one-source-out sign accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

N_PERM = 10_000
DATASETS = ["OxfordIIITPet", "Flowers102", "EuroSAT", "DTD"]
BASELINES = [
    {"name": "gap_v - gap_l (threshold tuned on same data)", "sign": "10/12", "auroc": "0.833",
     "rho": "+0.501 (p=0.10)"},
    {"name": "gradient interference, Adam (exp/gradient-interference)", "sign": "7/12",
     "auroc": "0.486", "rho": "+0.018 (p=0.96)"},
]


def spearman_perm(x, y, n_perm: int = N_PERM, seed: int = 0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), float("nan")
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n_perm)])
    p = (np.sum(np.abs(null) >= abs(rho) - 1e-12) + 1) / (n_perm + 1)
    return float(rho), float(p)


def dominance(v, l):
    v, l = np.asarray(v, float), np.asarray(l, float)
    den = np.abs(v) + np.abs(l)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (v - l) / den, np.nan)


def evaluate(df: pd.DataFrame, score_col: str) -> dict:
    """df must have O, N, D. Returns metrics for one score column."""
    s = df[score_col].astype(float).to_numpy()
    truth = (df["D"].to_numpy() > 0)
    pred_vision = np.nan_to_num(s, nan=0.0) > 0
    valid = ~np.isnan(s) & (s != 0)
    correct = valid & (pred_vision == truth)
    rho, p = spearman_perm(s, df["D"])
    loso = {o: int(correct[(df["O"] == o).to_numpy()].sum()) for o in DATASETS}
    return {"score": score_col, "sign_correct": int(correct.sum()),
            "auroc": float(roc_auc_score(truth, np.nan_to_num(s, nan=0.0))),
            "rho_D": rho, "p_D": p, "loso": loso, "correct": correct}


def dev_pass(m: dict) -> bool:
    return m["sign_correct"] >= 11 or m["auroc"] > 0.833


def fmt_rho(rho, p):
    return "n/a" if np.isnan(rho) else f"{rho:+.3f} (p={p:.4f})"


def metrics_table(results: list[dict], primary: str) -> list[str]:
    L = ["| score | sign acc | AUROC (D>0) | Spearman vs D (perm p) | dev pass? |",
         "|---|:-:|:-:|:-:|:-:|"]
    for m in results:
        tag = " (PRIMARY)" if m["score"] == primary else " (secondary)"
        L.append(f"| {m['score']}{tag} | {m['sign_correct']}/12 | {m['auroc']:.3f} | "
                 f"{fmt_rho(m['rho_D'], m['p_D'])} | "
                 f"{('yes' if dev_pass(m) else 'no') if m['score'] == primary else '-'} |")
    for b in BASELINES:
        L.append(f"| baseline: {b['name']} | {b['sign']} | {b['auroc']} | {b['rho']} | - |")
    return L


def loso_table(results: list[dict]) -> list[str]:
    L = ["| held-out O | " + " | ".join(m["score"] for m in results) + " |",
         "|---|" + ":-:|" * len(results)]
    for o in DATASETS:
        L.append(f"| {o} | " + " | ".join(f"{m['loso'][o]}/3" for m in results) + " |")
    return L


def extra_spearman(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> list[str]:
    L = ["| signal column | ground truth | Spearman (perm p) |", "|---|---|:-:|"]
    for a, b in pairs:
        rho, p = spearman_perm(df[a], df[b])
        L.append(f"| {a} | {b} | {fmt_rho(rho, p)} |")
    return L

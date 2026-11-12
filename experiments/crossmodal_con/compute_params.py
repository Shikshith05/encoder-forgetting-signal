"""Step 2: log every parameter for the 12 grid12 pairs (A = old, B = new). CPU only.

Reads the feature bank (build_features.py). Every image-based parameter uses TRAIN images,
except zs_acc_B_test (B's TEST split). A's TEST split is never read here.
Each parameter is computed under pretrained CLIP (prefix pre_) and under theta_A (prefix thA_).
Output: pair_parameters.csv (12 rows, merged with D:\\CAPSTONE\\grid12\\results\\grid12_results.csv).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(r"D:\CAPSTONE\encoder-forgetting-signal")
FEAT = Path(r"D:\CAPSTONE\grid12\cache\features")
GT = Path(r"D:\CAPSTONE\grid12\results\grid12_results.csv")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.signal_gap import compute_gap, compute_iimm_score, compute_zeroshot_accuracy  # noqa: E402

DATASETS = ["OxfordIIITPet", "Flowers102", "EuroSAT", "DTD"]
PAIRS = [(a, b) for a in DATASETS for b in DATASETS if a != b]
SCALE = 100.0


def load(model: str, ds: str, split: str):
    assert split == "train" or split == "test"
    z = np.load(FEAT / f"{model}_{ds}_{split}.npz", allow_pickle=False)
    return z["I"].astype(np.float64), z["T"].astype(np.float64), z["labels"]


def mean_pairwise(X):  # mean cosine over distinct pairs of unit rows
    n = len(X)
    s = X.sum(0)
    return float((s @ s - (X * X).sum()) / (n * (n - 1)))


def task_stats(I, T, y, tag):
    S = I @ T.T
    n = len(y)
    true = S[np.arange(n), y]
    wrong = S.copy()
    wrong[np.arange(n), y] = -np.inf
    logits = SCALE * S
    logits -= logits.max(1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(1, keepdims=True)
    iimm = compute_iimm_score(I, T, y)
    return {
        f"S_inter_{tag}": iimm["inter"], f"S_intra_{tag}": iimm["intra"],
        f"IIMM_{tag}": iimm["iimm"],
        f"zs_acc_{tag}_train": compute_zeroshot_accuracy(I, T, y),
        f"margin_{tag}": float((true - wrong.max(1)).mean()),
        f"maxprob_{tag}": float(P.max(1).mean()),
        f"entropy_{tag}": float((-(P * np.log(np.clip(P, 1e-300, None))).sum(1)).mean()),
    }


def pair_params(model: str, a: str, b: str) -> dict:
    IA, TA, yA = load(model, a, "train")
    IB, TB, yB = load(model, b, "train")
    IBt, TBt, yBt = load(model, b, "test")
    assert np.allclose(TB, TBt)
    r = {}
    r.update(task_stats(IB, TB, yB, "B"))
    r["zs_acc_B_test"] = compute_zeroshot_accuracy(IBt, TB, yBt)
    r["text_intra_B"] = mean_pairwise(TB)
    r["modality_gap_B"] = float(np.linalg.norm(IB.mean(0) - TB.mean(0)))
    r["n_classes_B"] = int(len(TB))
    sa = task_stats(IA, TA, yA, "A")
    for k in ("S_inter_A", "S_intra_A", "IIMM_A", "zs_acc_A_train", "margin_A"):
        r[k] = sa[k]
    r["Con_II"] = float(IA.mean(0) @ IB.mean(0))
    r["Con_TT"] = float(TA.mean(0) @ TB.mean(0))
    r["Con_IT"] = float(IA.mean(0) @ TB.mean(0))
    r["Con_TI"] = float(TA.mean(0) @ IB.mean(0))
    r["CMS"] = (r["Con_IT"] + r["Con_TI"]) / 2
    r["CMS4"] = (r["Con_II"] + r["Con_TT"] + r["Con_IT"] + r["Con_TI"]) / 4
    SBA = IB @ TA.T
    r["cross_inter"] = float(SBA.mean())
    r["nearest_old_label"] = float(SBA.max(1).mean())
    r["text_overlap"] = float((TA @ TB.T).max(1).mean())
    r["Gap_V"] = compute_gap(IA, IB)
    r["Gap_L"] = compute_gap(TA, TB)
    return r


def main() -> None:
    t0 = time.perf_counter()
    rows = []
    for a, b in PAIRS:
        row = {"O": a, "N": b}
        for prefix, model in (("pre_", "pre"), ("thA_", f"theta_{a}")):
            row.update({prefix + k: v for k, v in pair_params(model, a, b).items()})
        rows.append(row)
        print(f"{a}->{b} done", flush=True)
    P = pd.DataFrame(rows)
    gt = pd.read_csv(GT).rename(columns={c: f"grid12_{c}" for c in
                                         ("gap_v", "gap_l", "zs_N", "iimm_N")})
    out = P.merge(gt, on=["O", "N"], how="left", validate="one_to_one")
    assert len(out) == 12 and out["total"].notna().all()
    out.to_csv(HERE / "pair_parameters.csv", index=False)
    print(f"seconds={time.perf_counter() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()

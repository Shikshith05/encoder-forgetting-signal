"""Evaluate this folder's signal against grid12 ground truth (prereg_signals_v2.md, section 2).

Same file in experiments/{fisher_consistency, early_probe, fragility}; settings are picked by
folder name. Writes analysis.md, analysis_table.csv and analysis_summary.json here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from signal_eval import (evaluate, dev_pass, metrics_table, loso_table,  # noqa: E402
                         extra_spearman)

CONFIG = {
    "fisher_consistency": {
        "title": "Signal A: Fisher x Adam-consistency (no training)",
        "csv": "fisher_signals.csv", "primary": "D_A",
        "secondary": ["D_A2_paramnorm", "D_A3_fisher_only", "D_A4_cons_only"],
        "table_cols": ["S_V", "S_L"],
        "extra": [("S_V", "F_V"), ("S_L", "F_L"), ("S_V", "total"), ("S_L", "total")],
        "notes": [
            "D_A2 (parameter-count normalised) multiplies S_V/S_L by n_L/n_V = 63,428,096/87,849,216 "
            "= 0.722. Pairs with 1 < S_V/S_L < 1.385 therefore change sign: here Flowers102->EuroSAT "
            "(ratio 1.193; correct under D_A, wrong under D_A2) and Flowers102->DTD (ratio 1.190; "
            "wrong under D_A, correct under D_A2). The total stays 7/12 by coincidence. "
            "D_A2 is a monotone function of D_A, so AUROC and Spearman are identical; the sign "
            "accuracy is not guaranteed to be.",
        ],
    },
    "early_probe": {
        "title": "Signal B: early probe (8 training steps)",
        "csv": "probe_signals.csv", "primary": "D_B",
        "secondary": ["D_B2_acc"],
        "table_cols": ["M_F_V", "M_F_L", "effective_updates"],
        "extra": [("M_F_V", "F_V"), ("M_F_L", "F_L"), ("M_total", "total"),
                  ("probe_total", "total")],
    },
    "fragility": {
        "title": "Signal C: old-domain fragility (forward only)",
        "csv": "fragility_signals.csv", "primary": "D_C",
        "secondary": ["D_C_s0.25", "D_C_s0.5"],
        "table_cols": ["drop_V", "drop_L"],
        "extra": [("drop_V", "F_V"), ("drop_L", "F_L")],
    },
}
CAVEAT = ("Exploratory: 12 pairs from 4 datasets (not independent), single seed. These 12 pairs are a "
          "development set already used by the gap and gradient-interference signals. A dev pass is "
          "'promising on dev only', not a confirmation.")


def main() -> None:
    c = CONFIG[HERE.name]
    sig = pd.read_csv(HERE / c["csv"])
    gt = pd.read_csv(HERE / "ground_truth_grid12.csv")
    df = sig.merge(gt[["O", "N", "F_V", "F_L", "total", "D"]], on=["O", "N"], how="inner")
    assert len(df) == 12, f"expected 12 pairs, got {len(df)}"
    results = [evaluate(df, s) for s in [c["primary"]] + c["secondary"]]
    prim = results[0]
    df["correct_primary"] = prim["correct"]
    passed = dev_pass(prim)

    keep = ["O", "N"] + c["table_cols"] + [c["primary"]] + c["secondary"] + ["F_V", "F_L", "D", "correct_primary"]
    df[keep].to_csv(HERE / "analysis_table.csv", index=False)
    summary = {"primary": c["primary"], "dev_pass": bool(passed),
               "results": [{k: v for k, v in m.items() if k != "correct"} for m in results]}
    (HERE / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    L = [f"# {c['title']}\n",
         "Pre-registration: `prereg_signals_v2.md` (this folder; written before the run).",
         "Prediction: vision if score > 0 (0/NaN = wrong). DEV PASS = primary sign accuracy >= 11/12 "
         "OR primary AUROC > 0.833.\n",
         "## Per-pair table\n",
         "| O | N | " + " | ".join(c["table_cols"]) + f" | {c['primary']} | true F_V | true F_L | true D | correct? |",
         "|---|---|" + "---:|" * (len(c["table_cols"]) + 4) + ":-:|"]
    for _, r in df.iterrows():
        vals = " | ".join(f"{r[k]:.4g}" for k in c["table_cols"])
        L.append(f"| {r.O} | {r.N} | {vals} | {r[c['primary']]:+.3f} | {r.F_V:.2f} | {r.F_L:.2f} | "
                 f"{r.D:+.3f} | {'yes' if r.correct_primary else 'no'} |")
    L += ["", "## Secondary scores per pair (reported only)\n",
          "| O | N | " + " | ".join(f"{s} | correct?" for s in c["secondary"]) + " |",
          "|---|---|" + "---:|:-:|" * len(c["secondary"])]
    for i, (_, r) in enumerate(df.iterrows()):
        cells = " | ".join(f"{r[m['score']]:+.3f} | {'yes' if m['correct'][i] else 'no'}"
                           for m in results[1:])
        L.append(f"| {r.O} | {r.N} | {cells} |")
    L += ["", "## Metrics vs baselines\n"] + metrics_table(results, c["primary"])
    L += ["\nSpearman p: two-sided permutation, 10,000 permutations, seed 0.\n",
          "## Leave-one-source-out (sign accuracy on the 3 pairs of each held-out O; nothing fitted)\n"]
    L += loso_table(results)
    L += ["\n## Extra Spearman (reported only, not used for pass)\n"] + extra_spearman(df, c["extra"])
    if c.get("notes"):
        L += ["\n## Notes\n"] + [f"- {n}" for n in c["notes"]]
    L += ["\n## Verdict\n",
          f"**{'DEV PASS: promising on dev only' if passed else 'DEV FAIL: negative result'}** "
          f"(primary {c['primary']}: {prim['sign_correct']}/12, AUROC {prim['auroc']:.3f}).\n",
          "## Caveats\n", "> " + CAVEAT + "\n"]
    (HERE / "analysis.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

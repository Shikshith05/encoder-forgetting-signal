"""Signal C: old-domain fragility (prereg_signals_v2.md, section 5). Forward only.

At theta_O on the first 512 images of O's TRAIN split: add Gaussian noise of norm ~sigma to
the unit image embeddings (drop_V) or to the unit class prototypes (drop_L), renormalise,
measure the accuracy drop (mean over 20 draws, fresh numpy default_rng(0) per (O, sigma);
per draw the image noise first, then the text noise).
PRIMARY: sigma = 1.0, D_C = (drop_V - drop_L) / (|drop_V| + |drop_L|).
Same score for all 3 pairs of a source O.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
CKPT = Path(r"D:\CAPSTONE\grid12\ckpt")
OUT_CSV = HERE / "fragility_signals.csv"
RUNTIME_JSON = HERE / "runtime.json"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src.common import (load_config, load_clip, get_dataset_loader, free_model,  # noqa: E402
                        extract_image_features, extract_text_features)

DATASETS = ["OxfordIIITPet", "Flowers102", "EuroSAT", "DTD"]
SIGMAS = [1.0, 0.25, 0.5]  # 1.0 is primary
DRAWS = 20
N_IMGS = 512
DEVICE = "cuda"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def acc(I, T, labels):
    return float(((I @ T.T).argmax(1) == labels).mean() * 100)


def fragility(I, T, labels, sigma):
    rng = np.random.default_rng(0)
    d = I.shape[1]
    clean = acc(I, T, labels)
    av, al = [], []
    for _ in range(DRAWS):
        eI = rng.normal(0.0, sigma / np.sqrt(d), size=I.shape)
        eT = rng.normal(0.0, sigma / np.sqrt(d), size=T.shape)
        av.append(acc(unit(I + eI), T, labels))
        al.append(acc(I, unit(T + eT), labels))
    drop_v, drop_l = clean - np.mean(av), clean - np.mean(al)
    den = abs(drop_v) + abs(drop_l)
    return clean, drop_v, drop_l, ((drop_v - drop_l) / den if den > 0 else float("nan"))


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    t_start = time.perf_counter()
    cfg = copy.deepcopy(load_config(str(REPO / "configs" / "config.yaml")))
    cfg["data"]["root"] = str(REPO / "data")
    cfg["finetune"]["batch_size"] = 64
    rows = []
    for o in DATASETS:
        t0 = time.perf_counter()
        model, preprocess, tokenizer = load_clip(str(CKPT / f"theta_{o}.pt"), device=DEVICE, cfg=cfg)
        loader, _ids, classes = get_dataset_loader(o, preprocess, cfg, "train")
        sub = DataLoader(Subset(loader.dataset, range(N_IMGS)), batch_size=64, shuffle=False,
                         num_workers=cfg["data"].get("num_workers", 2), pin_memory=True)
        I, labels = extract_image_features(model, sub, DEVICE)
        T = extract_text_features(model, tokenizer, classes, cfg["prompts"], DEVICE)
        free_model(model)
        torch.cuda.empty_cache()
        I, T = I.astype(np.float64), T.astype(np.float64)
        res = {}
        for s in SIGMAS:
            clean, dv, dl, dc = fragility(I, T, labels, s)
            res[s] = (clean, dv, dl, dc)
            log(f"O={o} sigma={s}: clean={clean:.2f} drop_V={dv:.2f} drop_L={dl:.2f} D_C={dc:+.3f}")
        sec = time.perf_counter() - t0
        for n in DATASETS:
            if n == o:
                continue
            row = {"O": o, "N": n, "clean_acc": res[1.0][0],
                   "drop_V": res[1.0][1], "drop_L": res[1.0][2], "D_C": res[1.0][3],
                   "D_C_s0.25": res[0.25][3], "D_C_s0.5": res[0.5][3],
                   "drop_V_s0.25": res[0.25][1], "drop_L_s0.25": res[0.25][2],
                   "drop_V_s0.5": res[0.5][1], "drop_L_s0.5": res[0.5][2],
                   "n_imgs": len(labels), "n_classes": T.shape[0], "seconds_O": sec}
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    secs = time.perf_counter() - t_start
    rt = json.loads(RUNTIME_JSON.read_text()) if RUNTIME_JSON.exists() else {"sessions": []}
    rt["sessions"].append({"seconds": secs, "gpu": torch.cuda.get_device_name(0),
                           "torch": torch.__version__, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    RUNTIME_JSON.write_text(json.dumps(rt, indent=2) + "\n")
    log(f"done in {secs / 60:.1f} min")


if __name__ == "__main__":
    main()

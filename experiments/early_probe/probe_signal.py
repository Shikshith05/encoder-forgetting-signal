"""Signal B: early probe (prereg_signals_v2.md, section 4). Uses 8 training steps.

For each pair O->N: start from theta_O, run the unchanged repo train_model (AdamW, lr 3e-5,
wd 0.1, batch 64, autocast + GradScaler, seed 0) on a loader holding only the first 8 batches
(first 512 images) of N's TRAIN loader, in the same order (= the first 8 steps of the real run).
Then swap theta_O vs probe model on the first 512 images of O's TRAIN split (repo
swap_from_features) and take the margin-Shapley split.
PRIMARY D_B = (M_F_V - M_F_L) / (|M_F_V| + |M_F_L|). Saved after every pair (resumable).
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
OUT_CSV = HERE / "probe_signals.csv"
RUNTIME_JSON = HERE / "runtime.json"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src.common import (load_config, load_clip, get_dataset_loader, free_model,  # noqa: E402
                        extract_image_features, extract_text_features)
from src.finetune import train_model, seed_everything  # noqa: E402
from src.swap_shapley import swap_from_features  # noqa: E402

DATASETS = ["OxfordIIITPet", "Flowers102", "EuroSAT", "DTD"]
LR_PAIR = 3e-5
BATCH = 64
PROBE_STEPS = 8
PROBE_IMGS = 512
DEVICE = "cuda"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cfg_base() -> dict:
    cfg = copy.deepcopy(load_config(str(REPO / "configs" / "config.yaml")))
    cfg["data"]["root"] = str(REPO / "data")
    cfg["finetune"]["batch_size"] = BATCH
    cfg["finetune"]["lr"] = LR_PAIR
    assert float(cfg["model"]["logit_scale"]) == 100.0
    assert int(cfg["finetune"]["epochs"]) == 1 and int(cfg["data"]["seed"]) == 0
    return cfg


def first_n_loader(loader, n: int, num_workers: int) -> DataLoader:
    """Same dataset order and batch size as loader, restricted to the first n images."""
    return DataLoader(Subset(loader.dataset, range(n)), batch_size=loader.batch_size,
                      shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def probe_features(model, preprocess, tokenizer, o: str, cfg: dict):
    loader, ids, classes = get_dataset_loader(o, preprocess, cfg, "train")
    sub = first_n_loader(loader, PROBE_IMGS, cfg["data"].get("num_workers", 2))
    I, labels = extract_image_features(model, sub, DEVICE)
    T = extract_text_features(model, tokenizer, classes, cfg["prompts"], DEVICE)
    return I, T, labels, ids[:PROBE_IMGS]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    t_start = time.perf_counter()
    cfg = cfg_base()
    rows, done = [], set()
    if OUT_CSV.exists():
        rows = pd.read_csv(OUT_CSV).to_dict("records")
        done = {(r["O"], r["N"]) for r in rows}
    for o in DATASETS:
        todo = [n for n in DATASETS if n != o and (o, n) not in done]
        if not todo:
            log(f"O={o}: done, skip")
            continue
        ck = str(CKPT / f"theta_{o}.pt")
        model, preprocess, tokenizer = load_clip(ck, device=DEVICE, cfg=cfg)
        I_p, T_p, labels, ids = probe_features(model, preprocess, tokenizer, o, cfg)
        free_model(model)
        torch.cuda.empty_cache()
        log(f"O={o}: theta_O probe features I{I_p.shape} T{T_p.shape}")
        for n in todo:
            t0 = time.perf_counter()
            seed_everything(0)
            model, preprocess, tokenizer = load_clip(ck, device=DEVICE, cfg=cfg)
            loader, _ids, classes = get_dataset_loader(n, preprocess, cfg, "train")
            sub = first_n_loader(loader, PROBE_STEPS * BATCH, cfg["data"].get("num_workers", 2))
            assert len(sub) == PROBE_STEPS
            stats = train_model(model, sub, tokenizer, classes, cfg["prompts"], cfg, DEVICE)
            I_f, T_f, labels_f, ids_f = probe_features(model, preprocess, tokenizer, o, cfg)
            free_model(model)
            torch.cuda.empty_cache()
            if not (np.array_equal(labels, labels_f) and list(ids) == list(ids_f)):
                raise RuntimeError("probe image order differs")
            row, _preds = swap_from_features(I_p, I_f, T_p, T_f, labels, f"{o}->{n}", ids,
                                             scale=100.0, threshold=0.9)
            mv, ml = row["M_F_V"], row["M_F_L"]
            den = abs(mv) + abs(ml)
            av, al = row["F_V"], row["F_L"]
            den_a = abs(av) + abs(al)
            out = {"O": o, "N": n,
                   "D_B": (mv - ml) / den if den > 0 else float("nan"),
                   "D_B2_acc": (av - al) / den_a if den_a > 0 else float("nan"),
                   "M_F_V": mv, "M_F_L": ml, "M_total": row["M_total"],
                   "M_interaction": row["M_interaction"],
                   "A_pp": row["A_pp"], "A_pf": row["A_pf"], "A_fp": row["A_fp"], "A_ff": row["A_ff"],
                   "probe_F_V": av, "probe_F_L": al, "probe_total": row["total"],
                   "steps": PROBE_STEPS, "steps_skipped": stats["optimizer_steps_skipped"],
                   "effective_updates": PROBE_STEPS - stats["optimizer_steps_skipped"],
                   "train_seconds": stats["train_seconds"], "probe_train_loss": stats["train_loss"],
                   "pair_seconds": time.perf_counter() - t0,
                   "peak_gpu_gb": stats["peak_gpu_memory_gb"]}
            if out["effective_updates"] == 0:
                out["D_B"] = float("nan")
            rows.append(out)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            log(f"  {o}->{n}: D_B={out['D_B']:+.3f} (M_F_V={mv:.4g}, M_F_L={ml:.4g}), "
                f"skipped {out['steps_skipped']}/{PROBE_STEPS}, acc total {row['total']:.2f} pp, "
                f"{out['pair_seconds']:.0f}s")
    secs = time.perf_counter() - t_start
    rt = json.loads(RUNTIME_JSON.read_text()) if RUNTIME_JSON.exists() else {"sessions": []}
    rt["sessions"].append({"seconds": secs, "gpu": torch.cuda.get_device_name(0),
                           "torch": torch.__version__, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    RUNTIME_JSON.write_text(json.dumps(rt, indent=2) + "\n")
    log(f"done in {secs / 60:.1f} min")


if __name__ == "__main__":
    main()

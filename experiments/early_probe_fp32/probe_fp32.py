"""Signal B2-fp32: early probe trained in fp32 (prereg_signal_b_fp32.md).

Identical to signal B (exp/early-probe) except: training is fp32 (no autocast, no GradScaler),
using train_fp32 below, a copy of src.finetune.train_model with only autocast and GradScaler
removed (src/ is not modified). If batch 64 runs out of GPU memory on the first pair, all pairs
use gradient accumulation 4 x 16 (same update). Real optimizer updates are counted; must be 8.
PRIMARY D_B2fp32 = (M_F_V - M_F_L) / (|M_F_V| + |M_F_L|). Saved after every pair (resumable).
"""
from __future__ import annotations

import copy
import gc
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
OUT_CSV = HERE / "probe_fp32_signals.csv"
RUNTIME_JSON = HERE / "runtime.json"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src.common import (load_config, load_clip, get_dataset_loader, free_model,  # noqa: E402
                        extract_image_features, extract_text_features)
from src.finetune import (seed_everything, configure_trainable, prompt_features,  # noqa: E402
                          tokenize_classes)
from src.swap_shapley import swap_from_features  # noqa: E402

DATASETS = ["OxfordIIITPet", "Flowers102", "EuroSAT", "DTD"]
LR_PAIR = 3e-5
BATCH = 64
PROBE_STEPS = 8
PROBE_IMGS = 512
ACCUM = 4  # fallback: 4 micro-batches of 16
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


def train_fp32(model, loader, tokenizer, classes, templates, cfg, device, accum: int) -> dict:
    """Copy of src.finetune.train_model (text_frozen=False) without autocast and GradScaler.

    accum = 1: one forward/backward per batch of 64. accum = 4: each batch of 64 is split into
    4 micro-batches of 16, each loss divided by 4, one optimizer step per batch (same update).
    """
    settings = cfg["finetune"]
    seed_everything(cfg["data"]["seed"])
    configure_trainable(model, False)
    model.train()
    if hasattr(model.visual, "set_grad_checkpointing"):
        model.visual.set_grad_checkpointing(True)
    scale_before = model.logit_scale.detach().clone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(settings["lr"]), weight_decay=float(settings.get("weight_decay", 0.1)),
    )
    tokens = tokenize_classes(tokenizer, classes, templates, device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss_sum, correct, count, updates = 0.0, 0, 0, 0
    for epoch in range(int(settings["epochs"])):
        for step, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            for img_mb, lab_mb in zip(images.chunk(accum), labels.chunk(accum)):
                image_vectors = F.normalize(model.encode_image(img_mb).float(), dim=-1)
                text_vectors = prompt_features(model, tokens, len(classes), len(templates),
                                               checkpoint_text=True)
                logits = model.logit_scale.exp().float() * image_vectors @ text_vectors.T
                loss = F.cross_entropy(logits, lab_mb)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss.")
                (loss / accum).backward()
                loss_sum += loss.detach().item() * lab_mb.numel()
                correct += (logits.argmax(dim=1) == lab_mb).sum().item()
                count += lab_mb.numel()
            optimizer.step()
            updates += 1
            print(f"  step {step}/{len(loader)}, loss={loss_sum / count:.4f}", flush=True)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if not torch.equal(scale_before, model.logit_scale.detach()):
        raise AssertionError("logit_scale changed.")
    model.eval()
    if hasattr(model.visual, "set_grad_checkpointing"):
        model.visual.set_grad_checkpointing(False)
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    del optimizer
    return {"train_seconds": seconds, "train_loss": loss_sum / count, "updates": updates,
            "train_images": count, "peak_gpu_memory_gb": peak}


def first_n_loader(loader, n: int, num_workers: int) -> DataLoader:
    return DataLoader(Subset(loader.dataset, range(n)), batch_size=loader.batch_size,
                      shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def probe_features(model, preprocess, tokenizer, o: str, cfg: dict):
    loader, ids, classes = get_dataset_loader(o, preprocess, cfg, "train")
    sub = first_n_loader(loader, PROBE_IMGS, cfg["data"].get("num_workers", 2))
    I, labels = extract_image_features(model, sub, DEVICE)
    T = extract_text_features(model, tokenizer, classes, cfg["prompts"], DEVICE)
    return I, T, labels, ids[:PROBE_IMGS]


def run_pair(ck: str, o: str, n: str, cfg: dict, accum: int):
    seed_everything(0)
    model, preprocess, tokenizer = load_clip(ck, device=DEVICE, cfg=cfg)
    try:
        loader, _ids, classes = get_dataset_loader(n, preprocess, cfg, "train")
        sub = first_n_loader(loader, PROBE_STEPS * BATCH, cfg["data"].get("num_workers", 2))
        assert len(sub) == PROBE_STEPS
        stats = train_fp32(model, sub, tokenizer, classes, cfg["prompts"], cfg, DEVICE, accum)
        I_f, T_f, labels_f, ids_f = probe_features(model, preprocess, tokenizer, o, cfg)
    finally:
        model.zero_grad(set_to_none=True)
        free_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    return stats, I_f, T_f, labels_f, ids_f


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    t_start = time.perf_counter()
    cfg = cfg_base()
    rows, done, accum = [], set(), None
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        rows = prev.to_dict("records")
        done = {(r["O"], r["N"]) for r in rows}
        accum = 1 if (prev["mode"] == "batch64").all() else ACCUM
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
            if accum is None:  # first pair decides the mode for all pairs
                oom = False
                try:
                    res = run_pair(ck, o, n, cfg, 1)
                    accum = 1
                except torch.cuda.OutOfMemoryError:
                    oom = True
                if oom:
                    gc.collect()
                    torch.cuda.empty_cache()
                    log(f"CUDA OOM at batch 64 on {o}->{n}; all pairs use accumulation {ACCUM}x16")
                    accum = ACCUM
                    res = run_pair(ck, o, n, cfg, accum)
            else:
                res = run_pair(ck, o, n, cfg, accum)
            stats, I_f, T_f, labels_f, ids_f = res
            if not (np.array_equal(labels, labels_f) and list(ids) == list(ids_f)):
                raise RuntimeError("probe image order differs")
            row, _preds = swap_from_features(I_p, I_f, T_p, T_f, labels, f"{o}->{n}", ids,
                                             scale=100.0, threshold=0.9)
            mv, ml, av, al = row["M_F_V"], row["M_F_L"], row["F_V"], row["F_L"]
            den, den_a = abs(mv) + abs(ml), abs(av) + abs(al)
            out = {"O": o, "N": n,
                   "D_B2fp32": (mv - ml) / den if den > 0 else float("nan"),
                   "D_acc": (av - al) / den_a if den_a > 0 else float("nan"),
                   "M_F_V": mv, "M_F_L": ml, "M_total": row["M_total"],
                   "M_interaction": row["M_interaction"],
                   "A_pp": row["A_pp"], "A_pf": row["A_pf"], "A_fp": row["A_fp"], "A_ff": row["A_ff"],
                   "probe_F_V": av, "probe_F_L": al, "probe_total": row["total"],
                   "updates": stats["updates"],
                   "mode": "batch64" if accum == 1 else f"accum{ACCUM}x16",
                   "train_seconds": stats["train_seconds"], "probe_train_loss": stats["train_loss"],
                   "pair_seconds": time.perf_counter() - t0,
                   "peak_gpu_gb": stats["peak_gpu_memory_gb"]}
            if out["updates"] != PROBE_STEPS:
                out["D_B2fp32"] = float("nan")
            rows.append(out)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            log(f"  {o}->{n}: D={out['D_B2fp32']:+.3f} updates={out['updates']} mode={out['mode']} "
                f"{out['pair_seconds']:.0f}s")
    secs = time.perf_counter() - t_start
    rt = json.loads(RUNTIME_JSON.read_text()) if RUNTIME_JSON.exists() else {"sessions": []}
    rt["sessions"].append({"seconds": secs, "gpu": torch.cuda.get_device_name(0),
                           "torch": torch.__version__, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    RUNTIME_JSON.write_text(json.dumps(rt, indent=2) + "\n")
    log(f"done in {secs / 60:.1f} min")


if __name__ == "__main__":
    main()

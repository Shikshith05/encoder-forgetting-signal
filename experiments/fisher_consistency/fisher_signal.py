"""Signal A: Fisher x Adam-consistency (prereg_signals_v2.md, section 3). No training.

For each old domain O (theta_O from grid12):
  F_i  = mean over first 16 mini-batches of 8 (O TRAIN) of (batch-mean grad_i)^2
For each new domain N != O, at theta_O:
  c_i  = |mean over first 8 batches of 64 (N TRAIN) of sign(batch grad_i)|
  S_E  = sum_{i in E} F_i * c_i^2,  E in {V = visual.*, L = rest except logit_scale}
  PRIMARY D_A = (S_V - S_L) / (S_V + S_L)
Float32, no autocast. Results saved after every pair (resumable).
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
OUT_CSV = HERE / "fisher_signals.csv"
RUNTIME_JSON = HERE / "runtime.json"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.common import load_config, load_clip, get_dataset_loader, free_model  # noqa: E402
from src.finetune import prompt_features, tokenize_classes, seed_everything  # noqa: E402

DATASETS = ["OxfordIIITPet", "Flowers102", "EuroSAT", "DTD"]
FISHER_BATCH, FISHER_NB = 8, 16
CONS_BATCH, CONS_NB = 64, 8
DEVICE = "cuda"
SCALE = 100.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cfg_for(batch: int) -> dict:
    cfg = copy.deepcopy(load_config(str(REPO / "configs" / "config.yaml")))
    cfg["data"]["root"] = str(REPO / "data")
    cfg["finetune"]["batch_size"] = batch
    assert float(cfg["model"]["logit_scale"]) == SCALE
    return cfg


def load_model(o: str, cfg: dict):
    model, preprocess, tokenizer = load_clip(str(CKPT / f"theta_{o}.pt"), device=DEVICE, cfg=cfg)
    model.eval()
    for name, p in model.named_parameters():
        p.requires_grad_(name not in {"logit_scale", "logit_bias"})
    if hasattr(model.visual, "set_grad_checkpointing"):
        model.visual.set_grad_checkpointing(True)
    assert abs(model.logit_scale.exp().item() - SCALE) < 1e-3
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    is_v = torch.cat([torch.full((p.numel(),), n.startswith("visual."), dtype=torch.bool)
                      for n, p in model.named_parameters() if p.requires_grad])
    return model, preprocess, tokenizer, names, is_v


def per_batch_grads(model, preprocess, tokenizer, names, dataset, batch, n_batches):
    """Yield the flattened CPU float32 gradient of each of the first n_batches batches."""
    cfg = cfg_for(batch)
    seed_everything(0)
    loader, _ids, classes = get_dataset_loader(dataset, preprocess, cfg, "train")
    tokens = tokenize_classes(tokenizer, classes, cfg["prompts"], DEVICE)
    params = dict(model.named_parameters())
    for b, (images, labels) in enumerate(loader):
        if b == n_batches:
            return
        model.zero_grad(set_to_none=True)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        img = F.normalize(model.encode_image(images).float(), dim=-1)
        txt = prompt_features(model, tokens, len(classes), len(cfg["prompts"]), checkpoint_text=True)
        loss = F.cross_entropy(SCALE * img @ txt.T, labels)
        loss.backward()
        assert params["logit_scale"].grad is None
        g = torch.cat([params[n].grad.detach().reshape(-1).float() for n in names]).cpu()
        model.zero_grad(set_to_none=True)
        yield g, labels.numel()
    raise RuntimeError(f"{dataset}: fewer than {n_batches} batches")


def fisher(model, pre, tok, names, o):
    t0 = time.perf_counter()
    acc, n_img = None, 0
    for g, n in per_batch_grads(model, pre, tok, names, o, FISHER_BATCH, FISHER_NB):
        g2 = g.square()
        acc = g2 if acc is None else acc.add_(g2)
        n_img += n
    acc.div_(FISHER_NB)
    return acc, n_img, time.perf_counter() - t0


def consistency(model, pre, tok, names, n):
    t0 = time.perf_counter()
    acc, n_img = None, 0
    for g, k in per_batch_grads(model, pre, tok, names, n, CONS_BATCH, CONS_NB):
        s = torch.sign(g).to(torch.int8)
        acc = s if acc is None else acc.add_(s)
        n_img += k
    c = acc.abs().float().div_(CONS_NB)
    return c, n_img, time.perf_counter() - t0


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    t_start = time.perf_counter()
    rows, done = [], set()
    if OUT_CSV.exists():
        rows = pd.read_csv(OUT_CSV).to_dict("records")
        done = {(r["O"], r["N"]) for r in rows}
    for o in DATASETS:
        todo = [n for n in DATASETS if n != o and (o, n) not in done]
        if not todo:
            log(f"O={o}: done, skip")
            continue
        torch.cuda.reset_peak_memory_stats()
        model, pre, tok, names, is_v = load_model(o, cfg_for(CONS_BATCH))
        n_v, n_l = int(is_v.sum()), int((~is_v).sum())
        Fo, nF, tF = fisher(model, pre, tok, names, o)
        FV, FL = Fo[is_v].double(), Fo[~is_v].double()
        log(f"O={o}: Fisher on {nF} train imgs in {tF:.1f}s; sumF_V={FV.sum():.4g} sumF_L={FL.sum():.4g}")
        for n in todo:
            c, nC, tC = consistency(model, pre, tok, names, n)
            cV, cL = c[is_v].double(), c[~is_v].double()
            S_V = float((FV * cV.square()).sum())
            S_L = float((FL * cL.square()).sum())
            cons_V, cons_L = float(cV.square().sum()), float(cL.square().sum())
            row = {"O": o, "N": n, "S_V": S_V, "S_L": S_L,
                   "D_A": (S_V - S_L) / (S_V + S_L),
                   "D_A2_paramnorm": (S_V / n_v - S_L / n_l) / (S_V / n_v + S_L / n_l),
                   "sumF_V": float(FV.sum()), "sumF_L": float(FL.sum()),
                   "D_A3_fisher_only": float((FV.sum() - FL.sum()) / (FV.sum() + FL.sum())),
                   "cons_V": cons_V, "cons_L": cons_L,
                   "D_A4_cons_only": (cons_V - cons_L) / (cons_V + cons_L),
                   "mean_c_V": float(cV.mean()), "mean_c_L": float(cL.mean()),
                   "n_V": n_v, "n_L": n_l, "fisher_imgs": nF, "cons_imgs": nC,
                   "sec_fisher": tF, "sec_cons": tC,
                   "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            log(f"  {o}->{n}: D_A={row['D_A']:+.3f} (S_V={S_V:.4g}, S_L={S_L:.4g}) cons {tC:.1f}s")
        free_model(model)
        del Fo, FV, FL
        torch.cuda.empty_cache()
    secs = time.perf_counter() - t_start
    rt = json.loads(RUNTIME_JSON.read_text()) if RUNTIME_JSON.exists() else {"sessions": []}
    rt["sessions"].append({"seconds": secs, "gpu": torch.cuda.get_device_name(0),
                           "torch": torch.__version__, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    RUNTIME_JSON.write_text(json.dumps(rt, indent=2) + "\n")
    log(f"done in {secs / 60:.1f} min")


if __name__ == "__main__":
    main()

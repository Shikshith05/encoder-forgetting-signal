"""Per-encoder GRADIENT INTERFERENCE signal (PW26_VK_06, exploratory).

For each old domain O, load the grid12 source checkpoint theta_O and compute
first-order interference between the training-loss gradient on O and on each
new domain N, separately for the vision tower (visual.*) and the text tower
(all other params except logit_scale / logit_bias). No fine-tuning happens.

Loss = training loss of src.finetune.train_model: image features vs ALL class
prototypes of the dataset (10 config templates, prompt_features with
checkpoint_text=True), logits = 100 * img @ txt.T, cross-entropy. Float32, no
autocast. logit_scale is frozen (no gradient).

Gradient set of dataset D = mean of the per-batch mean gradients over the FIRST
K batches of get_dataset_loader(D, preprocess, cfg, "train") (seed 0, no
shuffling). TRAIN split only; O's TEST split is never touched.

Signals per encoder E in {V, L}:
    I_sgd_E  = -dot(g_O_E, g_N_E)
    I_adam_E = -dot(g_O_E, sign(g_N_E))      (primary: AdamW first step)
    cos_E, normN_E, normO_E

Usage:
    python grad_signal.py            # all 12 pairs (resumable)
    python grad_signal.py --sanity   # EuroSAT-on-EuroSAT gradient set twice
"""
from __future__ import annotations

import argparse
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

GRID = Path(r"D:\CAPSTONE\grid12")
CKPT = GRID / "ckpt"
OUT_CSV = HERE / "grad_signals.csv"
SANITY_JSON = HERE / "sanity.json"
RUNTIME_JSON = HERE / "runtime.json"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.common import load_config, load_clip, get_dataset_loader, free_model  # noqa: E402
from src.finetune import prompt_features, tokenize_classes, seed_everything  # noqa: E402

DATASETS = ["OxfordIIITPet", "Flowers102", "EuroSAT", "DTD"]
K = 4
DEVICE = "cuda"
SCALE = 100.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def base_cfg(batch: int) -> dict:
    cfg = copy.deepcopy(load_config(str(REPO / "configs" / "config.yaml")))
    cfg["data"]["root"] = str(REPO / "data")  # in-memory only
    cfg["finetune"]["batch_size"] = batch
    assert float(cfg["model"]["logit_scale"]) == SCALE
    return cfg


def load_model(o: str, cfg: dict):
    model, preprocess, tokenizer = load_clip(str(CKPT / f"theta_{o}.pt"), device=DEVICE, cfg=cfg)
    model.eval()  # CLIP ViT-B/32 has no dropout/BN; eval vs train is identical
    for name, p in model.named_parameters():
        p.requires_grad_(name not in {"logit_scale", "logit_bias"})
    if hasattr(model.visual, "set_grad_checkpointing"):
        model.visual.set_grad_checkpointing(True)
    assert abs(model.logit_scale.exp().item() - SCALE) < 1e-3
    return model, preprocess, tokenizer


def split_names(model) -> tuple[list[str], list[str]]:
    vis, txt = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (vis if name.startswith("visual.") else txt).append(name)
    return vis, txt


def gradient_set(model, preprocess, tokenizer, dataset: str, cfg: dict) -> dict:
    """Mean gradient over the first K train batches. Returns CPU float32 vectors."""
    seed_everything(0)
    loader, _ids, classes = get_dataset_loader(dataset, preprocess, cfg, "train")
    templates = cfg["prompts"]
    tokens = tokenize_classes(tokenizer, classes, templates, DEVICE)
    params = dict(model.named_parameters())
    vis, txt = split_names(model)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_batches, n_images, losses = 0, 0, []
    for images, labels in loader:
        if n_batches == K:
            break
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        img = F.normalize(model.encode_image(images).float(), dim=-1)
        txt_feat = prompt_features(model, tokens, len(classes), len(templates), checkpoint_text=True)
        logits = SCALE * img @ txt_feat.T
        loss = F.cross_entropy(logits, labels) / K  # mean of per-batch means
        loss.backward()
        losses.append(loss.item() * K)
        n_batches += 1
        n_images += labels.numel()
    if n_batches != K:
        raise RuntimeError(f"{dataset}: only {n_batches} batches available")
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    assert params["logit_scale"].grad is None, "logit_scale received gradient"

    def flat(names):
        return torch.cat([params[n].grad.detach().reshape(-1).float().cpu() for n in names])

    out = {"V": flat(vis), "L": flat(txt), "seconds": secs, "n_images": n_images,
           "loss": float(np.mean(losses)),
           "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    model.zero_grad(set_to_none=True)
    return out


def interference(gO: dict, gN: dict) -> dict:
    row = {}
    for e in ("V", "L"):
        a = gO[e].double()
        b = gN[e].double()
        na, nb = a.norm().item(), b.norm().item()
        row[f"I_sgd_{e}"] = -torch.dot(a, b).item()
        row[f"I_adam_{e}"] = -torch.dot(a, torch.sign(b)).item()
        row[f"cos_{e}"] = torch.dot(a, b).item() / (na * nb)
        row[f"normN_{e}"] = nb
        row[f"normO_{e}"] = na
    return row


def pick_batch() -> int:
    """Use batch 64 if a float32 gradient set fits, else 32 (probe on the largest
    prompt set, Flowers102 = 1020 prompts)."""
    for batch in (64, 32):
        cfg = base_cfg(batch)
        model = None
        try:
            torch.cuda.reset_peak_memory_stats()
            model, pre, tok = load_model("EuroSAT", cfg)
            g = gradient_set(model, pre, tok, "Flowers102", cfg)
            log(f"probe batch {batch}: OK, peak {g['peak_gb']:.2f} GB, {g['seconds']:.1f}s")
            free_model(model)
            return batch
        except torch.cuda.OutOfMemoryError:
            log(f"probe batch {batch}: CUDA OOM")
            if model is not None:
                model.zero_grad(set_to_none=True)
                free_model(model)
            torch.cuda.empty_cache()
    raise SystemExit("Batch 32 does not fit either; stopping.")


def run_pairs(batch: int) -> None:
    cfg = base_cfg(batch)
    rows, done = [], set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        if not (prev["batch"] == batch).all():
            raise SystemExit("Existing grad_signals.csv uses a different batch size.")
        rows = prev.to_dict("records")
        done = {(r["O"], r["N"]) for r in rows}
    for o in DATASETS:
        todo = [n for n in DATASETS if n != o and (o, n) not in done]
        if not todo:
            log(f"O={o}: all pairs present, skipping")
            continue
        torch.cuda.reset_peak_memory_stats()
        model, pre, tok = load_model(o, cfg)
        gO = gradient_set(model, pre, tok, o, cfg)
        log(f"O={o}: g_O on {o} train: {gO['seconds']:.1f}s, loss={gO['loss']:.4f}, "
            f"|V|={gO['V'].numel()}, |L|={gO['L'].numel()}")
        for n in todo:
            gN = gradient_set(model, pre, tok, n, cfg)
            log(f"O={o}: g_N on {n} train: {gN['seconds']:.1f}s, loss={gN['loss']:.4f}")
            row = {"O": o, "N": n, **interference(gO, gN), "batch": batch, "K": K,
                   "n_images_O": gO["n_images"], "n_images_N": gN["n_images"],
                   "loss_O": gO["loss"], "loss_N": gN["loss"],
                   "sec_gO": gO["seconds"], "sec_gN": gN["seconds"],
                   "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
                   "dim_V": gO["V"].numel(), "dim_L": gO["L"].numel()}
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            log(f"  {o}->{n}: I_adam_V={row['I_adam_V']:.4g} I_adam_L={row['I_adam_L']:.4g} "
                f"cos_V={row['cos_V']:.3f} cos_L={row['cos_L']:.3f}")
        free_model(model)
        torch.cuda.empty_cache()


def run_sanity(batch: int) -> None:
    cfg = base_cfg(batch)
    model, pre, tok = load_model("EuroSAT", cfg)
    g1 = gradient_set(model, pre, tok, "EuroSAT", cfg)
    g2 = gradient_set(model, pre, tok, "EuroSAT", cfg)
    free_model(model)
    res = {"gradient_set": "theta_EuroSAT on EuroSAT train", "batch": batch, "K": K}
    for e in ("V", "L"):
        d = (g1[e] - g2[e]).abs()
        res[f"max_abs_diff_{e}"] = d.max().item()
        res[f"max_abs_grad_{e}"] = g1[e].abs().max().item()
        res[f"sign_agreement_{e}"] = (torch.sign(g1[e]) == torch.sign(g2[e])).float().mean().item()
    res["max_abs_diff"] = max(res["max_abs_diff_V"], res["max_abs_diff_L"])
    res["seconds"] = [g1["seconds"], g2["seconds"]]
    SANITY_JSON.write_text(json.dumps(res, indent=2) + "\n")
    log(f"sanity: {res}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--batch", type=int, default=None, help="skip the probe")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; stopping.")
    t0 = time.perf_counter()
    batch = args.batch or pick_batch()
    log(f"GPU {torch.cuda.get_device_name(0)}; batch={batch}; K={K}; float32")
    if args.sanity:
        run_sanity(batch)
    else:
        run_pairs(batch)
    secs = time.perf_counter() - t0
    rt = json.loads(RUNTIME_JSON.read_text()) if RUNTIME_JSON.exists() else {"sessions": []}
    rt["sessions"].append({"mode": "sanity" if args.sanity else "pairs", "seconds": secs,
                           "batch": batch, "gpu": torch.cuda.get_device_name(0),
                           "torch": torch.__version__,
                           "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    RUNTIME_JSON.write_text(json.dumps(rt, indent=2) + "\n")
    log(f"done in {secs / 60:.1f} min")


if __name__ == "__main__":
    main()

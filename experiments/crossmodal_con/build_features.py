"""Step 1: feature bank (encode once, save everything). No training, no signals.

For each model in {pre, theta_<A> x4}, dataset in the 4 benchmarks and split in {train, test}
writes D:\\CAPSTONE\\grid12\\cache\\features\\<model>_<dataset>_<split>.npz with
  I            [N, 512]     unit-normalised image embeddings (repo extract_image_features)
  labels       [N]          int64
  ids          [N]          repo image ids (seed-0 permutation order, <= 2000)
  T_templates  [C, 10, 512] unit-normalised per-template text embeddings (config prompts)
  T            [C, 512]     class prototypes (repo extract_text_features)
Existing files are skipped (resumable). A-test embeddings are stored for later analysis only;
no signal in this experiment reads them.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(r"D:\CAPSTONE\encoder-forgetting-signal")
CKPT = Path(r"D:\CAPSTONE\grid12\ckpt")
FEAT = Path(r"D:\CAPSTONE\grid12\cache\features")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.common import (  # noqa: E402
    load_config, load_clip, get_dataset_loader,
    extract_image_features, extract_text_features, free_model,
)

DATASETS = ["OxfordIIITPet", "Flowers102", "EuroSAT", "DTD"]
MODELS = ["pre"] + [f"theta_{d}" for d in DATASETS]
SPLITS = ["train", "test"]
DEVICE = "cuda"


def per_template_text(model, tokenizer, classnames, templates):
    out = []
    with torch.no_grad():
        for c in classnames:
            tok = tokenizer([t.format(c) for t in templates]).to(DEVICE)
            e = model.encode_text(tok).float()
            e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.cpu().numpy().astype(np.float32))
    return np.stack(out, 0)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = copy.deepcopy(load_config(str(REPO / "configs" / "config.yaml")))
    cfg["data"]["root"] = str(REPO / "data")
    FEAT.mkdir(parents=True, exist_ok=True)
    man_path = HERE / "features_manifest.csv"
    rows = pd.read_csv(man_path).to_dict("records") if man_path.exists() else []
    done = {(r["model"], r["dataset"], r["split"]) for r in rows}

    for m in MODELS:
        todo = [(d, s) for d in DATASETS for s in SPLITS
                if not (FEAT / f"{m}_{d}_{s}.npz").exists()]
        if not todo:
            continue
        ckpt = None if m == "pre" else str(CKPT / f"{m}.pt")
        model, preprocess, tokenizer = load_clip(ckpt, device=DEVICE, cfg=cfg)
        text_cache = {}
        for d, s in todo:
            t0 = time.perf_counter()
            loader, ids, classes = get_dataset_loader(d, preprocess, cfg, s)
            if d not in text_cache:
                T = extract_text_features(model, tokenizer, classes, cfg["prompts"], DEVICE)
                TT = per_template_text(model, tokenizer, classes, cfg["prompts"])
                proto = TT.mean(1)
                proto /= np.linalg.norm(proto, axis=1, keepdims=True)
                assert np.abs(proto - T).max() < 1e-4, "per-template mean != repo prototype"
                text_cache[d] = (T, TT)
            T, TT = text_cache[d]
            I, labels = extract_image_features(model, loader, DEVICE)
            out = FEAT / f"{m}_{d}_{s}.npz"
            np.savez(out, I=I.astype(np.float32), labels=labels.astype(np.int64),
                     ids=np.array(ids), T_templates=TT.astype(np.float32),
                     T=T.astype(np.float32))
            sec = time.perf_counter() - t0
            print(f"{m} {d} {s}: N={len(I)} C={len(T)} {sec:.1f}s", flush=True)
            if (m, d, s) not in done:
                rows.append({"model": m, "dataset": d, "split": s, "n_images": len(I),
                             "n_classes": len(T), "file": str(out), "seconds": round(sec, 2)})
                pd.DataFrame(rows).to_csv(man_path, index=False)
        free_model(model)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

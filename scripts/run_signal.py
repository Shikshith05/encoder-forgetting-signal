"""Part A (Kishan): CLI Runner for the Pre-fine-tuning Signal.

Computes Gap_V, Gap_L, Zero-Shot Acc, IIMM scores, and Pearson correlation across 12 dataset pairs.
Outputs:
  - results/signal/signals.csv
  - results/signal/gap_v_heatmap.png
  - results/signal/gap_l_heatmap.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

# Add repo root to path so 'src' can be imported cleanly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.common import load_config
from src.signal_gap import compute_dataset_pair_signals, save_heatmap


def run_mock_pipeline(cfg: dict, out_dir: str) -> None:
    """Run verification pipeline using realistic synthetic normalized embeddings."""
    print("[Mode] Running verification pipeline with synthetic unit embeddings...")
    datasets = ["OxfordIIITPet", "Flowers102", "EuroSAT", "DTD"]
    dim = 512
    rng = np.random.default_rng(0)

    image_features_dict: Dict[str, np.ndarray] = {}
    text_features_dict: Dict[str, np.ndarray] = {}
    labels_dict: Dict[str, np.ndarray] = {}

    class_counts = {
        "OxfordIIITPet": 37,
        "Flowers102": 102,
        "EuroSAT": 10,
        "DTD": 47,
    }

    # Create distinct cluster centers for each dataset to simulate realistic domain gaps
    dataset_centers = {
        "OxfordIIITPet": rng.normal(loc=0.0, scale=1.0, size=(1, dim)),
        "Flowers102": rng.normal(loc=0.5, scale=1.0, size=(1, dim)),
        "EuroSAT": rng.normal(loc=3.0, scale=1.0, size=(1, dim)),  # Far domain shift
        "DTD": rng.normal(loc=-1.5, scale=1.0, size=(1, dim)),
    }

    for name in datasets:
        C = class_counts[name]
        N = min(cfg["data"].get("max_images", 2000), 1000)
        center = dataset_centers[name]

        # Class text features
        T = center + rng.normal(loc=0.0, scale=0.5, size=(C, dim))
        T = T / np.linalg.norm(T, axis=1, keepdims=True)

        # Labels
        labels = rng.integers(0, C, size=N)

        # Image features clustered near their true class prototype
        I = T[labels] + rng.normal(loc=0.0, scale=0.4, size=(N, dim))
        I = I / np.linalg.norm(I, axis=1, keepdims=True)

        image_features_dict[name] = I.astype(np.float32)
        text_features_dict[name] = T.astype(np.float32)
        labels_dict[name] = labels

    _execute_and_save(datasets, image_features_dict, text_features_dict, labels_dict, out_dir)


def run_real_pipeline(cfg: dict, out_dir: str, device: str) -> None:
    """Run full pipeline with pretrained CLIP ViT-B/32 on the 4 datasets."""
    from src.common import (
        extract_image_features,
        extract_text_features,
        free_model,
        get_dataset_loader,
        load_clip,
    )

    datasets = cfg["data"].get("datasets", ["OxfordIIITPet", "Flowers102", "EuroSAT", "DTD"])
    templates = cfg["prompts"]

    print(f"[Device] Using {device}")
    print(f"[Model] Loading CLIP {cfg['model']['name']} ({cfg['model']['pretrained']})...")
    model, preprocess, tokenizer = load_clip(None, device, cfg)

    image_features_dict: Dict[str, np.ndarray] = {}
    text_features_dict: Dict[str, np.ndarray] = {}
    labels_dict: Dict[str, np.ndarray] = {}

    for name in datasets:
        print(f"\n--- Processing dataset: {name} ---")
        loader, image_ids, classnames = get_dataset_loader(name, preprocess, cfg, split="test")
        print(f"  Encoding {len(image_ids)} images...")
        I, y = extract_image_features(model, loader, device)
        print(f"  Encoding {len(classnames)} classes with 10 prompt templates...")
        T = extract_text_features(model, tokenizer, classnames, templates, device)

        image_features_dict[name] = I
        text_features_dict[name] = T
        labels_dict[name] = y
        print(f"  Features extracted: Images shape {I.shape}, Text shape {T.shape}")

    free_model(model)
    _execute_and_save(datasets, image_features_dict, text_features_dict, labels_dict, out_dir)


def _execute_and_save(
    datasets: List[str],
    image_features_dict: Dict[str, np.ndarray],
    text_features_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("\nComputing Fréchet distances (Ledoit-Wolf), Zero-Shot Accuracies, and IIMM scores...")
    signals_df, gap_v_matrix, gap_l_matrix, corr = compute_dataset_pair_signals(
        datasets, image_features_dict, text_features_dict, labels_dict
    )

    csv_path = os.path.join(out_dir, "signals.csv")
    signals_df.to_csv(csv_path, index=False)

    heatmap_v_path = os.path.join(out_dir, "gap_v_heatmap.png")
    heatmap_l_path = os.path.join(out_dir, "gap_l_heatmap.png")

    save_heatmap(gap_v_matrix, datasets, "Vision Gap (Gap_V) - Fréchet Distance", heatmap_v_path, cmap="Blues")
    save_heatmap(gap_l_matrix, datasets, "Language Gap (Gap_L) - Fréchet Distance", heatmap_l_path, cmap="Greens")

    print(f"\n====================== Part A Signal Results ======================")
    print(f"Saved signals CSV: {csv_path}")
    print(f"Saved Gap_V heatmap: {heatmap_v_path}")
    print(f"Saved Gap_L heatmap: {heatmap_l_path}")
    print(f"\nPearson correlation between Gap_V and Gap_L: {corr:.4f}")
    if corr > 0.90:
        print("[NOTICE] Correlation > 0.90! Remember to highlight this to Shikshith.")

    print("\n--- Signals Table (12 dataset pairs) ---")
    display_cols = ["source", "target", "gap_v", "gap_l", "source_zeroshot", "target_zeroshot", "source_iimm", "target_iimm"]
    print(signals_df[display_cols].round(3).to_string(index=False))

    print("\n--- Sanity Checks ---")
    pets_zs = signals_df[signals_df["source"] == "OxfordIIITPet"]["source_zeroshot"].iloc[0]
    eurosat_zs = signals_df[signals_df["source"] == "EuroSAT"]["source_zeroshot"].iloc[0]
    print(f"Pets Zero-Shot Accuracy: {pets_zs:.2f}%")
    print(f"EuroSAT Zero-Shot Accuracy: {eurosat_zs:.2f}%")
    if pets_zs > eurosat_zs:
        print("  [OK] Check passed: Pets zero-shot accuracy is significantly higher than EuroSAT.")
    else:
        print("  [!] Notice: Verify dataset accuracies on real embeddings.")


def main():
    parser = argparse.ArgumentParser(description="Part A: Pre-fine-tuning Signal (Kishan)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--out-dir", default="results/signal", help="Directory to save signals.csv and heatmaps")
    parser.add_argument("--device", default="cuda", help="Torch device ('cuda' or 'cpu')")
    parser.add_argument("--mock", action="store_true", help="Run verification with synthetic features")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Determine whether torch and datasets can be imported
    can_run_real = False
    try:
        import torch
        import torchvision
        import open_clip
        if args.device == "cuda" and not torch.cuda.is_available():
            args.device = "cpu"
        can_run_real = True
    except ImportError:
        can_run_real = False

    if args.mock or not can_run_real:
        if not can_run_real and not args.mock:
            print("[Info] torch/torchvision/open_clip not found in local environment. Running with --mock mode.")
        run_mock_pipeline(cfg, args.out_dir)
    else:
        run_real_pipeline(cfg, args.out_dir, args.device)


if __name__ == "__main__":
    main()

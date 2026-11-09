"""Run Shreyas's four Phase-1 experiments (requires datasets and CLIP weights)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_config
from src.finetune import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fallback", action="store_true", help="If forgetting <1 pp, add a matched EuroSAT pair at lr=3e-5.")
    args = parser.parse_args()
    run_experiments(load_config(args.config), args.out_dir, args.device,
                    fallback_lr=3e-5 if args.fallback else None)


if __name__ == "__main__":
    main()

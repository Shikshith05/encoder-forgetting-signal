"""Run the Part C encoder swap and Shapley split; owner: Shikshith V.

Usage:
    python scripts/run_swap.py --plain    # sanity check on pretrained CLIP
    python scripts/run_swap.py            # all checkpoint pairs that exist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.swap_shapley import DEFAULT_PAIRS, PLAIN_PAIRS, _load_config, run_pairs  # noqa: E402


def default_device() -> str:
    """Return "cuda" when a GPU is usable, otherwise "cpu"."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description="2x2 encoder swap and Shapley split (Part C).")
    parser.add_argument("--config", default="configs/config.yaml", help="path to the project config")
    parser.add_argument("--ckpt-dir", default="checkpoints/", help="directory holding the checkpoints")
    parser.add_argument("--out-dir", default="results/swap/", help="directory for the output CSVs")
    parser.add_argument("--plain", action="store_true", help="run only the plain-CLIP check")
    parser.add_argument("--device", default=None, help='device to use (default: "cuda" if available else "cpu")')
    return parser.parse_args()


def main() -> None:
    """Load the config and run either the plain-CLIP check or the real pairs."""
    args = parse_args()
    device = args.device or default_device()
    pairs = PLAIN_PAIRS if args.plain else DEFAULT_PAIRS
    run_pairs(pairs, args.ckpt_dir, args.out_dir, _load_config(args.config), device)


if __name__ == "__main__":
    main()

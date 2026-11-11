"""Part C (Shikshith): 2x2 encoder swap and Shapley split.

CLIP's image and text encoders meet only at a dot product, so features from two
checkpoints can be mixed without copying any weights. With p = "before" and
f = "after", four hybrids are evaluated on the old domain (Oxford-IIIT Pets);
letter 1 names the vision checkpoint and letter 2 the text checkpoint:

    pp = (I_p, T_p)    pf = (I_p, T_f)    fp = (I_f, T_p)    ff = (I_f, T_f)

The Shapley value then splits total forgetting A_pp - A_ff between the encoders.

The maths uses numpy only. torch, torchvision and open_clip are imported inside
the functions that need them, so this module and its tests run without them.
"""

from __future__ import annotations

import gc
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

HYBRIDS: Tuple[str, ...] = ("pp", "pf", "fp", "ff")

PLAIN_NAME = "plain-CLIP check"
PLAIN_PAIRS: List[Tuple[Optional[str], Optional[str], str]] = [(None, None, PLAIN_NAME)]

DEFAULT_PAIRS: List[Tuple[str, str, str]] = [
    ("theta_pets.pt", "theta_pets_eurosat.pt", "Pets->EuroSAT"),
    ("theta_pets.pt", "theta_pets_flowers.pt", "Pets->Flowers"),
    ("theta_pets.pt", "theta_pets_eurosat_textfrozen.pt", "Pets->EuroSAT (text frozen)"),
    ("theta_pets.pt", "sensitivity_lr3e-5/theta_pets_eurosat.pt", "Pets->EuroSAT (lr 3e-5)"),
    ("theta_pets.pt", "sensitivity_lr3e-5/theta_pets_eurosat_textfrozen.pt", "Pets->EuroSAT text frozen (lr 3e-5)"),
]


# ---------------------------------------------------------------------------
# Numpy maths (testable without torch or CLIP)
# ---------------------------------------------------------------------------


def shapley_split(A_pp: float, A_pf: float, A_fp: float, A_ff: float) -> Dict[str, float]:
    """Split the drop A_pp - A_ff between the vision and text encoders.

    F_V and F_L are the Shapley values of swapping each encoder from p to f:
    each encoder's change is averaged over the two possible orders of swapping.
    They always sum to the total drop. The interaction measures how much the
    vision encoder's effect depends on which text encoder it is paired with.
    D is the normalised vision-minus-text dominance in [-1, 1]; it is NaN when
    the total drop is at most 1 point, because a ratio of noise is meaningless.

    Returns:
        Dict with keys F_V, F_L, total, interaction and D.
    """
    F_V = 0.5 * ((A_pf - A_ff) + (A_pp - A_fp))
    F_L = 0.5 * ((A_fp - A_ff) + (A_pp - A_pf))
    total = A_pp - A_ff
    interaction = (A_pf - A_ff) - (A_pp - A_fp)
    D = (F_V - F_L) / (abs(F_V) + abs(F_L) + 1e-8) if total > 1.0 else float("nan")
    return {"F_V": F_V, "F_L": F_L, "total": total, "interaction": interaction, "D": D}


def evaluate(
    I: np.ndarray, T: np.ndarray, labels: np.ndarray, scale: float = 100.0
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Classify images against class text features and score the result.

    The softmax subtracts each row's largest logit before exponentiating, so it
    cannot overflow at CLIP's large logit scale.

    Args:
        I: [N, d] L2-normalised image features.
        T: [C, d] L2-normalised class text features.
        labels: [N] true class indices.
        scale: Fixed multiplier on the cosine similarities.

    Returns:
        (accuracy in percent, [N] predictions, [N] max probabilities,
        [N, C] class probabilities).
    """
    I = np.asarray(I, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    labels = np.asarray(labels)

    logits = scale * I @ T.T
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probs = exponentiated / exponentiated.sum(axis=1, keepdims=True)

    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    acc_percent = float((pred == labels).mean() * 100.0)
    return acc_percent, pred, conf, probs


def linear_margin(I: np.ndarray, T: np.ndarray, labels: np.ndarray) -> float:
    """Return the mean over images of <u, t_y - mean_k t_k>.

    This is the similarity to the true class text minus the similarity to the
    mean class text. Unlike accuracy it is linear in each encoder's features
    and has no argmax, so its Shapley split is exact per example and is not
    affected by predictions flipping across a decision boundary.
    """
    I = np.asarray(I, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    labels = np.asarray(labels)
    centred = T[labels] - T.mean(axis=0, keepdims=True)
    return float(np.sum(I * centred, axis=1).mean())


def swap_from_features(
    I_p: np.ndarray,
    I_f: np.ndarray,
    T_p: np.ndarray,
    T_f: np.ndarray,
    labels: np.ndarray,
    name: str,
    image_ids: Sequence[Any],
    scale: float,
    threshold: float,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate the four hybrids and split forgetting between the encoders.

    Both accuracy and linear margin are split with the Shapley value, and the
    confident-wrong rate (confidence above threshold and wrong, in percent) is
    recorded for each hybrid.

    Returns:
        row: One results row with keys pair, A_*, total, F_V, F_L, interaction,
            D, M_*, M_total, M_F_V, M_F_L, M_interaction and CW_*.
        preds_df: Per-image predictions for the ECE code, with 4*N rows and
            columns pair, hybrid, image_id, true, pred, conf.
    """
    labels = np.asarray(labels)
    if len(image_ids) != len(labels):
        raise ValueError(f"{len(image_ids)} image_ids for {len(labels)} labels")

    features = {"pp": (I_p, T_p), "pf": (I_p, T_f), "fp": (I_f, T_p), "ff": (I_f, T_f)}
    accuracy: Dict[str, float] = {}
    margin: Dict[str, float] = {}
    confident_wrong: Dict[str, float] = {}
    frames: List[pd.DataFrame] = []

    for hybrid in HYBRIDS:
        I, T = features[hybrid]
        accuracy[hybrid], pred, conf, _ = evaluate(I, T, labels, scale)
        margin[hybrid] = linear_margin(I, T, labels)
        confident_wrong[hybrid] = float(((conf > threshold) & (pred != labels)).mean() * 100.0)
        frames.append(
            pd.DataFrame(
                {
                    "pair": name,
                    "hybrid": hybrid,
                    "image_id": list(image_ids),
                    "true": labels,
                    "pred": pred,
                    "conf": conf,
                }
            )
        )

    acc_split = shapley_split(accuracy["pp"], accuracy["pf"], accuracy["fp"], accuracy["ff"])
    margin_split = shapley_split(margin["pp"], margin["pf"], margin["fp"], margin["ff"])
    for split in (acc_split, margin_split):
        assert abs(split["F_V"] + split["F_L"] - split["total"]) < 1e-6, "Shapley split is not efficient"

    row: Dict[str, Any] = {"pair": name}
    row.update({f"A_{hybrid}": accuracy[hybrid] for hybrid in HYBRIDS})
    row.update({key: acc_split[key] for key in ("total", "F_V", "F_L", "interaction", "D")})
    row.update({f"M_{hybrid}": margin[hybrid] for hybrid in HYBRIDS})
    row.update({f"M_{key}": margin_split[key] for key in ("total", "F_V", "F_L", "interaction")})
    row.update({f"CW_{hybrid}": confident_wrong[hybrid] for hybrid in HYBRIDS})
    return row, pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Torch / CLIP (imports stay inside the functions)
# ---------------------------------------------------------------------------


def _load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Read the project's YAML settings into a dict."""
    # can move to src/common.py later
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_state_dict(ckpt: str) -> Dict[str, Any]:
    """Load a checkpoint file on CPU and return a clean state dict.

    Unwraps a "state_dict" or "model" entry when present and strips a leading
    "module." from keys, which DataParallel-style saves add.
    """
    # can move to src/common.py later
    import torch

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    return {key.removeprefix("module."): value for key, value in state.items()}


def _load_clip(
    ckpt: Optional[str] = None, device: str = "cpu", cfg: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Any, Any]:
    """Build CLIP in eval mode, optionally loading a checkpoint strictly.

    Args:
        ckpt: Checkpoint path, or None for the pretrained weights.
        device: Device to place the model on.
        cfg: Project config; read from the default path when None.

    Returns:
        (model, eval preprocess transform, tokenizer).
    """
    # Activation functions are not stored in checkpoints. Reuse Part B's
    # corrected factory settings so both towers retain OpenAI QuickGELU.
    from src.common import load_clip

    return load_clip(ckpt, device, cfg if cfg is not None else _load_config())


def _pets_loader(preprocess: Any, cfg: Dict[str, Any]) -> Tuple[Any, List[str], List[str]]:
    """Build the deterministic Oxford-IIIT Pets test loader for the old domain.

    The subset follows the team's data rule: a seeded randperm over the test
    set, truncated to data.max_images, so every part sees the same images.

    Returns:
        (DataLoader, image ids in loader order, class names).
    """
    # can move to src/common.py later
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import OxfordIIITPet

    data = cfg["data"]
    ds = OxfordIIITPet(root=data["root"], split="test", download=True, transform=preprocess)
    generator = torch.Generator().manual_seed(data["seed"])
    idx = torch.randperm(len(ds), generator=generator)[: data["max_images"]].tolist()

    loader = DataLoader(Subset(ds, idx), batch_size=128, shuffle=False, num_workers=2)
    image_paths = getattr(ds, "_images", None)
    if image_paths is not None:
        image_ids = [os.path.splitext(os.path.basename(str(image_paths[i])))[0] for i in idx]
    else:
        image_ids = [str(i) for i in idx]
    classnames = [name.replace("_", " ") for name in ds.classes]
    return loader, image_ids, classnames


def image_features(model: Any, loader: Any, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Encode every image and return float32 L2-normalised features and labels."""
    import torch

    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for images, targets in loader:
            encoded = model.encode_image(images.to(device)).float()
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            features.append(encoded.cpu().numpy().astype(np.float32))
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def text_features(
    model: Any, tokenizer: Any, classnames: Sequence[str], templates: Sequence[str], device: str
) -> np.ndarray:
    """Build one L2-normalised text feature per class.

    For each class, all prompt templates are encoded and L2-normalised, then
    averaged and re-normalised, which is the standard CLIP prompt ensemble.

    Returns:
        [C, d] float32 array.
    """
    import torch

    class_features: List[np.ndarray] = []
    with torch.no_grad():
        for classname in classnames:
            prompts = [template.format(classname) for template in templates]
            encoded = model.encode_text(tokenizer(prompts).to(device)).float()
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            mean = encoded.mean(dim=0)
            mean = mean / mean.norm()
            class_features.append(mean.cpu().numpy().astype(np.float32))
    return np.stack(class_features)


def _free_model(model: Any) -> None:
    """Move a finished model off the GPU so the next one has room."""
    # can move to src/common.py later
    import torch

    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def text_encoder_identical(
    ckpt_p: Optional[str], ckpt_f: Optional[str], cfg: Optional[Dict[str, Any]] = None
) -> bool:
    """Check whether two checkpoints share exactly the same text encoder.

    Compares every entry whose key does not start with "visual.", excluding
    "logit_scale". None means the pretrained CLIP weights. This verifies that a
    "text frozen" run really left the text side untouched.

    Args:
        ckpt_p: First checkpoint path, or None.
        ckpt_f: Second checkpoint path, or None.
        cfg: Project config used to build pretrained CLIP; default path if None.
    """
    import torch

    if ckpt_p == ckpt_f:
        return True

    def text_state(ckpt: Optional[str]) -> Dict[str, Any]:
        """Return the non-visual entries of one checkpoint's state dict."""
        if ckpt is None:
            state = _load_clip(None, "cpu", cfg)[0].state_dict()
        else:
            state = _read_state_dict(ckpt)
        return {k: v for k, v in state.items() if not k.startswith("visual.") and k != "logit_scale"}

    state_p = text_state(ckpt_p)
    state_f = text_state(ckpt_f)
    if state_p.keys() != state_f.keys():
        return False
    return all(torch.equal(state_p[key], state_f[key]) for key in state_p)


def swap(
    ckpt_p: Optional[str], ckpt_f: Optional[str], name: str, cfg: Dict[str, Any], device: str
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run the 2x2 swap for one checkpoint pair on the Pets test subset.

    Each model encodes the images once and the text once, so the four hybrids
    cost two image passes and two text passes. No weights are ever copied.

    Args:
        ckpt_p: "Before" checkpoint path, or None for pretrained CLIP.
        ckpt_f: "After" checkpoint path, or None for pretrained CLIP.
        name: Label for this pair in the outputs.
        cfg: Project config.
        device: Device to run on.

    Returns:
        (row with an added "text_frozen" flag, per-image predictions).
    """
    templates = cfg["prompts"]

    model, preprocess, tokenizer = _load_clip(ckpt_p, device, cfg)
    loader, image_ids, classnames = _pets_loader(preprocess, cfg)
    I_p, labels = image_features(model, loader, device)
    T_p = text_features(model, tokenizer, classnames, templates, device)
    _free_model(model)
    del model

    model, _, tokenizer = _load_clip(ckpt_f, device, cfg)
    I_f, labels_f = image_features(model, loader, device)
    T_f = text_features(model, tokenizer, classnames, templates, device)
    _free_model(model)
    del model

    if not np.array_equal(labels, labels_f):
        raise RuntimeError("Image order differs between the two passes; hybrids would be misaligned")

    row, preds_df = swap_from_features(
        I_p,
        I_f,
        T_p,
        T_f,
        labels,
        name,
        image_ids,
        scale=float(cfg["model"]["logit_scale"]),
        threshold=float(cfg["confidence"]["confident_wrong_threshold"]),
    )
    row["text_frozen"] = text_encoder_identical(ckpt_p, ckpt_f, cfg)
    return row, preds_df


def run_pairs(
    pairs: Sequence[Tuple[Optional[str], Optional[str], str]],
    ckpt_dir: str,
    out_dir: str,
    cfg: Dict[str, Any],
    device: str,
) -> pd.DataFrame:
    """Run the swap for each pair, write the CSVs and print a readable table.

    A pair with a missing checkpoint is skipped with a message. If every pair
    is skipped, nothing is written, so earlier results are not overwritten.

    Args:
        pairs: (p filename or None, f filename or None, name) tuples.
        ckpt_dir: Directory containing the checkpoint files.
        out_dir: Directory for swap_results.csv and predictions.csv.
        cfg: Project config.
        device: Device to run on.

    Returns:
        One row per pair that ran (empty if none did).
    """
    rows: List[Dict[str, Any]] = []
    predictions: List[pd.DataFrame] = []

    for file_p, file_f, name in pairs:
        ckpt_p = os.path.join(ckpt_dir, file_p) if file_p is not None else None
        ckpt_f = os.path.join(ckpt_dir, file_f) if file_f is not None else None
        missing = [path for path in (ckpt_p, ckpt_f) if path is not None and not os.path.isfile(path)]
        if missing:
            print(f"[skip] {name}: missing checkpoint(s): {', '.join(missing)}")
            continue
        print(f"[run]  {name}")
        row, preds_df = swap(ckpt_p, ckpt_f, name, cfg, device)
        rows.append(row)
        predictions.append(preds_df)

    results = pd.DataFrame(rows)
    if results.empty:
        print("No pair could be run; no results were written.")
        return results

    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "swap_results.csv")
    predictions_path = os.path.join(out_dir, "predictions.csv")
    results.to_csv(results_path, index=False)
    pd.concat(predictions, ignore_index=True).to_csv(predictions_path, index=False)

    print()
    print(results.round(3).set_index("pair").T.to_string())
    print(f"\nWrote {results_path} and {predictions_path}")
    return results

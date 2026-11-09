"""Shared utilities for datasets, CLIP loading, and feature extraction.

Used across Part A (Signal), Part B (Finetune), Part C (Swap), and Part D (Confidence).
Imports of torch, torchvision, and open_clip are deferred inside functions so that
lightweight maths and tests can run in environments without GPU or torch installed.
"""

from __future__ import annotations

import gc
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# Oxford label order; see torchvision/datasets/flowers102.py. Kept locally for
# torchvision releases which do not expose Flowers102.classes.
FLOWERS102_CLASSES: List[str] = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold",
    "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura",
    "geranium", "orange dahlia", "pink-yellow dahlia", "cautleya spicata",
    "japanese anemone", "black-eyed susan", "silverbush", "californian poppy",
    "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium",
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily"
]

# Standard class names for EuroSAT (satellite land-use classes)
EUROSAT_CLASSES: List[str] = [
    "annual crop", "forest", "herbaceous vegetation", "highway", "industrial",
    "pasture", "permanent crop", "residential", "river", "sea lake"
]


def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Read project configuration from YAML."""
    if not os.path.exists(path):
        # Fallback to look one directory up if called from scripts or notebooks
        alt_path = os.path.join(os.path.dirname(__file__), "..", path)
        if os.path.exists(alt_path):
            path = alt_path
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_clip(
    ckpt: Optional[str] = None,
    device: str = "cpu",
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, Any]:
    """Load CLIP ViT-B/32 in eval mode.

    Returns:
        (model, preprocess_transform, tokenizer)
    """
    import open_clip
    import torch

    cfg = cfg if cfg is not None else load_config()
    model_name = cfg["model"]["name"]
    pretrained = cfg["model"]["pretrained"]

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained,
        # OpenCLIP 3.3 does not automatically apply this pretrained-tag setting.
        # OpenAI weights require QuickGELU in BOTH towers, including when loading
        # our own state dict (activation functions are absent from state dicts).
        force_quick_gelu=pretrained == "openai",
    )

    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            for k in ("state_dict", "model"):
                if k in state and isinstance(state[k], dict):
                    state = state[k]
                    break
        clean_state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=True)

    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def free_model(model: Any) -> None:
    """Move model off GPU and release memory."""
    import torch

    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_dataset_loader(
    dataset_name: str,
    preprocess: Any,
    cfg: Dict[str, Any],
    split: str = "test",
) -> Tuple[Any, List[str], List[str]]:
    """Build a deterministic DataLoader for any of the 4 benchmark datasets.

    Supports: 'OxfordIIITPet', 'Flowers102', 'EuroSAT', 'DTD'.
    Applies the team standard seed (0) and at most 2000 images per split.
    Pets train maps to trainval. EuroSAT test is the first capped seed-0
    permutation block; train is the next block, with no overlap.

    Returns:
        (DataLoader, list_of_image_ids, list_of_classnames)
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    import torchvision.datasets as tv_datasets

    data_cfg = cfg["data"]
    root = data_cfg["root"]
    max_images = data_cfg.get("max_images", 2000)
    seed = data_cfg.get("seed", 0)

    ds_name_lower = dataset_name.lower().replace("-", "").replace("_", "")

    if split not in {"train", "test"}:
        raise ValueError("Use split='train' or split='test'. Pets train maps to trainval.")
    if not 0 < max_images <= 2000:
        raise ValueError("max_images must be between 1 and 2000.")

    if "pet" in ds_name_lower:
        ds = tv_datasets.OxfordIIITPet(
            root=root, split="trainval" if split == "train" else "test",
            download=True, transform=preprocess
        )
        classnames = [name.replace("_", " ") for name in ds.classes]

    elif "flower" in ds_name_lower:
        ds = tv_datasets.Flowers102(
            root=root, split=split, download=True, transform=preprocess
        )
        classnames = FLOWERS102_CLASSES

    elif "eurosat" in ds_name_lower:
        ds = tv_datasets.EuroSAT(
            root=root, download=True, transform=preprocess
        )
        classnames = EUROSAT_CLASSES

    elif "dtd" in ds_name_lower:
        ds = tv_datasets.DTD(
            root=root, split=split, download=True, transform=preprocess
        )
        classnames = [name.replace("_", " ") for name in ds.classes]

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Expected Pets, Flowers102, EuroSAT, or DTD.")

    # Apply deterministic shuffling & limit to max_images
    generator = torch.Generator().manual_seed(seed)
    total_len = len(ds)
    num_samples = min(total_len, max_images)
    permutation = torch.randperm(total_len, generator=generator)
    # EuroSAT has no official split. Preserve the shared test subset and select
    # training images from the remainder of the SAME seed-0 permutation.
    start = num_samples if "eurosat" in ds_name_lower and split == "train" else 0
    permuted_idx = permutation[start:start + num_samples].tolist()
    if not permuted_idx:
        raise ValueError("No disjoint training images remain after the EuroSAT test holdout.")

    subset = Subset(ds, permuted_idx)
    loader = DataLoader(
        subset, batch_size=cfg.get("finetune", {}).get("batch_size", 64),
        shuffle=False, num_workers=data_cfg.get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
    )

    prefix = dataset_name if split == "test" else f"{dataset_name}_train"
    image_ids = [f"{prefix}_{idx}" for idx in permuted_idx]
    return loader, image_ids, classnames


def extract_image_features(
    model: Any, loader: Any, device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode all images in loader and return float32 L2-normalized features and labels."""
    import torch

    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            targets = batch[1]

            encoded = model.encode_image(images).float()
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)

            features.append(encoded.cpu().numpy().astype(np.float32))
            if isinstance(targets, torch.Tensor):
                labels.append(targets.cpu().numpy())
            else:
                labels.append(np.asarray(targets))

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def extract_text_features(
    model: Any,
    tokenizer: Any,
    classnames: Sequence[str],
    templates: Sequence[str],
    device: str,
) -> np.ndarray:
    """Build L2-normalized prompt-ensembled text feature vectors for each class.

    For each class, encodes all 10 prompt templates, averages the normalized
    embeddings, and re-normalizes the result.

    Returns:
        [C, d] float32 array.
    """
    import torch

    class_features: List[np.ndarray] = []

    with torch.no_grad():
        for classname in classnames:
            prompts = [template.format(classname) for template in templates]
            tokens = tokenizer(prompts).to(device)
            encoded = model.encode_text(tokens).float()
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            mean_feature = encoded.mean(dim=0)
            mean_feature = mean_feature / mean_feature.norm()
            class_features.append(mean_feature.cpu().numpy().astype(np.float32))

    return np.stack(class_features, axis=0)

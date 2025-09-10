import os
from typing import List, Tuple, Dict
from PIL import Image

def list_images_with_labels(root_dir: str, img_ext: str = ".png") -> List[Tuple[str, int]]:
    """
    Expects a folder layout:
        root_dir/
          ClassA/*.png
          ClassB/*.png
          ...
    Returns list of (image_path, class_index) and an ordered class_name list.
    """
    if not os.path.isdir(root_dir):
        return []
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    items: List[Tuple[str, int]] = []
    for ci, cname in enumerate(classes):
        cdir = os.path.join(root_dir, cname)
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(img_ext):
                items.append((os.path.join(cdir, fn), ci))
    return items

def read_image_gray(path: str):
    """
    Loads an image as 8-bit grayscale PIL.Image.
    """
    img = Image.open(path).convert("L")
    return img

def class_names(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    return sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])


# ---
# NOTE on class imbalance (for future CNN training):
# Example (PyTorch):
#
# targets = [...]  # a list/array of int labels aligned with dataset indexing
# counts = torch.bincount(torch.tensor(targets, dtype=torch.long))
# class_weights = 1.0 / counts.float()
# sample_weights = class_weights[torch.tensor(targets, dtype=torch.long)]
# sampler = torch.utils.data.WeightedRandomSampler(
#     weights=sample_weights, num_samples=len(sample_weights), replacement=True
# )
# loader = DataLoader(dataset, batch_size=16, sampler=sampler)
#
# Rule-based classification in this repo does not require DataLoader,
# so this is only relevant when you switch to a CNN/ML classifier.

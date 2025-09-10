import os
from PIL import Image
from typing import List, Tuple

def list_pairs(root_dir: str, img_ext: str = ".png", mask_ext: str = ".png") -> List[Tuple[str, str]]:
    images_dir = os.path.join(root_dir, "images")
    masks_dir  = os.path.join(root_dir, "masks")
    pairs = []
    if not os.path.isdir(images_dir):
        return pairs
    for fn in sorted(os.listdir(images_dir)):
        if not fn.endswith(img_ext): 
            continue
        img_p = os.path.join(images_dir, fn)
        mask_p = os.path.join(masks_dir, fn.replace(img_ext, mask_ext))
        if os.path.exists(mask_p):
            pairs.append((img_p, mask_p))
    return pairs
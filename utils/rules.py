from typing import Dict, Tuple, List
import numpy as np
from PIL import Image

def _roi_crop(img: Image.Image, box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    w, h = img.size
    x0 = max(0, min(x0, w-1)); x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h-1)); y1 = max(0, min(y1, h))
    crop = img.crop((x0, y0, x1, y1)).convert("L")
    return np.array(crop, dtype=np.uint8)

def _mean_intensity(arr: np.ndarray) -> float:
    if arr.size == 0: return 0.0
    return float(arr.mean())

def _bright_count(arr: np.ndarray, thr: int) -> int:
    return int((arr >= thr).sum())

def detect_features(img: Image.Image, cfg: Dict) -> Dict[str, bool]:
    roi = cfg["rules"]["roi"]
    thr = cfg["rules"]["thresholds"]

    ilp = _roi_crop(img, tuple(roi["iliopectineal"]))
    ilis = _roi_crop(img, tuple(roi["ilioischial"]))
    spur = _roi_crop(img, tuple(roi["spur_area"]))
    pw   = _roi_crop(img, tuple(roi["posterior_wall"]))
    iw   = _roi_crop(img, tuple(roi["iliac_wing"]))

    bright_thr = int(thr.get("bright_thr", 200))

    # lines by bright-pixel counts
    line_pixels_thr = int(thr.get("line_pixels", 620))
    ilp_present = _bright_count(ilp, bright_thr) >= line_pixels_thr
    ilis_present = _bright_count(ilis, bright_thr) >= line_pixels_thr

    feats = {}
    feats["iliopectineal_broken"] = not ilp_present
    feats["ilioischial_broken"]   = not ilis_present

    # posterior wall by bright-pixel count (more robust)
    pw_pixels_thr = int(thr.get("pw_pixels", 800))
    feats["posterior_wall_frag"]  = _bright_count(pw, bright_thr) >= pw_pixels_thr

    # spur by bright-pixel count
    feats["spur_sign"]            = _bright_count(spur, bright_thr) >= int(thr.get("spur_pixels", 100))

    # iliac wing by mean intensity is fine in synthetic setup
    feats["iliac_wing_involved"]  = _mean_intensity(iw) >= float(thr.get("fragment_mean", 170))
    return feats

def classify_from_features(feats: Dict[str, bool], class_order: List[str]) -> int:
    ilp_broken = feats["iliopectineal_broken"]
    ilis_broken = feats["ilioischial_broken"]
    pw_frag = feats["posterior_wall_frag"]
    spur = feats["spur_sign"]
    iw_inv = feats["iliac_wing_involved"]

    if ilp_broken and ilis_broken and spur:
        return class_order.index("BC") if "BC" in class_order else 0
    if ilp_broken and ilis_broken and not spur:
        return class_order.index("T") if "T" in class_order else 0
    if ilis_broken and pw_frag:
        return class_order.index("PC") if "PC" in class_order else 0
    if pw_frag and not ilp_broken and not ilis_broken:
        return class_order.index("PW") if "PW" in class_order else 0
    if ilp_broken and iw_inv:
        return class_order.index("AC") if "AC" in class_order else 0
    if ilp_broken and "AC" in class_order: return class_order.index("AC")
    if ilis_broken and "PC" in class_order: return class_order.index("PC")
    if pw_frag and "PW" in class_order: return class_order.index("PW")
    return 0

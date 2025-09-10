import argparse, os, random
from PIL import Image, ImageDraw

# Reproducible seed
random.seed(2025)

W, H = 256, 256

# ---------- low-level helpers ----------
def _noise_bg(img, level=40):
    for y in range(H):
        for x in range(W):
            if (x + y) % 37 == 0:
                img.putpixel((x, y), level)

def _bright_rect(draw, xy, val=230):
    draw.rectangle(xy, fill=val)

def _bright_line(draw, y, broken=False, val=230, thick=3):
    if broken:
        _bright_rect(draw, [20, y, 120, y + thick], val)
        _bright_rect(draw, [140, y, 236, y + thick], val)
    else:
        _bright_rect(draw, [20, y, 236, y + thick], val)

def _spur(draw):
    pts = [(190, 190), (225, 185), (215, 220)]
    draw.polygon(pts, fill=240)

def _pw_block(draw):
    _bright_rect(draw, [45, 160, 85, 195], val=210)

def _iliac_wing_mark(draw):
    draw.polygon([(25, 25), (65, 25), (25, 65)], fill=210)

# ---------- segmentation pair (blob) ----------
def gen_seg_pair():
    img = Image.new("L", (W, H), 0)
    mask = Image.new("L", (W, H), 0)
    _noise_bg(img)
    # oval "organ-like" blob as mask
    draw_m = ImageDraw.Draw(mask)
    rw, rh = random.randint(60, 120), random.randint(60, 120)
    x0 = random.randint(20, W - rw - 20)
    y0 = random.randint(20, H - rh - 20)
    draw_m.ellipse([x0, y0, x0 + rw, y0 + rh], fill=255)
    # paint brighter texture inside mask
    for y in range(H):
        for x in range(W):
            if mask.getpixel((x, y)) > 0:
                img.putpixel((x, y), 120 + ((x + y) % 40))
    return img, mask

# ---------- classification image (Letournel cues; toy) ----------
def gen_cls_image(label):
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    _noise_bg(img)
    # intact iliopectineal / ilioischial base lines
    _bright_line(draw, y=44, broken=False)    # iliopectineal
    _bright_line(draw, y=124, broken=False)   # ilioischial

    if label == "AC":
        _bright_rect(draw, [20, 40, 236, 48], val=0)
        _bright_line(draw, y=44, broken=True)
        _iliac_wing_mark(draw)
    elif label == "PC":
        _bright_rect(draw, [20, 120, 236, 128], val=0)
        _bright_line(draw, y=124, broken=True)
        _pw_block(draw)
    elif label == "PW":
        _pw_block(draw)
    elif label == "T":
        _bright_rect(draw, [20, 40, 236, 48], val=0)
        _bright_line(draw, y=44, broken=True)
        _bright_rect(draw, [20, 120, 236, 128], val=0)
        _bright_line(draw, y=124, broken=True)
    elif label == "BC":
        _bright_rect(draw, [20, 40, 236, 48], val=0)
        _bright_line(draw, y=44, broken=True)
        _bright_rect(draw, [20, 120, 236, 128], val=0)
        _bright_line(draw, y=124, broken=True)
        _spur(draw)
    return img

def save_seg_dataset(out_dir, n=64):
    tr_img = os.path.join(out_dir, "train/images"); tr_msk = os.path.join(out_dir, "train/masks")
    va_img = os.path.join(out_dir, "val/images");   va_msk = os.path.join(out_dir, "val/masks")
    for d in [tr_img, tr_msk, va_img, va_msk]:
        os.makedirs(d, exist_ok=True)
    for i in range(n):
        img, msk = gen_seg_pair()
        if i < int(0.8 * n):
            img.save(os.path.join(tr_img, f"{i:04d}.png"))
            msk.save(os.path.join(tr_msk, f"{i:04d}.png"))
        else:
            img.save(os.path.join(va_img, f"{i:04d}.png"))
            msk.save(os.path.join(va_msk, f"{i:04d}.png"))

def save_cls_dataset(out_dir, n_per_class=40):
    classes = ["AC", "PC", "PW", "T", "BC"]
    for split in ["train", "val"]:
        for c in classes:
            os.makedirs(os.path.join(out_dir, split, c), exist_ok=True)
    for c in classes:
        for i in range(n_per_class):
            img = gen_cls_image(c)
            split = "train" if i < int(0.8 * n_per_class) else "val"
            img.save(os.path.join(out_dir, split, c, f"{c}_{i:04d}.png"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_out", default="data/synth")
    ap.add_argument("--seg_n", type=int, default=64)
    ap.add_argument("--cls_out", default="data/synth_cls")
    ap.add_argument("--cls_n_per_class", type=int, default=40)
    args = ap.parse_args()
    save_seg_dataset(args.seg_out, args.seg_n)
    save_cls_dataset(args.cls_out, args.cls_n_per_class)
    print(f"[OK] segmentation pairs -> {args.seg_out} ; classification -> {args.cls_out}")

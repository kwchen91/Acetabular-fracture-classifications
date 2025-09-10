import argparse, os
from PIL import Image, ImageDraw

"""
Synthetic generator for acetabular fracture "classes" using bright lines/shapes
that our rule engine can detect in predefined ROIs.

Classes created:
- AC : anterior column pattern (iliopectineal broken + iliac wing mark)
- PC : posterior column pattern (ilioischial broken + posterior wall fragment)
- PW : isolated posterior wall fragment
- T  : both lines broken, no spur
- BC : both lines broken + spur sign
"""

W, H = 256, 256

def draw_background_noise(img: Image.Image, level: int = 50):
    # very light noise-ish background so lines stand out
    for y in range(H):
        for x in range(W):
            if (x + y) % 37 == 0:
                img.putpixel((x, y), level)

def draw_line(draw: ImageDraw.ImageDraw, y: int, broken: bool):
    """
    Draw a bright horizontal "line" at row y. If broken=True, leave a gap.
    """
    color = 230
    thickness = 3
    if broken:
        # draw left and right segments leaving a gap in the middle
        draw.rectangle([20, y, 120, y + thickness], fill=color)
        draw.rectangle([140, y, 236, y + thickness], fill=color)
    else:
        draw.rectangle([20, y, 236, y + thickness], fill=color)

def draw_spur(draw: ImageDraw.ImageDraw):
    """
    Draw a small bright triangle spur in the lower-right area.
    """
    color = 240
    pts = [(190, 190), (225, 185), (215, 220)]
    draw.polygon(pts, fill=color)

def draw_posterior_wall_fragment(draw: ImageDraw.ImageDraw):
    """
    Draw a bright block around posterior wall ROI.
    """
    color = 210
    draw.rectangle([45, 160, 85, 195], fill=color)

def draw_iliac_wing_mark(draw: ImageDraw.ImageDraw):
    """
    Draw a bright diagonal/patch in upper-left area.
    """
    color = 210
    draw.polygon([(25, 25), (65, 25), (25, 65)], fill=color)

def make_image(label: str) -> Image.Image:
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    draw_background_noise(img)

    # two baseline lines (intact)
    draw_line(draw, y=44, broken=False)   # iliopectineal
    draw_line(draw, y=124, broken=False)  # ilioischial

    if label == "AC":
        # break iliopectineal + iliac wing involvement
        draw.rectangle([20, 40, 236, 48], fill=0)  # erase line
        draw_line(draw, y=44, broken=True)
        draw_iliac_wing_mark(draw)

    elif label == "PC":
        # break ilioischial + posterior wall fragment
        draw.rectangle([20, 120, 236, 128], fill=0)
        draw_line(draw, y=124, broken=True)
        draw_posterior_wall_fragment(draw)

    elif label == "PW":
        # intact lines; posterior wall fragment only
        draw_posterior_wall_fragment(draw)

    elif label == "T":
        # both lines broken; no spur
        draw.rectangle([20, 40, 236, 48], fill=0)
        draw_line(draw, y=44, broken=True)
        draw.rectangle([20, 120, 236, 128], fill=0)
        draw_line(draw, y=124, broken=True)

    elif label == "BC":
        # both lines broken + spur sign
        draw.rectangle([20, 40, 236, 48], fill=0)
        draw_line(draw, y=44, broken=True)
        draw.rectangle([20, 120, 236, 128], fill=0)
        draw_line(draw, y=124, broken=True)
        draw_spur(draw)

    else:
        # default: intact lines (falls outside main classes)
        pass

    return img

def main(out_dir: str, n_per_class: int):
    classes = ["AC", "PC", "PW", "T", "BC"]
    for split in ["train", "val"]:
        for c in classes:
            os.makedirs(os.path.join(out_dir, split, c), exist_ok=True)

    # generate images
    for c in classes:
        for i in range(n_per_class):
            img = make_image(c)
            split = "train" if i < int(0.8 * n_per_class) else "val"
            img.save(os.path.join(out_dir, split, c, f"{c}_{i:04d}.png"))
    print(f"Generated {n_per_class} images per class at {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth_cls")
    ap.add_argument("--n_per_class", type=int, default=40)
    args = ap.parse_args()
    main(args.out, args.n_per_class)

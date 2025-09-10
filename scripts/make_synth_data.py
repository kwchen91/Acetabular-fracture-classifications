import argparse, os, random
from PIL import Image, ImageDraw
random.seed(2025)

def draw_blob(draw, w, h):
    rw, rh = random.randint(int(0.2*w), int(0.5*w)), random.randint(int(0.2*h), int(0.5*h))
    x0 = random.randint(10, w-rw-10)
    y0 = random.randint(10, h-rh-10)
    draw.ellipse([x0, y0, x0+rw, y0+rh], fill=255)

def generate_pair(w=256, h=256):
    img = Image.new("L", (w,h), 0)
    mask = Image.new("L", (w,h), 0)
    draw_m = ImageDraw.Draw(mask)
    draw_blob(draw_m, w, h)
    for y in range(h):
        for x in range(w):
            if mask.getpixel((x,y))>0:
                img.putpixel((x,y), 120 + (x+y)%40)
    return img, mask

def main(out_dir, n):
    tr_img = os.path.join(out_dir, "train/images"); tr_msk = os.path.join(out_dir, "train/masks")
    va_img = os.path.join(out_dir, "val/images");   va_msk = os.path.join(out_dir, "val/masks")
    for d in [tr_img, tr_msk, va_img, va_msk]:
        os.makedirs(d, exist_ok=True)
    for i in range(n):
        img, mask = generate_pair()
        if i < int(0.8*n):
            img.save(os.path.join(tr_img, f"{i:04d}.png"))
            mask.save(os.path.join(tr_msk, f"{i:04d}.png"))
        else:
            img.save(os.path.join(va_img, f"{i:04d}.png"))
            mask.save(os.path.join(va_msk, f"{i:04d}.png"))
    print(f"Generated {n} pairs at {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    main(args.out, args.n)
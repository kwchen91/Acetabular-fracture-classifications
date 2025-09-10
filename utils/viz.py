from PIL import Image

def overlay(img_path, mask_path, out_path):
    img = Image.open(img_path).convert("L").convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    r = Image.new("L", mask.size, 255)
    g = Image.new("L", mask.size, 0)
    b = Image.new("L", mask.size, 0)
    a = mask.point(lambda p: int(p>0)*120)
    overlay = Image.merge("RGBA", (r,g,b,a))
    composed = Image.alpha_composite(img, overlay)
    composed.save(out_path)
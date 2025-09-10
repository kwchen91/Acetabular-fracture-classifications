import argparse, os, glob, shutil, yaml

def main(cfg_path, images_dir):
    cfg = yaml.safe_load(open(cfg_path, 'r'))
    os.makedirs(cfg["viz_dir"], exist_ok=True)
    imgs = glob.glob(os.path.join(images_dir, "*.png"))
    if not imgs:
        print("No PNG images found. Provide a folder of .png images.")
        return
    for i, p in enumerate(imgs[:5]):
        dst = os.path.join(cfg["viz_dir"], f"pred_{i}.png")
        shutil.copy(p, dst)
    print(f"Saved {min(5, len(imgs))} mock predictions to {cfg['viz_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/segmenter_unet.yaml")
    parser.add_argument("--images", required=True, help="folder of PNGs")
    args = parser.parse_args()
    main(args.cfg, args.images)

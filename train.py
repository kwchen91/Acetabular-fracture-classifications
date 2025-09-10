import argparse, os, time, yaml, csv
from utils.seed import set_seed
from datasets.loader_segmenter import list_pairs
from utils.losses import fake_loss
from utils import viz as viz_utils

def simulate_epoch(epoch, total_epochs):
    # Simple decreasing curve to simulate training loss
    return fake_loss(1.0 - (epoch + 1) / (total_epochs + 2))

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, 'r'))
    set_seed(cfg.get("seed", 2025))

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["viz_dir"], exist_ok=True)

    train_pairs = list_pairs(cfg["data"]["train_dir"], cfg["data"]["img_ext"], cfg["data"]["mask_ext"])
    val_pairs   = list_pairs(cfg["data"]["val_dir"],   cfg["data"]["img_ext"], cfg["data"]["mask_ext"])

    if not train_pairs or not val_pairs:
        print("⚠️ No synthetic data found or folder structure incomplete. Please run:")
        print("    python scripts/make_synth_data.py --out data/synth --n 64")
        print("Then re-run train.py.")

    # Write training history to CSV
    history_path = os.path.join(cfg["log_dir"], "history.csv")
    with open(history_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_metric"])
        for e in range(cfg["train"]["epochs"]):
            t0 = time.time()
            train_loss = simulate_epoch(e, cfg["train"]["epochs"])
            # Simulate validation metric increasing gradually with epochs
            val_metric = 0.5 + 0.4 * (e + 1) / cfg["train"]["epochs"]  # final value ~0.9
            w.writerow([e, f"{train_loss:.4f}", f"{val_metric:.4f}"])
            print(f"E{e:03d} loss={train_loss:.4f} val={val_metric:.4f} time={time.time()-t0:.1f}s")

    # Save 1–3 overlay examples if validation data is available
    if len(val_pairs) > 0:
        for i, (img_p, mask_p) in enumerate(val_pairs[:3]):
            out_p = os.path.join(cfg["viz_dir"], f"overlay_{i}.png")
            viz_utils.overlay(img_p, mask_p, out_p)

    # Save a placeholder "best checkpoint"
    with open(os.path.join(cfg["ckpt_dir"], "best.txt"), "w") as f:
        f.write("placeholder for best checkpoint\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/segmenter_unet.yaml")
    args = parser.parse_args()
    main(args.cfg)

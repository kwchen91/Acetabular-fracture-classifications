import argparse, os, csv, yaml

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, 'r'))
    hist_p = os.path.join(cfg["log_dir"], "history.csv")
    if not os.path.exists(hist_p):
        print("No history.csv found. Please run train.py first.")
        return
    rows = list(csv.DictReader(open(hist_p)))
    if rows:
        last = rows[-1]
        print(f'Final val_metric: {last["val_metric"]}')
    else:
        print("Empty history.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/segmenter_unet.yaml")
    args = parser.parse_args()
    main(args.cfg)
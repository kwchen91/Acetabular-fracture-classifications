import argparse, os, yaml, csv
import numpy as np
from typing import List, Tuple
from PIL import Image
from datasets.loader_classifier import list_images_with_labels, read_image_gray, class_names
from utils.rules import detect_features, classify_from_features
import matplotlib.pyplot as plt

def confusion_matrix(y_true: List[int], y_pred: List[int], ncls: int) -> np.ndarray:
    cm = np.zeros((ncls, ncls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def per_class_accuracy(cm: np.ndarray) -> List[float]:
    accs = []
    for i in range(cm.shape[0]):
        denom = cm[i].sum()
        accs.append(float(cm[i, i]) / denom if denom > 0 else 0.0)
    return accs

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))

    os.makedirs(cfg["output"]["log_dir"], exist_ok=True)

    # gather class lists from VAL set (authoritative for label order)
    classes = class_names(cfg["data"]["val_dir"])
    if not classes:
        print("No classes found in val_dir. Expected subfolders per class.")
        return
    print("Classes:", classes)

    val_items = list_images_with_labels(cfg["data"]["val_dir"], cfg["data"]["img_ext"])
    if not val_items:
        print("No images found in val_dir. Generate synthetic data first:")
        print("  python scripts/make_synth_fracture_cls.py --out data/synth_cls --n_per_class 40")
        return

    if cfg["rules"]["enabled"]:
        y_true, y_pred = [], []
        for path, label in val_items:
            img = read_image_gray(path)
            feats = detect_features(img, cfg)
            pred_idx = classify_from_features(feats, classes)
            y_true.append(label)
            y_pred.append(pred_idx)

        ncls = len(classes)
        cm = confusion_matrix(y_true, y_pred, ncls)
        acc = float(np.trace(cm)) / float(cm.sum()) if cm.sum() > 0 else 0.0
        pc_acc = per_class_accuracy(cm)

        # save a small report CSV
        report_csv = cfg["output"]["report_csv"]
        with open(report_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["accuracy", f"{acc:.4f}"])
            for i, c in enumerate(classes):
                w.writerow([f"acc_{c}", f"{pc_acc[i]:.4f}"])
        print(f"Accuracy: {acc:.4f}")
        for i, c in enumerate(classes):
            print(f"  {c}: {pc_acc[i]:.4f}")

        # plot confusion matrix
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion Matrix (rule-based)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(ncls)); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(ncls)); ax.set_yticklabels(classes)
        for i in range(ncls):
            for j in range(ncls):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        plt.tight_layout()
        plt.savefig(cfg["output"]["cm_path"], bbox_inches="tight")
        plt.close()
    else:
        print("rules.enabled == False; no ML baseline implemented in this lightweight template.")
        print("You can add a simple sklearn or PyTorch classifier later.")
        return

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/classifier_letournel.yaml")
    args = ap.parse_args()
    main(args.cfg)

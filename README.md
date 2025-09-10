# Acetabular Fracture Classification (Toy Baseline)

This is a **toy baseline** using synthetic data and a simple rule-based pipeline.  
Not a clinical model — intended for teaching and reproducibility only.

## Quickstart

```bash
pip install -r requirements.txt

# Generate synthetic data
python scripts/make_synth_data.py --out data/synth_cls --n 60

# Run rule-based classification
python eval.py --cfg configs/classifier_letournel.yaml

Layout
configs/     # hyper-params
data/        # (no real data in repo) + synth generator output
datasets/    # file pair listing
models/      # rule-based or CNN placeholder
utils/       # helpers (seed, viz, metrics)
scripts/     # synthetic data generator
train.py     # (optional) training script
eval.py      # evaluation / classification
outputs/     # logs, confusion matrix, reports

Results

Example confusion matrix (synthetic run):
Confusion matrix → outputs/logs/cm_cls.png
Classification report → outputs/logs/cls_report.csv

Notes
Rule-based classification inspired by Letournel cues:
Anterior/posterior line breaks
Spur sign
Posterior wall fragment
Thresholds tuned only on synthetic images (e.g. line break if mean gray < 180).
Next steps: add sklearn/SVM baseline and lightweight CNN.
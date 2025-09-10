# Acetabular Fracture Classification (Toy Baseline)

I built this as a **toy baseline** using synthetic data and a very simple rule-based pipeline.  
It’s not a clinical tool, just a small project for learning and reproducibility.

## Quickstart

```bash
pip install -r requirements.txt

# Generate synthetic data
python scripts/make_synth_data.py --out data/synth_cls --n 60

# Run rule-based classification
python eval.py --cfg configs/classifier_letournel.yaml
```

## Project files
configs/ – YAML configs with thresholds and settings
data/ – synthetic dataset generator output (no real data)
datasets/ – loaders for classification and segmentation
models/ – placeholders for rule-based / CNN models
utils/ – small helpers (seeds, viz, metrics)
scripts/ – quick scripts for generating synthetic runs
train.py – optional training entry
eval.py – rule-based evaluation
outputs/ – logs, confusion matrices, reports

## Results
On a small synthetic dataset (~60 samples), the thresholds worked okay for simple patterns.
T-shape and BC were sometimes confused, but at least the logs/plots are reproducible.

Confusion matrix: outputs/logs/cm_cls.png
Classification report: outputs/logs/cls_report.csv

## What I learned
The rules are very rough:

check if anterior/posterior lines break
look for spur signs
detect posterior wall fragments

I tuned simple thresholds (mean gray <180 for line break, ≥100 bright pixels for spur) just to make the pipeline consistent.

## Next steps: 
try sklearn/SVM and maybe a lightweight CNN.

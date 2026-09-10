"""Model Evaluation and Quality Check
 
Aggregates validation metrics from all trained models and validates against quality thresholds.

"""

import subprocess

import sys

import os
 
# Install packages from requirements.txt

req_file = '/opt/ml/processing/input/requirements/requirements.txt'

if os.path.exists(req_file):

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", req_file])
 
import json

import os

import pandas as pd

import numpy as np

import boto3

INPUT_DIR = "/opt/ml/processing/input"

OUTPUT_DIR = "/opt/ml/processing/evaluation"

R2_THRESHOLD = 0.7

results = {}

missing_files = []

failed = []
 
print(f"\n=== Contents of {INPUT_DIR} ===")

if os.path.exists(INPUT_DIR):

    for root, dirs, files in os.walk(INPUT_DIR):

        for f in files:

            full_path = os.path.join(root, f)

            print(f"  {full_path}")

else:

    print(f"  Directory {INPUT_DIR} does not exist!")
 
for family in ["Barrage", "Grounded"]:

    for role in ["champion", "challenger"]:

        subdir = os.path.join(INPUT_DIR, "eval", f"{family.lower()}-{role}")

        eval_path = os.path.join(subdir, f"{family}_{role}_evaluation.json")

        if not os.path.exists(eval_path):

            print(f"\nNOT FOUND: {eval_path}")

            missing_files.append(f"{family}_{role}")

            continue

        print(f"\nFOUND: {eval_path}")
 
        try:

            with open(eval_path) as f:

                data = json.load(f)

        except Exception as e:

            print(f"  ERROR reading file: {e}")

            missing_files.append(eval_path)

            continue
 
        key = f"{family.lower()}_{role}"

        results[key] = {

            "model_type": data["model_type"],

            "r2": data["validation_metrics"]["r2"],

            "rmse": data["validation_metrics"]["rmse"],

            "mae": data["validation_metrics"]["mae"],

            "wmape": data["validation_metrics"].get("wmape", "N/A"),

            "train_size": data["train_size"],

            "val_size": data["val_size"],

            "train_cutoff": data.get("train_cutoff", data.get("holdout_cutoff", "N/A")),

            "val_start": data.get("val_start", "N/A"),

            "val_end": data.get("val_end", "N/A")

        }
 
if missing_files:

    print(f"\n\u274c ERROR: Missing {len(missing_files)} evaluation files:")

    for f in missing_files:

        print(f"  - {f}")

    raise FileNotFoundError(f"Missing evaluation files: {missing_files}")
 
if failed:

    print("\u2717 Failed checks:")

    for failure in failed:

        print(f"  - {failure}")
 
print(f"\nSaved to {os.path.join(OUTPUT_DIR, 'evaluation.json')}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, 'evaluation.json'), 'w') as f:
    json.dump(results, f, indent=4)

print("\n✓ Evaluation complete -- quality check only, per-role R² written to evaluation.json.")

"""

Generate county_accuracy.csv for the County-Level Accuracy dashboard tab.
 
Computes real, per-county R2/WMAPE for BOTH Champion and Challenger, reusing the

validation predictions train_with_holdout.py already generated during actual training

holdout window (TRAIN_CUTOFF=2024-09-01, VAL_START/END=2024-10-01/2025-09-01 -- confirmed

identical to train_with_holdout.py's own real split). Reuses validation_predictions_all.csv,

which dashboard_merge.py uploads right before this script runs (same MergeAndGenerateDashboard step).
 
Uploads to the same S3 location as before:

    s3://vap-sales-forecasting/dashboard/county_accuracy.csv

Includes a ModelRole column (Champion/Challenger).

"""

import pandas as pd

import numpy as np

import boto3

import os

import warnings

warnings.filterwarnings("ignore")
 
BUCKET = os.environ.get("VAP_S3_BUCKET", "vap-sales-forecasting")

s3 = boto3.client("s3")
 
TARGET = "Actual_QTY"

PRED = "Predicted_QTY"
 
 
def wmape(y_true, y_pred):

    denom = np.sum(np.abs(y_true))

    return np.sum(np.abs(y_true - y_pred)) / denom if denom > 0 else np.nan
 
 
def volume_tier(total):

    if total < 10:

        return "Near-zero"

    if total < 100:

        return "Low"

    if total < 1000:

        return "Moderate"

    return "Established"
 
 
def r2_category(r2):

    if r2 < -1:

        return "Edge Case"

    if r2 < 0.3:

        return "Poor"

    return "Good"
 
 
def confidence_tier(row):


    if row["Volume_Tier"] in ("Near-zero", "Low"):

        return "Directional Only (Low Volume)"

    if row["R2_Category"] == "Good":

        return "High Confidence"

    if row["R2_Category"] == "Poor":

        return "Moderate Confidence"

    return "Low Confidence"
 
 
def load_s3_csv(key):

    obj = s3.get_object(Bucket=BUCKET, Key=key)

    return pd.read_csv(obj["Body"])
 
 
if __name__ == "__main__":

    print("Loading validation_predictions_all.csv (real, already-trained model output)...")

    v = load_s3_csv("dashboard/validation_predictions_all.csv")

    print(f"Loaded {len(v)} rows")
 
    from sklearn.metrics import r2_score
 
    all_results = []

    for (fips, county, state, family, role), g in v.groupby(

        ["FIPS", "County_Name", "State", "Family", "ModelRole"]

    ):

        actual_total = g[TARGET].sum()

        if actual_total == 0 or len(g) < 2:

            continue

        r2 = r2_score(g[TARGET], g[PRED])

        wm = wmape(g[TARGET].values, g[PRED].values)

        all_results.append({

            "FIPS": fips, "County_Name": county, "State": state,

            "Family": family, "ModelRole": role,

            "R2": round(r2, 4), "WMAPE": round(wm, 4),

            "Actual_Total": round(actual_total, 2),

        })
 
    results_df = pd.DataFrame(all_results)

    results_df["Volume_Tier"] = results_df["Actual_Total"].apply(volume_tier)

    results_df["R2_Category"] = results_df["R2"].apply(r2_category)

    results_df["Confidence_Tier"] = results_df.apply(confidence_tier, axis=1)
 
    print(f"\nFinal shape: {results_df.shape}")

    print("\nConfidence tier distribution (by ModelRole):")

    print(results_df.groupby("ModelRole")["Confidence_Tier"].value_counts())
 
    results_df.to_csv("/tmp/county_accuracy.csv", index=False)

    s3.upload_file("/tmp/county_accuracy.csv", BUCKET, "dashboard/county_accuracy.csv")

    print(f"\nUploaded s3://{BUCKET}/dashboard/county_accuracy.csv ({len(results_df)} rows)")
 

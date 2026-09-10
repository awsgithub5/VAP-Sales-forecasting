"""Merge step: combines the 4 per-family-role forecast chunks Repack generated into
one forecast_2yr_all_models.csv, then runs dashboard_merge.py (fact_table.csv,
national_trailing.csv) and generate_county_accuracy.py (using Scenario 1's real
validation predictions, per direct instruction -- Scenario 2 has no held-out period
to validate accuracy against, since it trains on the full dataset)."""
import os
import sys
import subprocess
import tarfile

# Extract the code_deps tarball (dashboard_merge.py, generate_county_accuracy.py,
# requirements.txt -- see _upload_code_deps in pipeline_vpc.py). ProcessingInput
# copies this tarball as-is, it does not auto-extract it.
CODE_DEPS_DIR = "/opt/ml/processing/input/code_deps"
if os.path.exists(CODE_DEPS_DIR):
    _tar_candidates = [f for f in os.listdir(CODE_DEPS_DIR) if f.endswith(".tar.gz")]
    if _tar_candidates:
        with tarfile.open(os.path.join(CODE_DEPS_DIR, _tar_candidates[0]), "r:gz") as _tar:
            _tar.extractall(CODE_DEPS_DIR)
        print(f"Extracted code deps from {_tar_candidates[0]} into {CODE_DEPS_DIR}")
else:
    raise FileNotFoundError(
        f"{CODE_DEPS_DIR!r} not found -- dashboard_merge.py and "
        f"generate_county_accuracy.py could not be located. Check that the code_deps "
        f"ProcessingInput in pipeline_vpc.py's build_merge_step() actually reached "
        f"this step."
    )

req_file = os.path.join(CODE_DEPS_DIR, "requirements.txt")
if os.path.exists(req_file):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", req_file])

import pandas as pd
import numpy as np
import boto3

BUCKET = os.environ.get("VAP_S3_BUCKET", "vap-sales-forecasting")
INPUT_DIR = "/opt/ml/processing/input/forecast_chunks"
SCRIPT_DIR = CODE_DEPS_DIR


def require_file(path, what):
    """DEFENSIVE: fail loudly and specifically if a prior step's output isn't where
    expected, rather than a cryptic downstream pandas/KeyError."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required {what} not found at {path!r} -- this step depends on all 4 "
            f"Repack{{Family}}{{Role}} steps having already completed and written "
            f"their forecast chunk here. Check each Repack step's own logs for "
            f"whether it actually reached PART 3 (forecast generation) successfully."
        )
    return path


def compute_demand_spikes(forecast_df):
    """For each FIPS+Family+ModelRole, finds the single largest month-over-month
    INCREASE across the 2-year forecast -- the month a Sales Head should be stocking
    up BEFORE, not after, for lead-time inventory planning. Reports absolute change
    as the primary signal (what actually matters for how much extra to stock), with
    percentage change suppressed (None, not a misleadingly huge number) when the
    baseline is too small to make a percentage meaningful -- a near-zero county going
    from 0.1 to 2 units is a real, useful 1.9-unit signal, not a genuine "1900%" story.
    Champion and Challenger are computed and reported separately, never silently
    combined into one number, same as everywhere else in this pipeline."""
    results = []
    forecast_df = forecast_df.copy()
   
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    forecast_df = forecast_df.sort_values(["FIPS", "Family", "ModelRole", "date"])
    for (fips, family, role), group in forecast_df.groupby(["FIPS", "Family", "ModelRole"]):
        group = group.sort_values("date").reset_index(drop=True)
        if len(group) < 2:
            continue
        group["prev_qty"] = group["Predicted_QTY"].shift(1)
        group["abs_change"] = group["Predicted_QTY"] - group["prev_qty"]
        group = group.dropna(subset=["abs_change"])
        if len(group) == 0:
            continue
        spike_row = group.loc[group["abs_change"].idxmax()]
        baseline = spike_row["prev_qty"]
        pct_change = (spike_row["abs_change"] / baseline * 100) if baseline > 0.5 else None
        results.append({
            "FIPS": fips, "County_Name": spike_row["County_Name"], "State": spike_row["State"],
            "Family": family, "ModelRole": role, "ModelName": spike_row["ModelName"],
            # Spike_Month is a real calendar date -- "months until spike" is left for
            # QuickSight's own TODAY() calculated field to compute at view time, not
            # baked in here, since the pipeline may run weeks before someone actually
            # looks at the dashboard and a baked-in count would go stale.
            "Spike_Month": spike_row["date"].strftime("%Y-%m-%d"),
            "Baseline_Before_Spike": round(baseline, 2),
            "Predicted_At_Spike": round(spike_row["Predicted_QTY"], 2),
            "Absolute_Increase": round(spike_row["abs_change"], 2),
            "Percentage_Increase": round(pct_change, 1) if pct_change is not None else None,
        })
    return pd.DataFrame(results)


def compute_state_feature_importance(importance_df, forecast_df):
    """Volume-weighted average of County-level SHAP values within each state -- state
    has no model of its own to compute real SHAP from directly, so this is the
    honest way to represent "state-level importance": a real rollup of real county
    values, weighted so a high-volume county's driver counts more than a near-zero
    county's, not a naive unweighted average across counties of wildly different size.
    Uses each county's October 2025 forecast volume (the same month County-level SHAP
    was computed for) as the weight."""
    county_imp = importance_df[importance_df["Level"] == "County"].copy()
    if len(county_imp) == 0:
        return pd.DataFrame(columns=importance_df.columns)

    forecast_df = forecast_df.copy()
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    oct_volumes = forecast_df[forecast_df["date"] == pd.Timestamp("2025-10-01")]
    weights = oct_volumes[["FIPS", "Family", "ModelRole", "Predicted_QTY"]].rename(
        columns={"Predicted_QTY": "weight"})
    county_imp = county_imp.merge(weights, on=["FIPS", "Family", "ModelRole"], how="left")
    # Tiny floor rather than 0 -- a near-zero-volume county's real driver still
    # counts a little, just far less than a high-volume county's, never dropped
    # entirely from the state picture.
    county_imp["weight"] = county_imp["weight"].fillna(0.01).clip(lower=0.01)

    state_rows = []
    for (state, family, role, feature), group in county_imp.groupby(
            ["State", "Family", "ModelRole", "Feature"]):
        weighted_avg = np.average(group["SHAP_Value"], weights=group["weight"])
        state_rows.append({
            "Level": "State", "FIPS": None, "County_Name": None, "State": state,
            "Family": family, "ModelRole": role, "ModelName": group["ModelName"].iloc[0],
            "Feature": feature, "Friendly_Name": group["Friendly_Name"].iloc[0],
            "SHAP_Value": round(weighted_avg, 3), "Is_External": True, "Rank": None,
        })
    return pd.DataFrame(state_rows)


if __name__ == "__main__":
    print("=== Combining 4 forecast chunks from Repack ===")
    chunks = []
    for family in ["barrage", "grounded"]:
        for role in ["champion", "challenger"]:
            path = os.path.join(INPUT_DIR, f"{family}-{role}", f"forecast_2yr_{family}_{role}.csv")
            require_file(path, f"{family}/{role} forecast chunk")
            chunk = pd.read_csv(path)
            chunks.append(chunk)
            print(f"Loaded {family}/{role}: {len(chunk)} rows")

    combined = pd.concat(chunks, ignore_index=True)
    print(f"Combined: {len(combined)} total rows across all 4 family/role combinations")

    combined_path = "/tmp/forecast_2yr_all_models.csv"
    combined.to_csv(combined_path, index=False)
    s3 = boto3.client("s3")
    # SAME S3 paths as before -- dashboard_merge.py's own STEP 2 reads this exact key,
    # unchanged, so it doesn't need to know anything changed upstream of it.
    s3.upload_file(combined_path, BUCKET, "comparison/forecast_2yr_all_models.csv")
    s3.upload_file(combined_path, BUCKET, "dashboard/forecast_2yr_all_models.csv")
    print(f"Uploaded combined forecast to s3://{BUCKET}/comparison/ and dashboard/")

    print("\n=== Computing county demand spikes ===")
    spikes_df = compute_demand_spikes(combined)
    spikes_path = "/tmp/county_demand_spikes.csv"
    spikes_df.to_csv(spikes_path, index=False)
    s3.upload_file(spikes_path, BUCKET, "dashboard/county_demand_spikes.csv")
    print(f"Uploaded {len(spikes_df)} rows to s3://{BUCKET}/dashboard/county_demand_spikes.csv")

    print("\n=== Combining County + National feature importance chunks from Repack ===")
    # "feature importance all" and "external only" dashboard graphs -- filter on
    # Is_External for either view, one file, no duplication. An explicit customer
    # requirement re-wired here after being accidentally left behind during an
    # earlier rearchitecture -- see repack_evaluation.py PART 4. Now also includes
    # EVERY external feature (not just top-5) at County level, per direct
    # instruction -- users need the full ranked list to decide which feature to go
    # query the What-If agent about.
    IMPORTANCE_INPUT_DIR = "/opt/ml/processing/input/importance_chunks"
    importance_chunks = []
    for family in ["barrage", "grounded"]:
        for role in ["champion", "challenger"]:
            path = os.path.join(IMPORTANCE_INPUT_DIR, f"{family}-{role}",
                                 f"feature_importance_{family}_{role}.csv")
            require_file(path, f"{family}/{role} feature importance chunk")
            chunk = pd.read_csv(path)
            importance_chunks.append(chunk)
            print(f"Loaded {family}/{role}: {len(chunk)} rows")
    importance_combined = pd.concat(importance_chunks, ignore_index=True)

    print("\n=== Computing State-level feature importance (volume-weighted rollup) ===")
    # State has no model of its own to compute real SHAP from -- this is a genuine,
    # honest volume-weighted average of the real County-level SHAP values within
    # each state, using each county's October 2025 forecast volume as the weight, so
    # a high-volume county's real driver counts more than a near-zero county's.
    state_importance = compute_state_feature_importance(importance_combined, combined)
    importance_combined = pd.concat([importance_combined, state_importance], ignore_index=True)

    importance_path = "/tmp/feature_importance_global.csv"
    importance_combined.to_csv(importance_path, index=False)
    s3.upload_file(importance_path, BUCKET, "dashboard/feature_importance_global.csv")
    print(f"Uploaded {len(importance_combined)} rows "
          f"({(importance_combined['Level']=='National').sum()} National, "
          f"{(importance_combined['Level']=='State').sum()} State, "
          f"{(importance_combined['Level']=='County').sum()} County) "
          f"to s3://{BUCKET}/dashboard/feature_importance_global.csv")

    print("\n=== Running dashboard_merge.py ===")
    dashboard_merge_path = require_file(
        os.path.join(SCRIPT_DIR, "dashboard_merge.py"), "dashboard_merge.py"
    )
    subprocess.check_call([sys.executable, dashboard_merge_path])
    print("\u2713 Dashboard merge complete")

    print("\n=== Running generate_county_accuracy.py ===")
    # Uses Scenario 1's real validation_predictions_all.csv (which dashboard_merge.py
    # just produced above) -- per direct instruction, county accuracy is validated
    # against Scenario 1's genuine held-out period, not Scenario 2 (which has no
    # holdout at all, since it trains on the full dataset through the real cutoff).
    county_accuracy_path = require_file(
        os.path.join(SCRIPT_DIR, "generate_county_accuracy.py"), "generate_county_accuracy.py"
    )
    subprocess.check_call([sys.executable, county_accuracy_path])
    print("\u2713 County accuracy complete")

    print("\n\u2713 Merge step complete: forecast_2yr_all_models.csv, fact_table.csv, "
          "national_trailing.csv, county_accuracy.csv all generated and uploaded.")

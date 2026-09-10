"""Repack: extract Scenario 1's evaluation.json (UNCHANGED, feeds EvaluateAllModels'
quality gate) -- AND train Scenario 2 (full data through the real cutoff, no holdout),
package it for registration, and generate this family/role's 2-year forecast chunk
using that SAME Scenario 2 model object -- guaranteeing the model that gets deployed
and the model that generated the forecast are identical, never two separate objects.
"""
import os
import sys
import json
import pickle
import tarfile
import argparse
import subprocess

# Extract the code_deps tarball (train_with_holdout.py, inference.py,
# requirements.txt -- see _upload_code_deps in pipeline_vpc.py) and add its
# directory to sys.path BEFORE installing requirements or importing anything from
# it. ProcessingInput copies this tarball as-is, it does not auto-extract it.
CODE_DEPS_DIR = "/opt/ml/processing/input/code_deps"
if os.path.exists(CODE_DEPS_DIR):
    _tar_candidates = [f for f in os.listdir(CODE_DEPS_DIR) if f.endswith(".tar.gz")]
    if _tar_candidates:
        with tarfile.open(os.path.join(CODE_DEPS_DIR, _tar_candidates[0]), "r:gz") as _tar:
            _tar.extractall(CODE_DEPS_DIR)
        print(f"Extracted code deps from {_tar_candidates[0]} into {CODE_DEPS_DIR}")
    sys.path.insert(0, CODE_DEPS_DIR)
else:
    raise FileNotFoundError(
        f"{CODE_DEPS_DIR!r} not found -- train_with_holdout.py and inference.py "
        f"could not be located. Check that the code_deps ProcessingInput in "
        f"pipeline_vpc.py's build_repack_steps() actually reached this step."
    )

# Install packages from requirements.txt -- SAME file, SAME pinned versions used
# everywhere else in this pipeline (xgboost, lightgbm, shap, pandas, numpy,
# scikit-learn) -- no new/different packages introduced by this step. Now inside
# CODE_DEPS_DIR, since that's where _upload_code_deps bundled it.
req_file = os.path.join(CODE_DEPS_DIR, "requirements.txt")
if os.path.exists(req_file):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", req_file])

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# Reuse train_with_holdout.py's OWN functions directly for Scenario 2 -- not a second,
# separately-maintained copy of the training logic that could drift out of sync.
# Guarantees "same training parameters as Scenario 1" by construction, not by
# copy-pasted values someone could accidentally edit differently later.
import train_with_holdout as t1
# Reuse the live endpoint's own recursive forecasting logic directly, same reasoning.
import inference as inf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/input/scenario1_model")
    parser.add_argument("--train-data-dir", type=str, default="/opt/ml/processing/input/data")
    parser.add_argument("--eval-output-dir", type=str, default="/opt/ml/processing/output/evaluation")
    parser.add_argument("--model-output-dir", type=str, default="/opt/ml/processing/output/model")
    parser.add_argument("--forecast-output-dir", type=str, default="/opt/ml/processing/output/forecast")
    parser.add_argument("--importance-output-dir", type=str, default="/opt/ml/processing/output/importance")
    parser.add_argument("--family", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    return parser.parse_args()


def require_file(path, what):
    """DEFENSIVE: fail loudly and specifically, rather than a cryptic downstream
    error, if a file from a prior step genuinely isn't where expected."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required {what} not found at {path!r}. This step depends on a prior "
            f"pipeline step's output being present here -- check that step actually "
            f"completed and wrote to the expected S3 location before this one ran."
        )
    return path


# ============================================================
# PART 1: Extract Scenario 1's evaluation.json (UNCHANGED)
# ============================================================
if __name__ == "__main__":
    args = parse_args()

    print(f"\n=== PART 1: Extracting Scenario 1 evaluation.json ({args.family}/{args.role}) ===")
    print(f"Checking for model.tar.gz...")
    print(f"Contents of {args.model_dir}:")
    if os.path.exists(args.model_dir):
        for root, dirs, files in os.walk(args.model_dir):
            for f in files:
                print(f"  {os.path.join(root, f)}")
    else:
        raise FileNotFoundError(
            f"Scenario 1 model directory {args.model_dir!r} does not exist -- the "
            f"training step's output was not correctly mounted for this Repack step."
        )

    possible_paths = [
        os.path.join(args.model_dir, "model.tar.gz"),
        "/opt/ml/processing/input/model.tar.gz",
        os.path.join(args.model_dir, "model", "model.tar.gz"),
    ]
    tar_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if not tar_path:
        raise FileNotFoundError(f"No model.tar.gz found in any of: {possible_paths}")
    print(f"Found model.tar.gz at: {tar_path}")

    with tarfile.open(tar_path, "r:gz") as tar:
        eval_member = next((m for m in tar.getmembers() if m.name.endswith("evaluation.json")), None)
        if not eval_member:
            raise FileNotFoundError(f"No evaluation.json in {tar_path}")
        eval_data = json.load(tar.extractfile(eval_member))

    os.makedirs(args.eval_output_dir, exist_ok=True)
    eval_out_path = os.path.join(args.eval_output_dir, f"{args.family}_{args.role}_evaluation.json")
    with open(eval_out_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Extracted evaluation.json from {tar_path}")
    print(f"Saved to {eval_out_path}")
    print(json.dumps(eval_data, indent=2))

    # ============================================================
    # PART 2: Train Scenario 2 -- full data through the real cutoff, no holdout --
    # using train_with_holdout.py's own functions for identical hyperparameters.
    # ============================================================
    print(f"\n=== PART 2: Training Scenario 2 ({args.family}/{args.role}) ===")
    require_file(args.train_data_dir, "preprocessed training data directory")
    parquet_files = [f for f in os.listdir(args.train_data_dir) if f.endswith(".parquet")]
    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet file found in {args.train_data_dir!r} -- PreprocessData's "
            f"output was not correctly mounted for this Repack step."
        )

    df = t1.load_data(args.train_data_dir)
    df = df[df["Family"] == args.family]
    features = t1.discover_features(df)
    model_name = t1.MODEL_SPEC[args.family][args.role]
    print(f"Loaded {len(df)} rows for {args.family}, {len(features)} features, model={model_name}")
    print(f"Training on FULL data through {df['date'].max().date()} (no holdout -- Scenario 2)")

    X_full = df[features].fillna(0)
    y_full = df[t1.TARGET]

    os.makedirs(args.model_output_dir, exist_ok=True)
    scaler = None
    if model_name == "Ridge":
        scaler = StandardScaler()
        X_full_scaled = scaler.fit_transform(X_full)
        model = t1.make_model(model_name, ridge_alpha=1.0)
        model.fit(X_full_scaled, y_full)
        t1.save_model(model, model_name, args.model_output_dir, args.family, args.role, scaler)
    else:
        model = t1.make_model(model_name, ridge_alpha=1.0)
        model.fit(X_full, y_full)
        t1.save_model(model, model_name, args.model_output_dir, args.family, args.role)
    print(f"Scenario 2 {model_name} trained on {len(df)} rows")

    # Same supporting artifacts (lookup CSVs) the live endpoint's model_fn() and this
    # pipeline's dashboard scripts expect -- identical to what train_with_holdout.py
    # already produces for Scenario 1, just built from the Scenario 2 model's own
    # full-data run so both are internally consistent.
    t1.build_supporting_artifacts(df, args.family, args.model_output_dir)
    with open(os.path.join(args.model_output_dir, "feature_list.json"), "w") as f:
        json.dump({"features": features, "production_cutoff": df["date"].max().strftime("%Y-%m-%d")}, f)
    if os.path.exists(req_file):
        import shutil
        shutil.copy(req_file, os.path.join(args.model_output_dir, "requirements.txt"))

    # Package as model.tar.gz -- SAME structure SageMaker expects for model
    # registration, and what the live endpoint's model_fn() already knows how to load.
    model_tar_path = os.path.join(args.model_output_dir, "model.tar.gz")
    with tarfile.open(model_tar_path, "w:gz") as tar:
        for fname in os.listdir(args.model_output_dir):
            if fname != "model.tar.gz":
                tar.add(os.path.join(args.model_output_dir, fname), arcname=fname)
    print(f"Packaged Scenario 2 model.tar.gz at {model_tar_path}")

    # ============================================================
    # PART 3: Generate this family/role's 2-year forecast chunk, using the SAME,
    # just-trained Scenario 2 model object -- not a reloaded copy, guaranteeing the
    # deployed model and the model that generated the forecast are identical.
    # ============================================================
    print(f"\n=== PART 3: Generating 2-year forecast chunk ({args.family}/{args.role}) ===")
    inf.FULL_FEATURES = features
    FORECAST_START = pd.Timestamp("2025-10-01")
    FORECAST_END = pd.Timestamp("2027-09-01")
    base_date = pd.Timestamp("2013-10-01")

    fh_all = df.groupby("FIPS").apply(
        lambda g: g.nlargest(24, "date")[["date", "Sales_QTY", "Returns_QTY"]]
    ).reset_index(level=0).rename(columns={"level_0": "FIPS"})
    national_hist = df.groupby("date")["Sales_QTY"].sum().reset_index()
    static_table = df.groupby("FIPS").agg({
        "County_Name": "first", "State": "first", "county_alltime_share": "first",
        "county_recent_share": "first", "state_share": "first",
        "county_active_months": "first", "county_is_sparse": "first",
    })
    ext_all = df[["FIPS", "date"] + [c for c in inf.get_external_feature_names(features) if c in df.columns]]
    bounds_df = df.groupby(["FIPS", "calendar_month"])["Sales_QTY"].max().reset_index()
    bounds_df.columns = ["FIPS", "calendar_month", "max_qty"]

    ext_cols = [c for c in ext_all.columns if c not in ("date", "FIPS") and
                (c in inf.FULL_FEATURES or f"{c}_lag1" in inf.FULL_FEATURES or f"{c}_chg3m" in inf.FULL_FEATURES)]

    all_results = []
    fips_list = fh_all["FIPS"].unique()

    # Genuine PER-COUNTY feature importance -- the actual customer ask, not the
    # global/sampled version built earlier. One SHAP explainer created ONCE here and
    # reused across every county (not recreated per county, which would be wasteful
    # at this scale -- confirmed identical output either way via direct testing).
    # Computed only for each county's FIRST forecast month (October 2025) -- computing
    # this for every month of every county would be prohibitively expensive; one real,
    # representative month per county is the same trade-off already made for the
    # global feature importance sample.
    shap_explainer = None
    if model_name in ("XGBoost", "LightGBM"):
        import shap
       
       
        shap_explainer = shap.TreeExplainer(
            model.booster if isinstance(model, (inf._NativeXGBWrapper, inf._NativeLGBWrapper)) else model
        )

   
    top_volume_fips = set(fh_all.groupby("FIPS")["Sales_QTY"].sum().nlargest(30).index)
    shap_sample_rows = []
    county_feature_importance_rows = []

    for fips in fips_list:
        fips_hist_this = fh_all[fh_all["FIPS"] == fips]
        if len(fips_hist_this) == 0:
            continue
        static_row = (static_table.loc[fips] if fips in static_table.index
                      else pd.Series({"county_alltime_share": 0.0, "county_recent_share": 0.0,
                                       "state_share": 0.0, "county_active_months": 0.0, "county_is_sparse": 1}))
        county_name = static_row.get("County_Name", "")
        state = static_row.get("State", "")
        ext_fips = ext_all[ext_all["FIPS"] == fips]
        ext_fips = inf._extend_series(ext_fips, ext_cols, FORECAST_END) if len(ext_fips) else ext_fips

        work = fips_hist_this[["date", "Sales_QTY", "Returns_QTY"]].copy()
        cur = work["date"].max()
        while cur < FORECAST_END:
            nxt = (cur + pd.DateOffset(months=1)).replace(day=1)
            feat_row = inf._build_feature_row(work, national_hist, ext_fips, static_row, nxt, base_date)
            x = feat_row[features].fillna(0)
            x_input = scaler.transform(x) if scaler is not None else x
           
            x_input_df = pd.DataFrame(x_input, columns=features, index=x.index) if scaler is not None else x_input
            pred_raw = float(model.predict(x_input)[0])
            lo, hi = inf._seasonal_bounds_for_fips(bounds_df, fips, nxt.month)
            pred = float(np.clip(pred_raw, lo, hi))
            work = pd.concat([work, pd.DataFrame([{"date": nxt, "Sales_QTY": pred, "Returns_QTY": 0.0}])], ignore_index=True)
            if nxt >= FORECAST_START:
                row = {
                    "FIPS": fips, "County_Name": county_name, "State": state,
                    "date": nxt.strftime("%Y-%m-%d"), "Predicted_QTY": pred,
                    "Family": args.family, "ModelRole": args.role.capitalize(), "ModelName": model_name,
                    "External_Factors_Share_Pct": 0,
                    "Top1_Feature": "", "Top1_SHAP": 0, "Top2_Feature": "", "Top2_SHAP": 0,
                    "Top3_Feature": "", "Top3_SHAP": 0, "Top4_Feature": "", "Top4_SHAP": 0,
                    "Top5_Feature": "", "Top5_SHAP": 0,
                }
                if nxt == FORECAST_START and model_name in ("XGBoost", "LightGBM", "Ridge"):
                    
                    all_features_ranked, ext_share = inf.compute_per_row_shap_reusing_explainer(
                        shap_explainer, model, model_name, x_input_df, features
                    )
                    if all_features_ranked:
                        row["External_Factors_Share_Pct"] = ext_share
                        
                        for i, feat_entry in enumerate(all_features_ranked[:5], start=1):
                            row[f"Top{i}_Feature"] = feat_entry["feature"]
                            row[f"Top{i}_SHAP"] = feat_entry["shap_value"]
                        for rank, feat_entry in enumerate(all_features_ranked, start=1):
                            county_feature_importance_rows.append({
                                "Level": "County", "FIPS": fips, "County_Name": county_name,
                                "State": state, "Family": args.family,
                                "ModelRole": args.role.capitalize(), "ModelName": model_name,
                                "Feature": feat_entry["feature"],
                                "Friendly_Name": inf.friendly_feature_name(feat_entry["feature"]),
                                "SHAP_Value": feat_entry["shap_value"], "Rank": rank,
                            })
                    
                    if fips in top_volume_fips:
                        shap_sample_rows.append(x)
                all_results.append(row)
            cur = nxt

    os.makedirs(args.forecast_output_dir, exist_ok=True)
    forecast_df = pd.DataFrame(all_results)
    forecast_path = os.path.join(args.forecast_output_dir, f"forecast_2yr_{args.family.lower()}_{args.role}.csv")
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Saved {len(forecast_df)} forecast rows to {forecast_path}")


    # ============================================================
    print(f"\n=== PART 4: Generating feature importance ({args.family}/{args.role}) ===")
    combined_importance_rows = []

    if shap_sample_rows:
        sample_df = pd.concat(shap_sample_rows, ignore_index=True)
        importance = inf.compute_global_feature_importance(model, model_name, sample_df, features)
        for r in importance:
            combined_importance_rows.append({
                "Level": "National", "FIPS": None, "County_Name": None, "State": None,
                "Family": args.family, "ModelRole": args.role.capitalize(), "ModelName": model_name,
                "Feature": r["feature"], "Friendly_Name": r["friendly_name"],
                "SHAP_Value": r["mean_abs_shap"], "Is_External": r["is_external"], "Rank": None,
            })
    else:
        print("WARNING: no top-volume-county sample rows collected -- skipping National level.")

    for r in county_feature_importance_rows:
        r["Is_External"] = True  # county-level rows are always external-only, by construction
        combined_importance_rows.append(r)

    importance_df = pd.DataFrame(combined_importance_rows, columns=[
        "Level", "FIPS", "County_Name", "State", "Family", "ModelRole", "ModelName",
        "Feature", "Friendly_Name", "SHAP_Value", "Is_External", "Rank",
    ])
    os.makedirs(args.importance_output_dir, exist_ok=True)
    importance_path = os.path.join(args.importance_output_dir, f"feature_importance_{args.family.lower()}_{args.role}.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved {len(importance_df)} feature importance rows "
          f"({(importance_df['Level']=='National').sum()} National, "
          f"{(importance_df['Level']=='County').sum()} County) to {importance_path}")

    print(f"\n\u2713 Repack complete for {args.family}/{args.role}: "
          f"evaluation extracted, Scenario 2 trained ({len(df)} rows), "
          f"forecast generated ({len(forecast_df)} rows).")

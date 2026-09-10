"""Generate 2-year forecast for all models"""
import subprocess
import sys
import os

req_file = '/opt/ml/processing/input/requirements/requirements.txt'
if os.path.exists(req_file):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", req_file])

import json
import pickle
import tarfile
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import boto3
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference as inf

INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"

MODEL_SPEC = {
    'Barrage': {'champion': 'XGBoost', 'challenger': 'Ridge'},
    'Grounded': {'champion': 'LightGBM', 'challenger': 'Ridge'}
}

MAX_WORKERS = 10
_worker_model = None
_worker_scaler = None

def _init_worker(extract_dir, family, role):
    global _worker_model, _worker_scaler
    sys.path.insert(0, '/opt/ml/processing/code')
    model_name = MODEL_SPEC[family][role]
    prefix = f"{role}_{family}_{model_name}"
    if model_name == "XGBoost":
        import inference as inf_worker
        booster = xgb.Booster()
        booster.load_model(os.path.join(extract_dir, f"{prefix}.ubj"))
        _worker_model, _worker_scaler = inf_worker._NativeXGBWrapper(booster), None
    elif model_name == "LightGBM":
        import inference as inf_worker
        booster = lgb.Booster(model_file=os.path.join(extract_dir, f"{prefix}.txt"))
        _worker_model, _worker_scaler = inf_worker._NativeLGBWrapper(booster), None
    else:
        with open(os.path.join(extract_dir, f"{prefix}.pkl"), 'rb') as f:
            _worker_model = pickle.load(f)
        scaler_path = os.path.join(extract_dir, f"ridge_scaler_{family}.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                _worker_scaler = pickle.load(f)

def _forecast_one_county(args):
    (fips, county_name, state, fips_hist_this, national_hist, ext_fips, static_row,
     bounds_df, full_features, forecast_start, forecast_end, base_date,
     carry_through_cols, family, role, model_name, is_shap_sample_county) = args
    sys.path.insert(0, '/opt/ml/processing/code')
    import inference as inf_worker
    inf_worker.FULL_FEATURES = full_features
    work = fips_hist_this[["date", "Sales_QTY", "Returns_QTY"]].copy()
    cur = work["date"].max()
    results = []
    shap_feat_row = None
    while cur < forecast_end:
        nxt = (cur + pd.DateOffset(months=1)).replace(day=1)
        feat_row = inf_worker._build_feature_row(work, national_hist, ext_fips, static_row, nxt, base_date)
        if is_shap_sample_county and nxt == forecast_start:
            shap_feat_row = feat_row
        x = feat_row[full_features].fillna(0)
        x_input = _worker_scaler.transform(x) if _worker_scaler is not None else x
        pred_raw = float(_worker_model.predict(x_input)[0])
        lo, hi = inf_worker._seasonal_bounds_for_fips(bounds_df, fips, nxt.month)
        pred = float(np.clip(pred_raw, lo, hi))
        work = pd.concat([work, pd.DataFrame([{"date": nxt, "Sales_QTY": pred, "Returns_QTY": 0.0}])], ignore_index=True)
        if nxt >= forecast_start:
            row = {
                'FIPS': fips, 'County_Name': county_name, 'State': state,
                'date': nxt.strftime('%Y-%m-%d'), 'Predicted_QTY': pred,
                'Family': family, 'ModelRole': role.capitalize(), 'ModelName': model_name,
                'External_Factors_Share_Pct': 0,
                'Top1_Feature': '', 'Top1_SHAP': 0, 'Top2_Feature': '', 'Top2_SHAP': 0,
                'Top3_Feature': '', 'Top3_SHAP': 0, 'Top4_Feature': '', 'Top4_SHAP': 0,
                'Top5_Feature': '', 'Top5_SHAP': 0,
            }
            for c in carry_through_cols:
                row[c] = float(feat_row[c].iloc[0]) if c in feat_row.columns and pd.notna(feat_row[c].iloc[0]) else None
            results.append(row)
        cur = nxt
    return fips, results, shap_feat_row

def extract_model(model_dir, family, role):
    tar_path = os.path.join(model_dir, 'model.tar.gz')
    extract_dir = os.path.join(model_dir, 'extracted')
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    model_name = MODEL_SPEC[family][role]
    prefix = f"{role}_{family}_{model_name}"
    if model_name == "XGBoost":
        booster = xgb.Booster()
        booster.load_model(os.path.join(extract_dir, f"{prefix}.ubj"))
        return inf._NativeXGBWrapper(booster), None, extract_dir
    elif model_name == "LightGBM":
        booster = lgb.Booster(model_file=os.path.join(extract_dir, f"{prefix}.txt"))
        return inf._NativeLGBWrapper(booster), None, extract_dir
    else:
        with open(os.path.join(extract_dir, f"{prefix}.pkl"), 'rb') as f:
            model = pickle.load(f)
        scaler_path = os.path.join(extract_dir, f"ridge_scaler_{family}.pkl")
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        return model, scaler, extract_dir

FORECAST_START = pd.Timestamp("2025-10-01")
FORECAST_END = pd.Timestamp("2027-09-01")
CARRY_THROUGH_FEATURE_COLS = [
    'TAVG', 'TMAX', 'TMIN', 'PRCP', 'GDD_proxy', 'CHU_proxy', 'Precip_Anomaly',
    'CPI_AllUrban', 'Fertilizer_PPI', 'Cotton_Price_USCentsPerLb', 'Corn_Price_USDPerMT',
    'Soybeans_Price_USDPerMT', 'Wheat_Price_USDPerMT', 'WTI_Oil_Price',
]

s3 = boto3.client('s3')
all_results = []
global_importance_results = []

if __name__ == "__main__":
    for family in ['Barrage', 'Grounded']:
        champion_dir = os.path.join(INPUT_DIR, 'models', f"{family.lower()}-champion")
        challenger_dir = os.path.join(INPUT_DIR, 'models', f"{family.lower()}-challenger")
        _, _, champion_extract_dir = extract_model(champion_dir, family, 'champion')
        _, _, challenger_extract_dir = extract_model(challenger_dir, family, 'challenger')
        model_dirs = {'champion': champion_extract_dir, 'challenger': challenger_extract_dir}
        with open(os.path.join(champion_extract_dir, 'feature_list.json')) as f:
            inf.FULL_FEATURES = json.load(f)['features']
        fh_all = pd.read_csv(os.path.join(champion_extract_dir, f"fips_trailing_history_{family.lower()}.csv"), parse_dates=["date"], dtype={"FIPS": str})
        fh_all["FIPS"] = fh_all["FIPS"].str.zfill(5)
        national_hist = pd.read_csv(os.path.join(champion_extract_dir, f"national_trailing_{family.lower()}.csv"), parse_dates=["date"])
        static_table = pd.read_csv(os.path.join(champion_extract_dir, f"fips_static_features_{family.lower()}.csv"), dtype={"FIPS": str})
        static_table["FIPS"] = static_table["FIPS"].str.zfill(5)
        static_table = static_table.set_index("FIPS")
        ext_all = pd.read_csv(os.path.join(champion_extract_dir, f"fips_external_weather_{family.lower()}.csv"), parse_dates=["date"], dtype={"FIPS": str})
        ext_all["FIPS"] = ext_all["FIPS"].str.zfill(5)
        bounds_df = pd.read_csv(os.path.join(champion_extract_dir, f"fips_guardrail_bounds_{family.lower()}.csv"), dtype={"FIPS": str})
        bounds_df["FIPS"] = bounds_df["FIPS"].str.zfill(5)
        ext_cols = [c for c in ext_all.columns if c not in ("date", "FIPS") and (c in inf.FULL_FEATURES or f"{c}_lag1" in inf.FULL_FEATURES or f"{c}_chg3m" in inf.FULL_FEATURES)]
        fips_list = fh_all["FIPS"].unique()
        base_date = pd.Timestamp("2013-10-01")
        top_volume_fips = set(fh_all.groupby("FIPS")["Sales_QTY"].sum().nlargest(30).index)
        for role in ['champion', 'challenger']:
            print(f"Generating {family}/{role}...")
            model_name = MODEL_SPEC[family][role]
            extract_dir = model_dirs[role]
            tasks = []
            for fips in fips_list:
                fips_hist_this = fh_all[fh_all["FIPS"] == fips]
                if len(fips_hist_this) == 0:
                    continue
                static_row = (static_table.loc[fips] if fips in static_table.index else pd.Series({"county_alltime_share": 0.0, "county_recent_share": 0.0, "state_share": 0.0, "county_active_months": 0.0, "county_is_sparse": 1}))
                county_name = static_row.get("County_Name", "")
                state = static_row.get("State", "")
                ext_fips = ext_all[ext_all["FIPS"] == fips]
                ext_fips = inf._extend_series(ext_fips, ext_cols, FORECAST_END) if len(ext_fips) else ext_fips
                bounds_df_this = bounds_df[bounds_df["FIPS"] == fips]
                tasks.append((fips, county_name, state, fips_hist_this, national_hist, ext_fips, static_row, bounds_df_this, inf.FULL_FEATURES, FORECAST_START, FORECAST_END, base_date, CARRY_THROUGH_FEATURE_COLS, family, role, model_name, fips in top_volume_fips))
            shap_sample_rows = []
            with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker, initargs=(extract_dir, family, role)) as executor:
                futures = [executor.submit(_forecast_one_county, t) for t in tasks]
                for future in as_completed(futures):
                    fips, county_results, shap_feat_row = future.result()
                    all_results.extend(county_results)
                    if shap_feat_row is not None:
                        shap_sample_rows.append(shap_feat_row)
            if shap_sample_rows:
                if model_name == "XGBoost":
                    booster = xgb.Booster()
                    booster.load_model(os.path.join(extract_dir, f"{role}_{family}_{model_name}.ubj"))
                    shap_model = inf._NativeXGBWrapper(booster)
                elif model_name == "LightGBM":
                    booster = lgb.Booster(model_file=os.path.join(extract_dir, f"{role}_{family}_{model_name}.txt"))
                    shap_model = inf._NativeLGBWrapper(booster)
                else:
                    with open(os.path.join(extract_dir, f"{role}_{family}_{model_name}.pkl"), 'rb') as f:
                        shap_model = pickle.load(f)
                sample_df = pd.concat(shap_sample_rows, ignore_index=True)
                importance = inf.compute_global_feature_importance(shap_model, model_name, sample_df, inf.FULL_FEATURES)
                for r in importance:
                    global_importance_results.append({"Family": family, "ModelRole": role.capitalize(), "ModelName": model_name, "Feature": r["feature"], "Friendly_Name": r["friendly_name"], "Mean_Abs_SHAP": r["mean_abs_shap"], "Is_External": r["is_external"]})
    df_out = pd.DataFrame(all_results)
    forecast_path = '/tmp/forecast_2yr_all_models.csv'
    df_out.to_csv(forecast_path, index=False)
    s3.upload_file(forecast_path, os.environ.get("VAP_S3_BUCKET", "vap-sales-forecasting"), 'comparison/forecast_2yr_all_models.csv')
    s3.upload_file(forecast_path, os.environ.get("VAP_S3_BUCKET", "vap-sales-forecasting"), 'dashboard/forecast_2yr_all_models.csv')
    print(f"Uploaded {len(df_out)} rows")
    global_importance_df = pd.DataFrame(global_importance_results)
    global_importance_path = '/tmp/feature_importance_global.csv'
    global_importance_df.to_csv(global_importance_path, index=False)
    s3.upload_file(global_importance_path, os.environ.get("VAP_S3_BUCKET", "vap-sales-forecasting"), 'dashboard/feature_importance_global.csv')
    print(f"Uploaded {len(global_importance_df)} rows to feature_importance_global.csv")
    subprocess.check_call([sys.executable, '/opt/ml/processing/code/dashboard/dashboard_merge.py'])
    subprocess.check_call([sys.executable, '/opt/ml/processing/code/dashboard/generate_county_accuracy.py'])
    print("✓ Complete!")

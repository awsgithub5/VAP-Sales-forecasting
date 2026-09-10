"""
inference.py -- Scenario 2 forecast/what-if endpoint.

KEY DESIGN CHOICE: one hit returns BOTH Champion and Challenger predictions together.

Supports exactly what the 3 downstream agents need:
  - Forecasting agent: {"family", "fips", "month", "year"} -> both models' predictions
  - What-If agent: same + {"scenario": true, "feature_changes": {...}} -> both models'
    predictions under the hypothetical, plus which feature_changes were actually applied
  - Comparison agent: NOT served by this endpoint -- per the client's instruction, the
    Comparison agent uses Scenario 1's already-computed forecast (scenario1_results.json,
    delivered separately) as its actual-vs-predicted baseline, since that's the run with
    real actuals to compare against. This endpoint only knows Scenario 2 (forecast into
    the unknown future, Oct-2025 onward), which has no actuals to compare to yet.

Request JSON:
    {
      "family": "Barrage" | "Grounded",
      "fips": "04013",
      "month": 9,
      "year": 2025,
      "scenario": false,            (optional)
      "feature_changes": {...}      (optional, only used if scenario=true)
    }

Response JSON:
    {
      "family": "Barrage", "fips": "04013", "month": 3, "year": 2026,
      "feature_source": "historical_actual" | "model_forecast_recursive_guardrailed" | "scenario_override",
      "champion": {"model_name": "XGBoost", "predicted_qty": 116.2},
      "challenger": {"model_name": "HistGradientBoosting", "predicted_qty": 108.9},
      "scenario_features": {...},              (only present when scenario=true)
      "unsupported_feature_changes": [...]     (only present when scenario=true and some
                                                 requested changes aren't real model features)
    }
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import shap


class _NativeXGBWrapper:
    """Wraps an xgb.Booster (loaded from a native .ubj file) so it exposes the same
    .predict(dataframe) -> array interface the rest of this module expects, matching
    the sklearn-style XGBRegressor.predict() calling convention used elsewhere."""
    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        return self.booster.predict(xgb.DMatrix(X))


class _NativeLGBWrapper:
    """Wraps a lgb.Booster (loaded from a native .txt file) the same way. LightGBM's
    raw Booster.predict already accepts a DataFrame directly, unlike XGBoost's."""
    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        return self.booster.predict(X)

# built before train.py started saving it). Any current bundle overrides this at model_fn()
# time with the EXACT feature list the model was actually trained on, read from
# feature_list.json -- this is what keeps inference.py in sync with train.py's dynamic,
# lean feature discovery instead of silently drifting out of date every time a feature is
# added or removed (as happened here: this hardcoded list was missing 7 features -- WTI oil
# price and 4 crop acreage columns -- that a real training run actually used).
FULL_FEATURES = [
    'time_idx', 'calendar_month', 'quarter', 'fiscal_month_idx',
    'month_sin', 'month_cos', 'fiscal_sin', 'fiscal_cos',
    'growing_season_proxy', 'is_planting_season', 'is_application_season',
    'is_harvest_season', 'is_dormant_season',
    'fips_lag_1', 'fips_lag_2', 'fips_lag_3', 'fips_lag_6', 'fips_lag_12',
    'fips_rollmean_3', 'fips_rollstd_3', 'fips_rollmean_6', 'fips_rollstd_6',
    'fips_rollmean_12', 'fips_rollstd_12', 'returns_lag_1',
    'county_alltime_share', 'county_recent_share', 'state_share',
    'county_active_months', 'county_is_sparse',
    'national_lag_1', 'national_lag_12', 'national_rollmean_3',
    'CPI_AllUrban', 'Fertilizer_PPI', 'Cotton_Price_USCentsPerLb', 'Corn_Price_USDPerMT',
    'Soybeans_Price_USDPerMT', 'Wheat_Price_USDPerMT',
    'CPI_AllUrban_lag1', 'CPI_AllUrban_chg3m', 'Fertilizer_PPI_lag1', 'Fertilizer_PPI_chg3m',
    'Cotton_Price_USCentsPerLb_lag1', 'Cotton_Price_USCentsPerLb_chg3m',
    'Corn_Price_USDPerMT_lag1', 'Corn_Price_USDPerMT_chg3m',
    'Soybeans_Price_USDPerMT_lag1', 'Soybeans_Price_USDPerMT_chg3m',
    'Wheat_Price_USDPerMT_lag1', 'Wheat_Price_USDPerMT_chg3m',
    'PRCP', 'TAVG', 'TMAX', 'TMIN', 'GDD_proxy', 'CHU_proxy', 'Precip_Anomaly', 'has_temperature_data',
]

# What-If feature-name -> real model-feature mapping. NOTE: WTI_Oil_Price is NOT in this
# model's feature set (the fetch didn't succeed when this file was built -- see the
# Feature Dictionary doc) -- any oil-price what-if request is genuinely unsupported here,
# not silently ignored; it will show up in unsupported_feature_changes.
FEATURE_CHANGE_MAP = {
    "CPIAUCSL": "CPI_AllUrban", "CPI": "CPI_AllUrban",
    "Fertilizer_NH3": "Fertilizer_PPI", "FertilizerPrice": "Fertilizer_PPI",
    "Cotton": "Cotton_Price_USCentsPerLb", "Corn": "Corn_Price_USDPerMT",
    "Soybeans": "Soybeans_Price_USDPerMT", "Wheat": "Wheat_Price_USDPerMT",
    # Oil price -- genuinely added to the model since the original absent-data caveat was
    # written; real data, no longer unsupported.
    "DCOILWTICO": "WTI_Oil_Price", "CrudeOil_WTI": "WTI_Oil_Price",
    "WTI": "WTI_Oil_Price", "OilPrice": "WTI_Oil_Price",
    # Crop ACREAGE -- also genuinely added since then (distinct from crop PRICE above).
    "CORN_Acreage": "CORN_Acres_prior_year", "SOYBEANS_Acreage": "SOYBEANS_Acres_prior_year",
    "WHEAT_Acreage": "WHEAT_Acres_prior_year", "COTTON_Acreage": "COTTON_Acres_prior_year",
}
KNOWN_UNSUPPORTED = {
    "GDD", "CHU", "HeatIndex", "PrecipAnomaly",  
    "CrudeOil_Brent", 
}
FEATURE_CHANGE_MAP.update({
    "GDD_actual": "GDD_proxy", "CHU_actual": "CHU_proxy", "PrecipAnomaly_actual": "Precip_Anomaly",
})

LAST_ACTUAL_DATE = pd.Timestamp("2025-09-01")


def get_external_feature_names(all_features):
    """Returns external features (weather, economic, crop acreage) by exclusion pattern.
    Excludes calendar/season, FIPS lags/rolling, national aggregates, county static shares.
    This is the single source of truth for what counts as 'external' -- used by both
    build_supporting_artifacts() and SHAP filtering in Changes 3/4.
    NOTE: This function is duplicated verbatim in train.py -- if you modify this logic,
    update both files identically to keep them in sync."""
    NON_EXTERNAL_PREFIXES = ("fips_lag", "fips_roll", "national_", "returns_lag",
                             "time_idx", "calendar_month", "quarter", "month_",
                             "fiscal_", "growing_season", "is_planting", "is_application",
                             "is_harvest", "is_dormant")
    NON_EXTERNAL_EXACT = {"county_alltime_share", "county_recent_share", "state_share",
                          "county_active_months", "county_is_sparse"}
    return [c for c in all_features if c not in NON_EXTERNAL_EXACT
            and not c.startswith(NON_EXTERNAL_PREFIXES)]


def get_display_external_feature_names(all_features):
    """Same as get_external_feature_names(), with one further exclusion: the _lag1/_chg3m
    DERIVATIVE columns (e.g. CPI_AllUrban_lag1, Fertilizer_PPI_chg3m). The model genuinely
    uses these internally -- get_external_feature_names() (above) is UNCHANGED and still
    includes them, since that function feeds real model logic (_extend_series' ext_cols,
    external_factors_share_pct). This second, display-only filter is for anything a human
    will actually SEE on a dashboard graph -- showing "CPI_AllUrban", "CPI_AllUrban_lag1",
    AND "CPI_AllUrban_chg3m" as three separate "features" would just be confusing, since
    they're three derived views of the one real-world concept a client cares about."""
    base = get_external_feature_names(all_features)
    return [c for c in base if not c.endswith("_lag1") and not c.endswith("_chg3m")]


# Human-readable display names for QuickSight -- "TAVG" and "GDD_proxy" mean nothing to a
# client; this is the single place that translation happens, used by both the global
# feature-importance graphs and anywhere else a feature name reaches a dashboard viewer.
# Only covers names that actually need translating; anything not listed here is shown as-is
# (falls back to the raw column name), so adding a new feature never breaks this lookup.
FRIENDLY_FEATURE_NAMES = {
    # Weather
    "TAVG": "Average Temperature",
    "TMAX": "Maximum Temperature",
    "TMIN": "Minimum Temperature",
    "PRCP": "Precipitation",
    "GDD_proxy": "Growing Degree Days",
    "CHU_proxy": "Heat Units",
    "Precip_Anomaly": "Precipitation Anomaly",
    "has_temperature_data": "Weather Data Availability",
    # Crop acreage
    "CORN_Acres_prior_year": "Corn Acreage",
    "SOYBEANS_Acres_prior_year": "Soybean Acreage",
    "WHEAT_Acres_prior_year": "Wheat Acreage",
    "COTTON_Acres_prior_year": "Cotton Acreage",
    # Crop prices
    "Corn_Price_USDPerMT": "Corn Price",
    "Soybeans_Price_USDPerMT": "Soybean Price",
    "Wheat_Price_USDPerMT": "Wheat Price",
    "Cotton_Price_USCentsPerLb": "Cotton Price",
    # Economic indicators
    "CPI_AllUrban": "Consumer Price Index",
    "Fertilizer_PPI": "Fertilizer Price",
    "WTI_Oil_Price": "Oil Price (WTI)",
    # Internal/sales-history features -- included so "feature importance all" reads
    # sensibly too, not just the external-only graph.
    "fips_lag_1": "Sales 1 Month Ago",
    "fips_lag_2": "Sales 2 Months Ago",
    "fips_lag_3": "Sales 3 Months Ago",
    "fips_lag_6": "Sales 6 Months Ago",
    "fips_lag_12": "Sales 12 Months Ago",
    "fips_rollmean_3": "3-Month Average Sales",
    "fips_rollmean_6": "6-Month Average Sales",
    "fips_rollmean_12": "12-Month Average Sales",
    "fips_rollstd_3": "3-Month Sales Volatility",
    "fips_rollstd_6": "6-Month Sales Volatility",
    "fips_rollstd_12": "12-Month Sales Volatility",
    "returns_lag_1": "Returns 1 Month Ago",
    "national_lag_1": "National Sales 1 Month Ago",
    "national_lag_12": "National Sales 12 Months Ago",
    "national_rollmean_3": "3-Month National Average Sales",
    "county_alltime_share": "County Share of All-Time Sales",
    "county_recent_share": "County Share of Recent Sales",
    "state_share": "County Share of State Sales",
    "county_active_months": "Months County Has Been Active",
    "county_is_sparse": "Low-Volume County Flag",
}


def friendly_feature_name(feature: str) -> str:
    """Look up a feature's client-facing display name, falling back to the raw column
    name for anything not in FRIENDLY_FEATURE_NAMES (e.g. a newly-added feature) rather
    than failing or showing a blank."""
    return FRIENDLY_FEATURE_NAMES.get(feature, feature)


def normalize_fips(fips) -> str:
    if fips is None:
        raise ValueError("FIPS is required.")
    value = str(fips).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(5)


def resolve_fips(payload, static_table):
    """Resolve FIPS from either direct fips or county_name + state lookup.
    FIPS takes precedence if both are provided."""
    if "fips" in payload and payload["fips"]:
        return normalize_fips(payload["fips"])
    
    county_name = payload.get("county_name")
    state = payload.get("state")
    
    if not county_name or not state:
        raise ValueError("Either 'fips' or both 'county_name' and 'state' must be provided")
    
    # Lookup in static table
    if "County_Name" not in static_table.columns or "State" not in static_table.columns:
        raise ValueError("County name lookup not supported - static features missing County_Name/State columns")
    
    match = static_table[
        (static_table["County_Name"].str.lower() == county_name.lower()) &
        (static_table["State"].str.upper() == state.upper())
    ]
    
    if match.empty:
        raise ValueError(f"County '{county_name}' in state '{state}' not found")
    
    return match.index[0]


def _calendar_feats(date: pd.Timestamp) -> dict:
    m = date.month
    fiscal_month_idx = ((m - 10) % 12) + 1
    return {
        "calendar_month": m, "quarter": date.quarter, "fiscal_month_idx": fiscal_month_idx,
        "month_sin": np.sin(2 * np.pi * m / 12), "month_cos": np.cos(2 * np.pi * m / 12),
        "fiscal_sin": np.sin(2 * np.pi * fiscal_month_idx / 12),
        "fiscal_cos": np.cos(2 * np.pi * fiscal_month_idx / 12),
        "growing_season_proxy": max(np.sin(np.pi * (m - 3) / 7), 0),
        "is_planting_season": int(m in [3, 4, 5]), "is_application_season": int(m in [4, 5, 6, 7, 8]),
        "is_harvest_season": int(m in [9, 10, 11]), "is_dormant_season": int(m in [12, 1, 2]),
    }


def _extend_series(series_df: pd.DataFrame, cols: list, target_date: pd.Timestamp) -> pd.DataFrame:
    """Seasonal-average carry-forward for external/weather columns beyond the bundled
    real data -- only used if a request reaches further than the real data goes.

    BUG FIX: the previous version used a "momentum" approach -- computing a single
    trend ratio from only the last 2 real data points, then applying it repeatedly
    (compounding) to every subsequent future month, with NO regard for calendar
    month. This ignored seasonality entirely: a weather/economic feature for July
    was extended identically to one for December, and critically, whenever the
    prior real value was 0 the ratio defaulted to exactly 1.0, meaning every future
    month received the EXACT SAME value with zero variation at all -- a direct,
    confirmed cause of flat predictions for far-future months, since a model
    trained on real seasonal patterns sees an input pattern with no seasonality
    left in it.

    Fixed to compute a genuine seasonal average instead: for each future month
    being extended, use the historical average of that SAME CALENDAR MONTH across
    all real years available (e.g. every real July on record), so a future July
    gets a value that actually reflects what July looks like, not a flat
    carried-forward number or a same trend line running through every month
    regardless of season. Falls back to simple carry-forward only if a column has
    fewer than 2 real historical values for that specific calendar month (not
    enough history to average meaningfully) -- never fabricated from nothing.
    """
    series_df = series_df.sort_values("date").reset_index(drop=True)
    last_date = series_df["date"].max()
    if target_date <= last_date:
        return series_df
    extended = series_df.copy()
    cur = last_date
    while cur < target_date:
        nxt = (cur + pd.DateOffset(months=1)).replace(day=1)
        new_row = {"date": nxt}
        for col in cols:
            # Only real, historical rows (not ones we've already synthetically
            # extended in a prior loop iteration) count toward the seasonal
            # average -- otherwise a long-range forecast would start averaging
            # its own earlier guesses together.
            same_month_real = series_df[
                (series_df["date"].dt.month == nxt.month) & series_df[col].notna()
            ][col]
            if len(same_month_real) >= 2:
                new_row[col] = float(same_month_real.mean())
            else:
                valid = extended[col].dropna()
                if len(valid) >= 1:
                    new_row[col] = float(valid.iloc[-1])
                else:
                    new_row[col] = np.nan
        extended = pd.concat([extended, pd.DataFrame([new_row])], ignore_index=True)
        cur = nxt
    return extended


def _build_feature_row(fips_hist: pd.DataFrame, national_hist: pd.DataFrame, ext_fips: pd.DataFrame,
                        static_row: pd.Series, target_date: pd.Timestamp, base_date: pd.Timestamp) -> pd.DataFrame:
    fh = fips_hist.sort_values("date").reset_index(drop=True)
    row = {"date": target_date}
    row.update(_calendar_feats(target_date))
    row["time_idx"] = (target_date - base_date).days // 30

    sales_series = fh["Sales_QTY"]
    for lag in [1, 2, 3, 6, 12]:
        row[f"fips_lag_{lag}"] = sales_series.iloc[-lag] if len(sales_series) >= lag else 0.0
    for w in [3, 6, 12]:
        window = sales_series.iloc[-w:] if len(sales_series) >= 1 else pd.Series(dtype=float)
        row[f"fips_rollmean_{w}"] = window.mean() if len(window) else 0.0
        row[f"fips_rollstd_{w}"] = window.std() if len(window) > 1 else 0.0
    row["returns_lag_1"] = fh["Returns_QTY"].iloc[-1] if len(fh) else 0.0

    row["county_alltime_share"] = float(static_row.get("county_alltime_share", 0.0))
    row["county_recent_share"] = float(static_row.get("county_recent_share", 0.0))
    row["state_share"] = float(static_row.get("state_share", 0.0))
    row["county_active_months"] = float(static_row.get("county_active_months", 0.0))
    row["county_is_sparse"] = int(static_row.get("county_is_sparse", 1))

    nat = national_hist.sort_values("date").reset_index(drop=True)
    nat_series = nat["Sales_QTY"]
    row["national_lag_1"] = nat_series.iloc[-1] if len(nat_series) >= 1 else 0.0
    row["national_lag_12"] = nat_series.iloc[-12] if len(nat_series) >= 12 else (nat_series.iloc[0] if len(nat_series) else 0.0)
    window3 = nat_series.iloc[-3:] if len(nat_series) else pd.Series(dtype=float)
    row["national_rollmean_3"] = window3.mean() if len(window3) else 0.0

    ext_cols = [c for c in FULL_FEATURES if c in ext_fips.columns]
    ext_row = ext_fips[ext_fips["date"] == target_date]
    for col in ext_cols:
        row[col] = ext_row[col].values[0] if len(ext_row) and col in ext_row.columns else np.nan

    econ_cols = [c for c in ext_cols if f"{c}_lag1" in FULL_FEATURES or f"{c}_chg3m" in FULL_FEATURES]
    for col in econ_cols:
        prev1 = ext_fips[ext_fips["date"] == target_date - pd.DateOffset(months=1)]
        prev3 = ext_fips[ext_fips["date"] == target_date - pd.DateOffset(months=3)]
        row[f"{col}_lag1"] = prev1[col].values[0] if len(prev1) and col in prev1.columns else row.get(col, np.nan)
        row[f"{col}_chg3m"] = (row[col] - prev3[col].values[0]) if len(prev3) and col in prev3.columns and pd.notna(row.get(col)) else 0.0

    return pd.DataFrame([row])


def _seasonal_bounds_for_fips(bounds_df: pd.DataFrame, fips: str, month: int):
    row = bounds_df[(bounds_df["FIPS"] == fips) & (bounds_df["calendar_month"] == month)]
    if row.empty or row.iloc[0]["max_qty"] == 0:
        return 0.0, np.inf
    return 0.0, float(row.iloc[0]["max_qty"]) * 1.15


def _forecast_one_model(model, fips_hist: pd.DataFrame, national_hist: pd.DataFrame, ext_fips: pd.DataFrame,
                         static_row: pd.Series, bounds_df: pd.DataFrame, fips: str,
                         target_date: pd.Timestamp, base_date: pd.Timestamp, model_scaler=None):
    exact = fips_hist[fips_hist["date"] == target_date]
    if not exact.empty and target_date <= LAST_ACTUAL_DATE:
        return float(exact.iloc[0]["Sales_QTY"]), "historical_actual"

    work = fips_hist[["date", "Sales_QTY", "Returns_QTY"]].copy()
    cur = work["date"].max()
    last_pred = None
    while cur < target_date:
        nxt = (cur + pd.DateOffset(months=1)).replace(day=1)
        feat_row = _build_feature_row(work, national_hist, ext_fips, static_row, nxt, base_date)
        x = feat_row[FULL_FEATURES].fillna(0)
        x_input = model_scaler.transform(x) if model_scaler is not None else x
        pred_raw = float(model.predict(x_input)[0])
        lo, hi = _seasonal_bounds_for_fips(bounds_df, fips, nxt.month)
        pred = float(np.clip(pred_raw, lo, hi))
        work = pd.concat([work, pd.DataFrame([{"date": nxt, "Sales_QTY": pred, "Returns_QTY": 0.0}])], ignore_index=True)
        last_pred = pred
        cur = nxt
    return last_pred, "model_forecast_recursive_guardrailed"


def _apply_scenario_one_model(model, fips_hist, national_hist, ext_fips, static_row, bounds_df, fips,
                               target_date, base_date, feature_changes, model_scaler=None, sensitivity_model=None, sensitivity_scaler=None):
    work = fips_hist[["date", "Sales_QTY", "Returns_QTY"]].copy()
    cur = work["date"].max()
    while cur < target_date and (cur + pd.DateOffset(months=1)).replace(day=1) < target_date:
        nxt = (cur + pd.DateOffset(months=1)).replace(day=1)
        feat_row = _build_feature_row(work, national_hist, ext_fips, static_row, nxt, base_date)
        x = feat_row[FULL_FEATURES].fillna(0)
        x_input = model_scaler.transform(x) if model_scaler is not None else x
        pred_raw = float(model.predict(x_input)[0])
        lo, hi = _seasonal_bounds_for_fips(bounds_df, fips, nxt.month)
        pred = float(np.clip(pred_raw, lo, hi))
        work = pd.concat([work, pd.DataFrame([{"date": nxt, "Sales_QTY": pred, "Returns_QTY": 0.0}])], ignore_index=True)
        cur = nxt

    final_feat = _build_feature_row(work, national_hist, ext_fips, static_row, target_date, base_date)
    x_baseline = final_feat[FULL_FEATURES].fillna(0).copy()
    x_whatif = x_baseline.copy()

    applied, unsupported = {}, []
    for raw_key, change in (feature_changes or {}).items():
        mapped = FEATURE_CHANGE_MAP.get(raw_key)
        if mapped is None:
            if raw_key in KNOWN_UNSUPPORTED or raw_key not in FULL_FEATURES:
                unsupported.append(raw_key)
                continue
            mapped = raw_key
        current_val = float(x_baseline.iloc[0][mapped])
        ctype = change.get("type", "percent")
        cval = float(change.get("value", 0))
        new_val = current_val * (1 + cval / 100.0) if ctype == "percent" else current_val + cval
        x_whatif.iloc[0, x_whatif.columns.get_loc(mapped)] = new_val
        applied[mapped] = {"before": current_val, "after": new_val, "change": change}

    if sensitivity_model is not None:
        x_base_input = model_scaler.transform(x_baseline) if model_scaler is not None else x_baseline
        baseline_pred = float(model.predict(x_base_input)[0])
        x_ridge_base = sensitivity_scaler.transform(x_baseline) if sensitivity_scaler is not None else x_baseline
        x_ridge_what = sensitivity_scaler.transform(x_whatif) if sensitivity_scaler is not None else x_whatif
        ridge_baseline = float(sensitivity_model.predict(x_ridge_base)[0])
        ridge_whatif = float(sensitivity_model.predict(x_ridge_what)[0])
        raw_pred = baseline_pred + (ridge_whatif - ridge_baseline)
    else:
        x_input = model_scaler.transform(x_whatif) if model_scaler is not None else x_whatif
        raw_pred = float(model.predict(x_input)[0])

    lo, hi = _seasonal_bounds_for_fips(bounds_df, fips, target_date.month)
    pred = float(np.clip(raw_pred, lo, hi))
    return pred, applied, unsupported


def model_fn(model_dir):
    global FULL_FEATURES
    feature_list_path = os.path.join(model_dir, "feature_list.json")
    if os.path.exists(feature_list_path):
        with open(feature_list_path) as f:
            feature_info = json.load(f)
        loaded_features = feature_info.get("features")
        if loaded_features:
            FULL_FEATURES = loaded_features
            print(f"Loaded {len(FULL_FEATURES)} features from feature_list.json "
                  f"(production_cutoff={feature_info.get('production_cutoff')}) -- "
                  f"this is the authoritative list the model was actually trained on.")
    else:
        print(f"WARNING: no feature_list.json found in {model_dir} -- falling back to the "
              f"hardcoded {len(FULL_FEATURES)}-feature list in this file, which may be stale "
              f"relative to what the model was actually trained on. This bundle should be "
              f"re-generated by a current train.py run.")

    bundle = {"models": {}, "ridge_scalers": {}, "fips_hist": {}, "national_hist": {}, "static": {}, "ext": {}, "bounds": {}}
    for family in ["Barrage", "Grounded"]:
        bundle["models"][family] = {}
        for role in ["champion", "challenger"]:
            for fname in os.listdir(model_dir):
                prefix = f"{role}_{family}_"
                if not fname.startswith(prefix):
                    continue
                full_path = os.path.join(model_dir, fname)
                if fname.endswith(".pkl"):
                    model_name = fname.replace(prefix, "").replace(".pkl", "")
                    with open(full_path, "rb") as f:
                        bundle["models"][family][role] = (model_name, pickle.load(f))
                    break
                elif fname.endswith(".ubj"):
                    # Native XGBoost format -- version-independent, no pickle fragility.
                    model_name = fname.replace(prefix, "").replace(".ubj", "")
                    booster = xgb.Booster()
                    booster.load_model(full_path)
                    bundle["models"][family][role] = (model_name, _NativeXGBWrapper(booster))
                    break
                elif fname.endswith(".txt"):
                    # Native LightGBM format -- version-independent, no pickle fragility.
                    model_name = fname.replace(prefix, "").replace(".txt", "")
                    booster = lgb.Booster(model_file=full_path)
                    bundle["models"][family][role] = (model_name, _NativeLGBWrapper(booster))
                    break

            # FAIL LOUDLY here rather than silently leaving this role out of the bundle --
            # a missing model file used to surface many steps later as a confusing,
            # unrelated-looking KeyError (e.g. bundle["models"]["Barrage"]["challenger"])
            # instead of a clear message pointing at the actual missing file.
            if role not in bundle["models"][family]:
                available = sorted(os.listdir(model_dir))
                raise FileNotFoundError(
                    f"No model file found for {family}/{role} in '{model_dir}'. Expected a "
                    f"file starting with '{role}_{family}_' (e.g. '{role}_{family}_XGBoost.ubj', "
                    f"'.txt', or '.pkl'). Files actually present in that directory: {available}"
                )

        fh = pd.read_csv(os.path.join(model_dir, f"fips_trailing_history_{family.lower()}.csv"),
                          parse_dates=["date"], dtype={"FIPS": str})
        fh["FIPS"] = fh["FIPS"].str.zfill(5)
        bundle["fips_hist"][family] = fh

        nh = pd.read_csv(os.path.join(model_dir, f"national_trailing_{family.lower()}.csv"), parse_dates=["date"])
        bundle["national_hist"][family] = nh

        st = pd.read_csv(os.path.join(model_dir, f"fips_static_features_{family.lower()}.csv"), dtype={"FIPS": str})
        st["FIPS"] = st["FIPS"].str.zfill(5)
        bundle["static"][family] = st.set_index("FIPS")

        ew = pd.read_csv(os.path.join(model_dir, f"fips_external_weather_{family.lower()}.csv"),
                          parse_dates=["date"], dtype={"FIPS": str})
        ew["FIPS"] = ew["FIPS"].str.zfill(5)
        bundle["ext"][family] = ew

        bd = pd.read_csv(os.path.join(model_dir, f"fips_guardrail_bounds_{family.lower()}.csv"), dtype={"FIPS": str})
        bd["FIPS"] = bd["FIPS"].str.zfill(5)
        bundle["bounds"][family] = bd

        scaler_path = os.path.join(model_dir, f"ridge_scaler_{family}.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                bundle["ridge_scalers"][family] = pickle.load(f)
        else:
            bundle["ridge_scalers"][family] = None
            print(f"WARNING: ridge_scaler_{family}.pkl not found -- Ridge predictions will use unscaled inputs (older model bundle)")

    bundle["base_date"] = pd.Timestamp("2013-10-01")
    return bundle


def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        return json.loads(request_body)
    elif content_type == "application/jsonlines":
        return json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def _compute_shap_importance(model, model_name, x, features):
    if model_name in ("XGBoost", "LightGBM"):
        # FIX: hasattr(model, 'booster') is the wrong check -- XGBRegressor's raw
        # sklearn wrapper has a 'booster' CONSTRUCTOR HYPERPARAMETER (defaults to
        # None, not the trained model), so hasattr() is True but model.booster is
        # None, and shap.TreeExplainer(None) fails. Confirmed directly: raw_model
        # .booster == None after fitting. The precise fix checks by TYPE, not
        # attribute existence -- _NativeXGBWrapper/_NativeLGBWrapper (this module's
        # live-endpoint model_fn() loading path) genuinely do store the real trained
        # booster under .booster, so they still need that access; a raw sklearn
        # XGBRegressor/LGBMRegressor (repack_evaluation.py's training path) does not,
        # and shap.TreeExplainer accepts those directly. Confirmed via direct testing
        # that passing model unconditionally (without this distinction) would break
        # the live endpoint, since shap.TreeExplainer does not recognize either
        # wrapper class passed directly (raises InvalidModelError).
        explainer = shap.TreeExplainer(
            model.booster if isinstance(model, (_NativeXGBWrapper, _NativeLGBWrapper)) else model
        )
        shap_values = explainer.shap_values(x)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_dict = {features[i]: shap_values[0][i] for i in range(len(features))}
    elif model_name == "Ridge":
        shap_dict = {features[i]: model.coef_[i] * x.iloc[0, i] for i in range(len(features))}
    else:
        return None, None
    
    external_features = get_external_feature_names(features)
    ext_shap = {k: v for k, v in shap_dict.items() if k in external_features}
    total_abs = sum(abs(v) for v in shap_dict.values())
    ext_abs = sum(abs(v) for v in ext_shap.values())
    ext_share = (ext_abs / total_abs * 100) if total_abs > 0 else 0.0
    
    top5 = sorted(ext_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    return [{"feature": k, "shap_value": round(v, 2)} for k, v in top5], round(ext_share, 1)


def compute_per_row_shap_reusing_explainer(explainer, model, model_name, x, features):
    """ALL external features, ranked, signed SHAP computation -- NOT truncated to
    top-5, per explicit customer requirement: they need to see every external
    feature's real contribution, top to bottom, since they use this to decide which
    feature to go query the What-If agent about next. Accepts an ALREADY-BUILT
    explainer to reuse across many rows (e.g. one call per county) instead of
    recreating a shap.TreeExplainer every single call -- meaningfully cheaper when
    computing this hundreds/thousands of times in a batch context, vs.
    _compute_shap_importance()'s single-call, live-endpoint explainability use case
    (which intentionally stays top-5 -- a concise chat answer vs. a dashboard's full
    list are genuinely different needs). Kept as a separate function rather than
    modifying _compute_shap_importance() itself, since that one is already used by
    the live endpoint and shouldn't risk changing behavior there."""
    if model_name in ("XGBoost", "LightGBM"):
        shap_values = explainer.shap_values(x)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_dict = {features[i]: shap_values[0][i] for i in range(len(features))}
    elif model_name == "Ridge":
        shap_dict = {features[i]: model.coef_[i] * x.iloc[0, i] for i in range(len(features))}
    else:
        return None, None

    external_features = get_external_feature_names(features)
    ext_shap = {k: v for k, v in shap_dict.items() if k in external_features}
    total_abs = sum(abs(v) for v in shap_dict.values())
    ext_abs = sum(abs(v) for v in ext_shap.values())
    ext_share = (ext_abs / total_abs * 100) if total_abs > 0 else 0.0

    ranked = sorted(ext_shap.items(), key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": k, "shap_value": round(v, 2)} for k, v in ranked], round(ext_share, 1)


def compute_global_feature_importance(model, model_name, sample_rows: "pd.DataFrame", features):
    """Computes GLOBAL feature importance -- 'what does this model care about overall' --
    from a representative SAMPLE of real feature rows, rather than one single prediction's
    explanation (that's what _compute_shap_importance() above is for). Averages the
    ABSOLUTE SHAP value per feature across the sample, the standard way to summarize a
    tree model's overall behavior. Deliberately NOT computed per-row across the full
    ~236,000-row batch forecast -- that would be prohibitively expensive and isn't what a
    dashboard "feature importance" graph needs anyway; a representative sample gives a
    genuine, real answer without that cost.

    Returns a list of {"feature", "friendly_name", "mean_abs_shap", "is_external"} dicts,
    covering EVERY feature (both the "all features" and "external only" dashboard graphs
    can be built by filtering this one list on is_external, rather than computing SHAP
    twice)."""
    if model_name not in ("XGBoost", "LightGBM", "Ridge"):
        return []

    x = sample_rows[features].fillna(0)
    if model_name in ("XGBoost", "LightGBM"):
        # FIX: see _compute_shap_importance() above for the full explanation -- same
        # bug, same precise fix (check by type, not attribute existence).
        explainer = shap.TreeExplainer(
            model.booster if isinstance(model, (_NativeXGBWrapper, _NativeLGBWrapper)) else model
        )
        shap_values = explainer.shap_values(x)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        mean_abs = np.abs(shap_values).mean(axis=0)
    else:  # Ridge -- coefficient x mean absolute feature value, standard linear-model importance
        mean_abs = np.abs(model.coef_ * x.mean(axis=0).values)

    display_external = set(get_display_external_feature_names(features))
    results = []
    for i, feat in enumerate(features):
        results.append({
            "feature": feat,
            "friendly_name": friendly_feature_name(feat),
            "mean_abs_shap": round(float(mean_abs[i]), 4),
            "is_external": feat in display_external,
        })
    return sorted(results, key=lambda r: r["mean_abs_shap"], reverse=True)


def predict_fn(payload, bundle):
    family = payload.get("family")
    if family not in ("Barrage", "Grounded"):
        raise ValueError(f"family must be 'Barrage' or 'Grounded', got {family!r}")
    
    static_table = bundle["static"][family]
    fips = resolve_fips(payload, static_table)
    
    month = int(payload["month"])
    year = int(payload["year"])
    if not (1 <= month <= 12):
        raise ValueError("month must be between 1 and 12.")
    scenario = bool(payload.get("scenario", False))
    feature_changes = payload.get("feature_changes")
    explain = bool(payload.get("explain", False))

    target_date = pd.Timestamp(year=year, month=month, day=1)
    fips_hist = bundle["fips_hist"][family]
    fips_hist_this = fips_hist[fips_hist["FIPS"] == fips]
    national_hist = bundle["national_hist"][family]
    static_table = bundle["static"][family]
    ext_all = bundle["ext"][family]
    ext_fips = ext_all[ext_all["FIPS"] == fips]
    # BUG FIX: this previously only included columns with a real _lag1/_chg3m sibling in
    # FULL_FEATURES -- which is ONLY the economic indicators (CPI, fertilizer, crop prices).
    # Weather columns (TAVG, TMAX, TMIN, PRCP, GDD_proxy, CHU_proxy, Precip_Anomaly) have no
    # such sibling, so they were NEVER extended into the future here -- meaning every forecast
    # beyond the real weather data's range (i.e. essentially all of Oct 2025 onward) received
    # NaN for every weather feature, silently zeroed out by the fillna(0) just before scoring.
    # Confirmed: preprocess.py's weather source (real NOAA history) has no future/climatology
    # data built in either, so this wasn't being covered anywhere else in the pipeline.
    # Fixed to extend every real external column _build_feature_row actually uses -- whether
    # referenced directly in FULL_FEATURES (weather) or via its _lag1/_chg3m derivative
    # (economic indicators) -- so _extend_series' genuine seasonal-average logic (a future
    # July gets a value based on real historical Julys) now correctly applies to weather too,
    # which is arguably the feature category it matters most for.
    ext_cols = [c for c in ext_fips.columns if c not in ("date", "FIPS") and
                (c in FULL_FEATURES or f"{c}_lag1" in FULL_FEATURES or f"{c}_chg3m" in FULL_FEATURES)]
    ext_fips = _extend_series(ext_fips, ext_cols, target_date) if len(ext_fips) else ext_fips
    bounds_df = bundle["bounds"][family]

    if fips in static_table.index:
        static_row = static_table.loc[fips]
        active_months = float(static_row.get("county_active_months", 0))
        quality = "sparse" if active_months < 6 else "active"
        features_found = True
    else:
        static_row = pd.Series({"county_alltime_share": 0.0, "county_recent_share": 0.0,
                                 "state_share": 0.0, "county_active_months": 0.0, "county_is_sparse": 1})
        active_months, quality, features_found = 0.0, "unknown_fips", False

    if fips_hist_this.empty:
        fips_hist_this = pd.DataFrame({"date": national_hist["date"], "Sales_QTY": 0.0, "Returns_QTY": 0.0})

    result = {
        "family": family, "fips": fips, "month": month, "year": year,
        "features_found": features_found, "county_data_quality": quality,
        "county_active_months": active_months,
    }

    ridge_role = None
    for role in ["champion", "challenger"]:
        if bundle["models"][family][role][0] == "Ridge":
            ridge_role = role
            break
    
    ridge_model = bundle["models"][family][ridge_role][1] if ridge_role else None
    ridge_scaler = bundle["ridge_scalers"][family] if ridge_role else None

    for role in ["champion", "challenger"]:
        model_name, model = bundle["models"][family][role]
        is_ridge = (model_name == "Ridge")
        model_scaler = bundle["ridge_scalers"][family] if is_ridge else None
        sensitivity_model = None if is_ridge else ridge_model
        sensitivity_scaler = None if is_ridge else ridge_scaler
        
        if scenario:
            pred, applied, unsupported = _apply_scenario_one_model(
                model, fips_hist_this, national_hist, ext_fips, static_row, bounds_df, fips,
                target_date, bundle["base_date"], feature_changes, model_scaler, sensitivity_model, sensitivity_scaler)
            result[role] = {"model_name": model_name, "predicted_qty": round(pred, 4)}
            result.setdefault("scenario_features", applied)
            if unsupported:
                result.setdefault("unsupported_feature_changes", unsupported)
        else:
            pred, source = _forecast_one_model(
                model, fips_hist_this, national_hist, ext_fips, static_row, bounds_df, fips,
                target_date, bundle["base_date"], model_scaler)
            result[role] = {"model_name": model_name, "predicted_qty": round(pred, 4)}
            result["feature_source"] = source
        
        if explain and role == "champion":
            feat_row = _build_feature_row(fips_hist_this, national_hist, ext_fips, static_row, target_date, bundle["base_date"])
            x = feat_row[FULL_FEATURES].fillna(0)
            if is_ridge and model_scaler:
                x_for_shap = pd.DataFrame(model_scaler.transform(x), columns=FULL_FEATURES)
            else:
                x_for_shap = x
            importance, ext_share = _compute_shap_importance(model, model_name, x_for_shap, FULL_FEATURES)
            if importance:
                result["feature_importance"] = importance
                result["external_factors_share_pct"] = ext_share

    return result


class NumpyJSONEncoder(json.JSONEncoder):
    """
    Handles numpy scalar types that json.dumps() cannot serialize natively.

    ROOT CAUSE (found via real production traceback, 2026-08-31): XGBoost's native
    Booster.predict() returns numpy.float32 (its internal engine uses 32-bit floats),
    while LightGBM's native Booster.predict() returns numpy.float64. Python's built-in
    round() PRESERVES the input type on numpy scalars -- round(np.float32(x), 4) is
    still a numpy.float32, not a native Python float. Since Barrage's Champion is
    XGBoost and Grounded's is LightGBM, this caused every single Barrage what-if/
    forecast request to fail with "TypeError: Object of type float32 is not JSON
    serializable" in output_fn() -- while Grounded (LightGBM) was never affected.

    This encoder is intentionally comprehensive (all numpy float/int/array types, not
    just float32) so any future numpy value anywhere in the response -- including
    nested SHAP feature_importance values -- is caught, rather than fixing only the
    one specific line that happened to be reported in this traceback.
    """
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction, cls=NumpyJSONEncoder), "application/json"

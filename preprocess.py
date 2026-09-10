"""
preprocess.py -- SageMaker Pipeline preprocessing step for the VAP forecasting pipeline.

Replaces the manual step of running Feature_Engineering_Pipeline.ipynb by hand and
uploading its output -- this script does the SAME tested feature engineering, as a real
pipeline step that runs automatically from raw inputs.

WHAT THIS SCRIPT DOES (identical logic to the notebook, just restructured as a script):
  1. Loads the raw sales Excel, builds leakage-safe internal features (calendar, county
     sales history lag/rolling, national context)
  2. Loads real weather data (NOAA), computes GDD/CHU/Precipitation Anomaly
  3. Loads real crop acreage data (USDA), lags it by 1 year (always safely known)
  4. Resolves real county names (the sales file's own tblCounties sheet + the Census
     Gazetteer file + documented manual overrides for pre-rename FIPS codes)
  5. Loads real economic data (FRED-sourced CPI/oil/fertilizer + crop commodity prices)
  6. Merges everything, adds leakage-safe static county/state scale features using an
     explicit, parameterized cutoff (matches train.py's --production-cutoff so both
     stay in sync)
  7. Saves complete_features_selected.parquet -- this is train.py's actual input

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO:
  - Does not select/drop feature groups based on ablation -- per prior instruction, all
    external features (economic + weather + acreage) are kept regardless of ablation
    contribution, since What-If responsiveness -- not ablation-measured accuracy -- is
    the priority for this project
  - Does not train any model -- that's train.py's job

SageMaker Pipeline I/O conventions (matches train.py's pattern):
  - Input data: /opt/ml/processing/input/ (or --s3-*-uri arguments as a fallback for each
    raw file, if running this directly rather than through a configured Processing Job)
  - Output: /opt/ml/processing/output/complete_features_selected.parquet
"""
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import boto3

# ============================================================
# SAGEMAKER I/O CONVENTIONS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str,
                         default=os.environ.get("SM_CHANNEL_INPUT", "/opt/ml/processing/input"),
                         help="Directory containing all raw input files (see REQUIRED_RAW_FILES below).")
    parser.add_argument("--output-dir", type=str,
                         default=os.environ.get("SM_OUTPUT_DIR", "/opt/ml/processing/output"))
    parser.add_argument("--s3-input-prefix", type=str, default=os.environ.get("S3_INPUT_PREFIX", ""),
                         help="s3://bucket/prefix/ containing all raw files, used as a fallback if "
                              "--input-dir is empty (e.g. running this directly rather than through "
                              "a configured Processing Job).")
    parser.add_argument("--production-cutoff", type=str, default=None,
                         help="If not given, this is computed automatically as the latest real "
                              "date found in the raw sales data -- this is what prevents data "
                              "leakage AND ensures all available real data gets used, without "
                              "needing anyone to manually update a hardcoded date each time new "
                              "data arrives. Only pass this explicitly if you deliberately want "
                              "to hold back recent months (e.g. to reproduce a specific past run).")
    return parser.parse_args()


REQUIRED_RAW_FILES = [
    "Grounded_Barrage_Sales_1.xlsx",
    "noaa_climate_features_by_county.csv",
    "usda_crop_acreage.csv",
    "external_features_real_only.csv",
    "crop_commodity_prices_full.csv",
    "2024_Gaz_counties_national.txt",
]


def ensure_raw_files(input_dir, s3_input_prefix):
    os.makedirs(input_dir, exist_ok=True)
    missing = [f for f in REQUIRED_RAW_FILES if not os.path.exists(os.path.join(input_dir, f))]
    if not missing:
        return
    if not s3_input_prefix:
        raise FileNotFoundError(
            f"Missing raw input files in {input_dir}: {missing}. Either fix the Processing "
            f"Job's input channel, or pass --s3-input-prefix s3://your-bucket/data/raw/ so "
            f"this script downloads them directly."
        )
    if not s3_input_prefix.startswith("s3://"):
        raise ValueError(f"--s3-input-prefix must start with 's3://', got: {s3_input_prefix}")
    bucket, prefix = s3_input_prefix.replace("s3://", "", 1).split("/", 1)
    s3 = boto3.client("s3")
    for fname in missing:
        print(f"Downloading {fname} from s3://{bucket}/{prefix.rstrip('/')}/{fname} ...", flush=True)
        s3.download_file(bucket, f"{prefix.rstrip('/')}/{fname}", os.path.join(input_dir, fname))
    print("All raw files present.", flush=True)


# ============================================================
# SECTION 1 -- Internal features (calendar, county sales history, national context)
# ============================================================
MONTH_MAP = {'JANUARY':1,'FEBRUARY':2,'MARCH':3,'APRIL':4,'MAY':5,'JUNE':6,'JULY':7,
             'AUGUST':8,'SEPTEMBER':9,'OCTOBER':10,'NOVEMBER':11,'DECEMBER':12}


def load_raw(path):
    raw = pd.read_excel(path, sheet_name='sales', engine='openpyxl')
    raw['fy'] = raw['Year'].astype(str).str.replace('F', '', regex=False).astype(int)
    raw['m'] = raw['Month'].map(MONTH_MAP)
    raw['calendar_year'] = np.where(raw['m'] >= 10, raw['fy'] - 1, raw['fy'])
    raw['date'] = pd.to_datetime(dict(year=raw['calendar_year'], month=raw['m'], day=1))
    raw['FIPS'] = raw['FIPS'].astype(str).str.replace('.0', '', regex=False).str.zfill(5)
    return raw


def build_full_grid(raw, family):
    fam_raw = raw[raw['Family'] == family].copy()
    all_months = pd.date_range(fam_raw['date'].min(), fam_raw['date'].max(), freq='MS')
    all_fips = fam_raw[['FIPS', 'State']].drop_duplicates()
    grid = all_fips.assign(key=1).merge(pd.DataFrame({'date': all_months, 'key': 1}), on='key').drop(columns='key')
    actual = fam_raw.groupby(['FIPS', 'date'], as_index=False).agg(
        Sales_QTY=('Sales QTY', 'sum'), Returns_QTY=('Returns/Credits QTY', 'sum'))
    full = grid.merge(actual, on=['FIPS', 'date'], how='left')
    full['Sales_QTY'] = full['Sales_QTY'].fillna(0.0)
    full['Returns_QTY'] = full['Returns_QTY'].fillna(0.0)
    full['Family'] = family
    return full.sort_values(['FIPS', 'date']).reset_index(drop=True)


def _calendar_feats(date):
    m = date.month
    fiscal_month_idx = ((m - 10) % 12) + 1
    return {
        'calendar_month': m, 'quarter': date.quarter, 'fiscal_month_idx': fiscal_month_idx,
        'month_sin': np.sin(2*np.pi*m/12), 'month_cos': np.cos(2*np.pi*m/12),
        'fiscal_sin': np.sin(2*np.pi*fiscal_month_idx/12), 'fiscal_cos': np.cos(2*np.pi*fiscal_month_idx/12),
        'growing_season_proxy': max(np.sin(np.pi*(m-3)/7), 0),
        'is_planting_season': int(m in [3,4,5]), 'is_application_season': int(m in [4,5,6,7,8]),
        'is_harvest_season': int(m in [9,10,11]), 'is_dormant_season': int(m in [12,1,2]),
    }


def add_internal_time_safe_features(full, national):
    df = full.sort_values(['FIPS', 'date']).reset_index(drop=True).copy()
    cal = df['date'].apply(_calendar_feats).apply(pd.Series)
    df = pd.concat([df, cal], axis=1)
    df['time_idx'] = (df['date'] - df['date'].min()).dt.days // 30

    g = df.groupby('FIPS')['Sales_QTY']
    for lag in [1, 2, 3, 6, 12]:
        df[f'fips_lag_{lag}'] = g.shift(lag)
    df['_shifted'] = df.groupby('FIPS')['Sales_QTY'].shift(1)
    for w in [3, 6, 12]:
        df[f'fips_rollmean_{w}'] = df.groupby('FIPS')['_shifted'].transform(lambda s: s.rolling(w).mean())
        df[f'fips_rollstd_{w}'] = df.groupby('FIPS')['_shifted'].transform(lambda s: s.rolling(w).std())
    df.drop(columns=['_shifted'], inplace=True)
    df['returns_lag_1'] = df.groupby('FIPS')['Returns_QTY'].shift(1)

    nat = national.sort_values('date').copy()
    nat['national_lag_1'] = nat['Sales_QTY'].shift(1)
    nat['national_lag_12'] = nat['Sales_QTY'].shift(12)
    nat['national_rollmean_3'] = nat['Sales_QTY'].shift(1).rolling(3).mean()
    df = df.merge(nat[['date', 'national_lag_1', 'national_lag_12', 'national_rollmean_3']], on='date', how='left')
    return df


def add_static_scale_features(df, train_cutoff):
    """LEAKAGE-SENSITIVE. train_cutoff must match train.py's --production-cutoff."""
    df = df.copy()
    for fam in df['Family'].unique():
        fam_mask = df['Family'] == fam
        train_hist = df[fam_mask & (df['date'] <= train_cutoff)]
        fam_total = train_hist['Sales_QTY'].sum()
        county_total = train_hist.groupby('FIPS')['Sales_QTY'].sum()
        county_share = (county_total / fam_total) if fam_total > 0 else county_total * 0
        recent_hist = train_hist[train_hist['date'] >= train_cutoff - pd.DateOffset(years=3)]
        recent_fam_total = recent_hist['Sales_QTY'].sum()
        county_recent_total = recent_hist.groupby('FIPS')['Sales_QTY'].sum()
        county_recent_share = (county_recent_total / recent_fam_total) if recent_fam_total > 0 else county_recent_total * 0
        state_total = train_hist.groupby('State')['Sales_QTY'].sum()
        state_share = (state_total / fam_total) if fam_total > 0 else state_total * 0
        county_n_active_months = train_hist[train_hist['Sales_QTY'] > 0].groupby('FIPS').size()
        df.loc[fam_mask, 'county_alltime_share'] = df.loc[fam_mask, 'FIPS'].map(county_share).fillna(0.0)
        df.loc[fam_mask, 'county_recent_share'] = df.loc[fam_mask, 'FIPS'].map(county_recent_share).fillna(0.0)
        df.loc[fam_mask, 'state_share'] = df.loc[fam_mask, 'State'].map(state_share).fillna(0.0)
        df.loc[fam_mask, 'county_active_months'] = df.loc[fam_mask, 'FIPS'].map(county_n_active_months).fillna(0.0)
        df.loc[fam_mask, 'county_is_sparse'] = (df.loc[fam_mask, 'county_active_months'] < 6).astype(int)
    return df


# ============================================================
# SECTION 2 -- External features (weather, acreage, county names, economic)
# ============================================================
def build_weather_features(noaa_path):
    raw_noaa = pd.read_csv(noaa_path, dtype={'FIPS': str})
    raw_noaa['FIPS'] = raw_noaa['FIPS'].str.zfill(5)
    pivoted = raw_noaa.pivot_table(index=['FIPS', 'date'], columns='datatype', values='value', aggfunc='first').reset_index()
    pivoted['date'] = pd.to_datetime(pivoted['date']).dt.to_period('M').dt.to_timestamp()

    if 'TAVG' in pivoted.columns:
        pivoted['GDD_proxy'] = (pivoted['TAVG'] - 50).clip(lower=0)
    if 'TMAX' in pivoted.columns and 'TMIN' in pivoted.columns:
        tmax_c = (pivoted['TMAX'] - 32) * 5 / 9
        tmin_c = (pivoted['TMIN'] - 32) * 5 / 9
        ymin = (1.8 * (tmin_c - 4.4) * (1 - 0.0195 * (tmin_c - 10) ** 2)).clip(lower=0)
        ymax = (3.33 * (tmax_c - 10) - 0.084 * (tmax_c - 10) ** 2).clip(lower=0)
        has_both = pivoted['TMAX'].notna() & pivoted['TMIN'].notna()
        pivoted.loc[has_both, 'CHU_proxy'] = ((ymin + ymax) / 2)[has_both]
    if 'PRCP' in pivoted.columns:
        pivoted['calendar_month_tmp'] = pivoted['date'].dt.month
        monthly_norm = pivoted.groupby(['FIPS', 'calendar_month_tmp'])['PRCP'].transform('mean')
        pivoted['Precip_Anomaly'] = pivoted['PRCP'] - monthly_norm
        pivoted.drop(columns=['calendar_month_tmp'], inplace=True)
    pivoted['has_temperature_data'] = pivoted.get('TAVG', pd.Series(dtype=float)).notna().astype(float)

    keep_cols = ['FIPS', 'date', 'PRCP', 'TAVG', 'TMAX', 'TMIN', 'GDD_proxy', 'CHU_proxy',
                 'Precip_Anomaly', 'has_temperature_data']
    return pivoted[[c for c in keep_cols if c in pivoted.columns]]


def build_acreage_features(usda_path, county_state_map):
    usda = pd.read_csv(usda_path)
    usda = usda[usda['statisticcat_desc'] == 'AREA PLANTED']
    usda['Value'] = usda['Value'].astype(str).str.replace(',', '', regex=False)
    usda['Value'] = pd.to_numeric(usda['Value'], errors='coerce')
    usda = usda.dropna(subset=['Value'])

    pivoted = usda.pivot_table(index=['year', 'state_alpha'], columns='commodity_desc',
                                 values='Value', aggfunc='first').reset_index()
    pivoted.columns = [f'{c}_Acres' if c not in ('year', 'state_alpha') else c for c in pivoted.columns]
    pivoted = pivoted.rename(columns={'year': 'Year', 'state_alpha': 'State'})

    acreage_cols = [c for c in pivoted.columns if c.endswith('_Acres')]
    pivoted = pivoted.sort_values(['State', 'Year'])
    for col in acreage_cols:
        pivoted[f'{col}_prior_year'] = pivoted.groupby('State')[col].shift(1)
    pivoted = pivoted.drop(columns=acreage_cols)

    merged = county_state_map.merge(pivoted, on=['State', 'Year'], how='left')
    return merged


MANUAL_COUNTY_NAMES = {
    '09001': 'Fairfield County', '09003': 'Hartford County', '09005': 'Litchfield County',
    '09007': 'Middlesex County', '09009': 'New Haven County', '09011': 'New London County',
    '09013': 'Tolland County', '09015': 'Windham County',
    '12025': 'Dade County',  # renamed Miami-Dade (12086) in 1997; this data uses the old code
}


def build_county_names(sales_path, gazetteer_path):
    tbl_counties = pd.read_excel(sales_path, sheet_name='tblCounties', engine='openpyxl')
    tbl_counties['FIPS'] = tbl_counties['FIPS'].astype(str).str.zfill(5)
    tbl_counties = tbl_counties.drop_duplicates(subset=['FIPS'], keep='first')

    gazetteer = pd.read_csv(gazetteer_path, sep='\t')
    gazetteer.columns = [c.strip() for c in gazetteer.columns]
    gazetteer['FIPS'] = gazetteer['GEOID'].astype(str).str.zfill(5)
    gazetteer_names = gazetteer[['FIPS', 'NAME']].rename(columns={'NAME': 'County_Name_Gazetteer'})

    county_names = tbl_counties[['FIPS', 'County']].rename(columns={'County': 'County_Name'})
    county_names = county_names.merge(gazetteer_names, on='FIPS', how='outer')
    county_names['County_Name'] = county_names['County_Name'].fillna(county_names['County_Name_Gazetteer'])
    county_names = county_names[['FIPS', 'County_Name']].drop_duplicates(subset=['FIPS'])

    for fips, name in MANUAL_COUNTY_NAMES.items():
        if fips in county_names['FIPS'].values:
            county_names.loc[county_names['FIPS'] == fips, 'County_Name'] = name
        else:
            county_names = pd.concat([county_names, pd.DataFrame([{'FIPS': fips, 'County_Name': name}])], ignore_index=True)
    return county_names


def build_economic_features(econ_path, crop_prices_path):
    economic_macro = pd.read_csv(econ_path, parse_dates=['date'])
    crop_prices = pd.read_csv(crop_prices_path, parse_dates=['date'])
    economic = economic_macro.merge(crop_prices, on='date', how='outer').sort_values('date')
    economic_cols = [c for c in economic.columns if c != 'date']
    return economic, economic_cols


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import sys
    if "--install-deps" in sys.argv:
        idx = sys.argv.index("--install-deps")
        deps = sys.argv[idx + 1]
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", deps])
        sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a not in ["--install-deps", deps]]
    
    args = parse_args()

    print("=== Step 0: Ensure all raw files are present ===", flush=True)
    ensure_raw_files(args.input_dir, args.s3_input_prefix)

    def p(fname):
        return os.path.join(args.input_dir, fname)

    print("\n=== Step 1: Internal features ===", flush=True)
    raw = load_raw(p("Grounded_Barrage_Sales_1.xlsx"))

    if args.production_cutoff:
        production_cutoff = pd.Timestamp(args.production_cutoff)
        print(f"Using explicitly-provided production cutoff: {production_cutoff.date()}", flush=True)
    else:
        production_cutoff = raw['date'].max()
        print(f"No --production-cutoff given -- computed automatically from the raw data's "
              f"own latest real month: {production_cutoff.date()}", flush=True)

    national_series = {fam: raw[raw['Family']==fam].groupby('date', as_index=False)['Sales QTY'].sum()
                            .rename(columns={'Sales QTY': 'Sales_QTY'}) for fam in ['Barrage', 'Grounded']}
    grids = {fam: build_full_grid(raw, fam) for fam in ['Barrage', 'Grounded']}
    internal_time_safe = {fam: add_internal_time_safe_features(grids[fam], national_series[fam]) for fam in ['Barrage', 'Grounded']}
    print(f"Internal features built: { {fam: internal_time_safe[fam].shape for fam in internal_time_safe} }", flush=True)

    print("\n=== Step 2: External features (weather, acreage, county names, economic) ===", flush=True)
    weather = build_weather_features(p("noaa_climate_features_by_county.csv"))
    print(f"Weather: {weather.shape}, {weather['FIPS'].nunique():,} counties with real data", flush=True)

    county_state_map = pd.concat([internal_time_safe[fam][['FIPS', 'State', 'date']] for fam in ['Barrage', 'Grounded']]).drop_duplicates()
    county_state_map['Year'] = county_state_map['date'].dt.year
    acreage = build_acreage_features(p("usda_crop_acreage.csv"), county_state_map[['FIPS', 'State', 'Year']].drop_duplicates())
    print(f"Acreage: {acreage.shape}", flush=True)

    county_names = build_county_names(p("Grounded_Barrage_Sales_1.xlsx"), p("2024_Gaz_counties_national.txt"))
    print(f"County names resolved: {county_names['County_Name'].notna().sum()} of {county_names['FIPS'].nunique()}", flush=True)

    economic, economic_cols = build_economic_features(p("external_features_real_only.csv"), p("crop_commodity_prices_full.csv"))
    print(f"Economic: {economic.shape}", flush=True)

    print("\n=== Step 3: Merge everything ===", flush=True)
    merged = {}
    for fam in ['Barrage', 'Grounded']:
        d = internal_time_safe[fam].copy()
        d = d.merge(weather, on=['FIPS', 'date'], how='left')
        d['Year'] = d['date'].dt.year
        d = d.merge(acreage.drop(columns=['State']), on=['FIPS', 'Year'], how='left')
        d = d.merge(economic, on='date', how='left')
        for col in economic_cols:
            if col in d.columns:
                d[f'{col}_lag1'] = d.groupby('FIPS')[col].shift(1)
                d[f'{col}_chg3m'] = d[col] - d.groupby('FIPS')[col].shift(3)
        merged[fam] = d
        print(f"{fam}: merged shape {d.shape}", flush=True)

    print("\n=== Step 4: Add leakage-safe static features + county names, save ===", flush=True)
    NON_FEATURE_COLUMNS = {'FIPS','County_Name','State','Family','date','Sales_QTY','Returns_QTY','Year'}
    STATIC_FEATURES = ['county_alltime_share', 'county_recent_share', 'state_share', 'county_active_months', 'county_is_sparse']

    final = {}
    for fam in ['Barrage', 'Grounded']:
        d = add_static_scale_features(merged[fam], production_cutoff)
        d = d.merge(county_names, on='FIPS', how='left')
        all_cols = set(d.columns) - NON_FEATURE_COLUMNS | set(STATIC_FEATURES)
        keep_cols = ['FIPS', 'County_Name', 'State', 'Family', 'date', 'Sales_QTY', 'Returns_QTY'] + sorted(all_cols)
        keep_cols = [c for c in keep_cols if c in d.columns]
        final[fam] = d[keep_cols]

    complete = pd.concat(final.values(), ignore_index=True)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "complete_features_selected.parquet")
    complete.to_parquet(out_path, index=False)
    n_features = len([c for c in complete.columns if c not in NON_FEATURE_COLUMNS])
    print(f"\nSaved {out_path}: {complete.shape} ({n_features} features)", flush=True)
    print(f"County names present for {complete[['FIPS','County_Name']].drop_duplicates()['County_Name'].notna().sum()} "
          f"of {complete['FIPS'].nunique()} counties", flush=True)

    # Write the ACTUAL cutoff used to a small metadata file -- train.py reads this by default
    # instead of taking its own separate --production-cutoff argument, so the two steps can
    # never independently drift to different cutoff values (which would silently reintroduce
    # a leakage risk: e.g. if train.py used a LATER cutoff than what preprocess.py actually
    # computed static features against, those static features would be leakage-safe for the
    # WRONG date, not the one train.py thinks it's training through).
    import json
    metadata_path = os.path.join(args.output_dir, "preprocessing_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({"production_cutoff": production_cutoff.strftime("%Y-%m-%d")}, f, indent=2)
    print(f"Saved {metadata_path} (production_cutoff={production_cutoff.date()}) -- "
          f"train.py reads this by default.", flush=True)

    print("\nPreprocessing complete.", flush=True)

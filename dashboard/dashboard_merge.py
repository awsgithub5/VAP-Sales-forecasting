"""
Merge VAP forecasting pipeline outputs for QuickSight dashboard
Consolidates predictions, actuals, weather, and reference data into fact + dimension tables
"""
import pandas as pd
import boto3
import numpy as np
import os
from datetime import datetime

BUCKET = os.environ.get("VAP_S3_BUCKET", "vap-sales-forecasting")
s3 = boto3.client('s3')

def load_s3_csv(key):
    """Load CSV from S3"""
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(obj['Body'])

def load_s3_excel(key):
    """Load Excel from S3"""
    import io
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_excel(io.BytesIO(obj['Body'].read()))

print("=" * 80)
print("STEP 1: DATE NORMALIZATION")
print("=" * 80)

# Load historical sales with fiscal year format
sales_hist = load_s3_excel("raw-data/Grounded_Barrage_Sales_1.xlsx")
print(f"\nHistorical sales: {len(sales_hist)} rows")
print(f"Sample: {sales_hist[['Year', 'Month']].head(3).to_dict('records')}")

# Convert fiscal year + month to calendar date
# F2014 = Fiscal 2014 (Oct 2013 - Sep 2014)
month_map = {
    'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12,
    'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4, 'MAY': 5, 'JUNE': 6,
    'JULY': 7, 'AUGUST': 8, 'SEPTEMBER': 9
}

def fiscal_to_calendar(fiscal_year, month_name):
    """Convert fiscal year (F2014) + month name to calendar date"""
    cal_year = int(fiscal_year.replace('F', ''))
    month_num = month_map[month_name.upper()]
    # Oct-Dec belong to previous calendar year
    if month_num >= 10:
        cal_year -= 1
    return pd.Timestamp(year=cal_year, month=month_num, day=1)

sales_hist['date'] = sales_hist.apply(lambda r: fiscal_to_calendar(r['Year'], r['Month']), axis=1)
print(f"Converted fiscal dates. Sample: {sales_hist[['Year', 'Month', 'date']].head(3).to_dict('records')}")

# Prepare historical sales for fact table
sales_hist['DataType'] = 'Historical'

if 'Sales QTY' not in sales_hist.columns:
    raise KeyError(
        f"'Sales QTY' column not found in raw sales file -- available columns: "
        f"{list(sales_hist.columns)}. The real column name may differ slightly "
        f"(e.g. underscore vs space) -- check before assuming this fix is correct."
    )
sales_hist['Actual_QTY'] = sales_hist['Sales QTY']
sales_hist['Predicted_QTY'] = np.nan
sales_hist['ModelName'] = np.nan
sales_hist = sales_hist.rename(columns={'State': 'State'})
print(f"Prepared historical sales for merge: {len(sales_hist)} rows")

print("\n" + "=" * 80)
print("STEP 2: LOAD ALL FILES")
print("=" * 80)

"""

Generate county_accuracy.csv for the County-Level Accuracy dashboard tab.
 
Computes real, per-county R2/WMAPE for BOTH Champion and Challenger, reusing the

validation predictions train_with_holdout.py already generated during actual training

-- rather than training a second, separate shadow model from scratch on the same

holdout window (TRAIN_CUTOFF=2024-09-01, VAL_START/END=2024-10-01/2025-09-01 -- confirmed

identical to train_with_holdout.py's own real split). Reuses validation_predictions_all.csv,

which dashboard_merge.py uploads right before this script runs (same EvaluateAllModels step).
 
Uploads to the same S3 location as before:

    s3://vap-sales-forecasting/dashboard/county_accuracy.csv

Now includes a ModelRole column (Champion/Challenger), which the original file did not have.

"""

import pandas as pd

import numpy as np

import boto3

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

    # Categorical only -- raw R2 is never the headline; volume gates confidence

    # before accuracy does, since a Good R2 on a near-zero-volume county is still

    # not something worth planning inventory around.

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
 
# Merge the 4 validation_predictions files inline 

validation_parts = []

for family in ["barrage", "grounded"]:

    for role in ["champion", "challenger"]:

        key = f"dashboard/validation_predictions_{family}_{role}.csv"

        try:

            validation_parts.append(load_s3_csv(key))

            print(f"  Loaded {key}")

        except Exception as e:

            print(f"  WARNING: could not load {key}: {e}")
 
validation = pd.concat(validation_parts, ignore_index=True)

validation['date'] = pd.to_datetime(validation['date'])


validation['ModelRole'] = validation['ModelRole'].str.title()

print(f"Validation predictions (merged): {len(validation)} rows")
 
validation.to_csv('/tmp/validation_predictions_all.csv', index=False)

s3.upload_file('/tmp/validation_predictions_all.csv', BUCKET, 'dashboard/validation_predictions_all.csv')

print(f"Uploaded validation_predictions_all.csv ({len(validation)} rows)")
 

forecast = load_s3_csv("comparison/forecast_2yr_all_models.csv")
forecast['date'] = pd.to_datetime(forecast['date'])
forecast['ModelRole'] = forecast['ModelRole'].str.title()
print(f"Forecast: {len(forecast)} rows")

# NOTE: weather is loaded WITHOUT a Family tag now - see STEP 3, weather is physically the same
# regardless of product family and should not be split/joined per family (that was the source of
# the duplicate-row and mismatched-Family bug in the previous version of this script).
weather_b = load_s3_csv("lookup/fips_external_weather_barrage.csv")
weather_b['date'] = pd.to_datetime(weather_b['date'])

weather_g = load_s3_csv("lookup/fips_external_weather_grounded.csv")
weather_g['date'] = pd.to_datetime(weather_g['date'])

static_b = load_s3_csv("lookup/fips_static_features_barrage.csv")
static_b['Family'] = 'Barrage'

static_g = load_s3_csv("lookup/fips_static_features_grounded.csv")
static_g['Family'] = 'Grounded'

bounds_b = load_s3_csv("lookup/fips_guardrail_bounds_barrage.csv")
bounds_b['Family'] = 'Barrage'

bounds_g = load_s3_csv("lookup/fips_guardrail_bounds_grounded.csv")
bounds_g['Family'] = 'Grounded'

national_b = load_s3_csv("lookup/national_trailing_barrage.csv")
national_b['date'] = pd.to_datetime(national_b['date'])
national_b['Family'] = 'Barrage'

national_g = load_s3_csv("lookup/national_trailing_grounded.csv")
national_g['date'] = pd.to_datetime(national_g['date'])
national_g['Family'] = 'Grounded'

print("\n" + "=" * 80)
print("STEP 3: BUILD A SINGLE UNIVERSAL WEATHER TABLE (FIX)")
print("=" * 80)
# Weather/prices don't vary by product family - the old script kept two family-specific weather
# files and joined on [FIPS, date] only (no Family), which silently fanned out any county that
# existed in both files, producing duplicate Champion/Challenger rows downstream.
# fips_external_weather_grounded.csv also only covers a small subset of Grounded's counties.
# Fix: use the more complete Barrage weather file as the base, and fill in any FIPS+date
# combinations only present in the Grounded file (keeps full coverage, one row per FIPS+date).
weather = (
    pd.concat([weather_b, weather_g], ignore_index=True)
    .drop_duplicates(subset=['FIPS', 'date'], keep='first')  # barrage rows come first -> preferred
)
print(f"Universal weather table: {len(weather)} rows, {weather['FIPS'].nunique()} counties")

grounded_counties = set(static_g['FIPS'].unique())
weather_counties = set(weather[weather['TAVG'].notna()]['FIPS'].unique())
missing_weather = grounded_counties - weather_counties
if len(missing_weather) > 0:
    print(f"\nWARNING: {len(missing_weather)} Grounded counties ({len(missing_weather)/len(grounded_counties)*100:.1f}%) "
          f"still have no weather station history in either source file.")
    print("These will be filled via state-level climatology in STEP 6b; a residual few may remain null.")

print("\n" + "=" * 80)
print("STEP 4: UNION PAIRED FILES")
print("=" * 80)

static = pd.concat([static_b, static_g], ignore_index=True)
# static_b/static_g also carry their own County_Name/State columns, which would collide with
# the ones already in `fact` (from validation/forecast) and get silently renamed to
# County_Name_x/_y, State_x/_y on merge. Drop them here - fact's copies are authoritative.
static = static.drop(columns=[c for c in ['County_Name', 'State'] if c in static.columns])
print(f"Static features: {len(static)} rows")

bounds = pd.concat([bounds_b, bounds_g], ignore_index=True)
bounds['calendar_month'] = bounds['calendar_month'].astype(int)
print(f"Guardrail bounds: {len(bounds)} rows")

national = pd.concat([national_b, national_g], ignore_index=True)

# Add historical national aggregates
hist_national = sales_hist.groupby(['date', 'Family']).agg({
    'Actual_QTY': 'sum'
}).reset_index()
hist_national['date'] = pd.to_datetime(hist_national['date'])
hist_national = hist_national.rename(columns={'Actual_QTY': 'Sales_QTY'})
hist_national['DataType'] = 'Historical'

# Keep only the original national trailing for reference (optional - can be removed)
national['date'] = pd.to_datetime(national['date'])
national = national.sort_values(['Family', 'date'])
print(f"National trailing (original): {len(national)} rows")

print("\n" + "=" * 80)
print("STEP 5: BUILD FACT TABLE")
print("=" * 80)

validation['DataType'] = 'Validation'
validation_cols = ['FIPS', 'County_Name', 'State', 'date', 'Family', 'ModelRole', 'ModelName',
                    'Actual_QTY', 'Predicted_QTY', 'DataType']
validation = validation[validation_cols]

forecast['DataType'] = 'Forecast'
forecast['Actual_QTY'] = np.nan
forecast_cols = ['FIPS', 'County_Name', 'State', 'date', 'Family', 'ModelRole', 'ModelName',
                  'Actual_QTY', 'Predicted_QTY', 'DataType', 'External_Factors_Share_Pct',
                  'Top1_Feature', 'Top1_SHAP', 'Top2_Feature', 'Top2_SHAP',
                  'Top3_Feature', 'Top3_SHAP', 'Top4_Feature', 'Top4_SHAP',
                  'Top5_Feature', 'Top5_SHAP']
forecast = forecast[forecast_cols]

for col in forecast_cols:
    if col not in validation.columns:
        validation[col] = np.nan

# Prepare historical sales - get County_Name by merging on FIPS only
county_lookup = validation[['FIPS', 'County_Name']].drop_duplicates(subset=['FIPS'])
sales_hist = sales_hist.merge(county_lookup, on='FIPS', how='left')

# Ensure all forecast_cols exist in sales_hist
for col in forecast_cols:
    if col not in sales_hist.columns:
        sales_hist[col] = np.nan

sales_hist_final = sales_hist[forecast_cols]

# BUG FIX: sales_hist['ModelRole'] was left as NaN (never set), unlike Validation and
# Forecast rows, which always have a real "Champion"/"Challenger" value. Confirmed as
# the actual cause of "historical data invisible on the dashboard, only Oct 2024
# onward visible" -- any ModelRole filter/selector (Champion vs Challenger, which
# this whole platform is built around) would never match NaN, silently hiding every
# Historical row whenever a specific model is selected. Duplicated across both roles
# instead, exactly like Historical rows already are for county-level data elsewhere
# in this pipeline, so they stay visible regardless of which model the user has
# selected -- historical actuals aren't tied to a specific model in the first place.
hist_champion = sales_hist_final.copy()
hist_champion['ModelRole'] = 'Champion'
hist_challenger = sales_hist_final.copy()
hist_challenger['ModelRole'] = 'Challenger'
sales_hist_final = pd.concat([hist_champion, hist_challenger], ignore_index=True)
print(f"Historical actuals: {len(sales_hist_final)} rows (x2 for Champion/Challenger visibility)")

fact = pd.concat([sales_hist_final, validation, forecast], ignore_index=True)
print(f"Stacked historical + validation + forecast: {len(fact)} rows")

# grain sanity check on the stacked prediction data BEFORE any joins - catches source-data
# duplicates early rather than after they've been obscured by a join
dupe_check = fact.groupby(['FIPS', 'date', 'Family', 'ModelRole']).size()
if (dupe_check > 1).any():
    print(f"WARNING: {(dupe_check > 1).sum()} FIPS+date+Family+ModelRole combinations "
          f"have more than one row BEFORE any lookup joins - investigate source files.")

fact['calendar_month'] = fact['date'].dt.month

# FIX: join weather on [FIPS, date] only (matches the new family-agnostic weather table) -
# do NOT join on Family, since Family is not a column in `weather` anymore.
fact = fact.merge(weather, on=['FIPS', 'date'], how='left')
print(f"Joined weather: {len(fact)} rows")

# FIX: static features and guardrail bounds genuinely differ by Family, so these keep Family in the join key.
fact = fact.merge(static, on=['FIPS', 'Family'], how='left')
print(f"Joined static features: {len(fact)} rows")

fact = fact.merge(bounds, on=['FIPS', 'Family', 'calendar_month'], how='left')
print(f"Joined guardrail bounds: {len(fact)} rows")

grain_final = fact.groupby(['FIPS', 'date', 'Family', 'ModelRole']).size()
print(f"Post-join grain check - max rows per FIPS+date+Family+ModelRole: {grain_final.max()} (should be 1)")

print("\n" + "=" * 80)
print("STEP 6: FILL FUTURE EXTERNAL FACTORS (weather has no data past the last actual month)")
print("=" * 80)
# Actual weather/prices don't exist for future forecast dates. Fill using:
#  - county-level climatology (10yr trailing mean by FIPS+calendar_month) for weather variables
#  - state-level climatology as a fallback where a county has no usable history
#  - trailing 12-month average for commodity prices / CPI / PPI (these trend rather than
#    follow a seasonal cycle, so climatology doesn't apply the same way - a futures-curve
#    price feed would be a better upgrade here if/when available)
weather_metric_cols = ['TAVG', 'TMAX', 'TMIN', 'PRCP', 'GDD_proxy', 'CHU_proxy', 'Precip_Anomaly']
price_cols = ['CPI_AllUrban', 'Fertilizer_PPI', 'Cotton_Price_USCentsPerLb', 'Corn_Price_USDPerMT',
              'Soybeans_Price_USDPerMT', 'Wheat_Price_USDPerMT', 'WTI_Oil_Price']

last_actual_date = fact.loc[fact['DataType'] == 'Validation', 'date'].max()
history = fact[fact['date'] <= last_actual_date].copy()
future = fact[fact['date'] > last_actual_date].copy()

recent_history = history[history['date'] >= last_actual_date - pd.DateOffset(years=10)]

county_clim = (
    recent_history.groupby(['FIPS', 'calendar_month'])[weather_metric_cols].mean()
    .reset_index().rename(columns={c: c + '_cclim' for c in weather_metric_cols})
)
# National-level (all-county) climatology as the fallback layer, in case a county has no usable
# history of its own. Deliberately NOT keyed on State - some fact tables from the pipeline don't
# carry a State column, so this avoids depending on one being present.
national_clim = (
    recent_history.groupby('calendar_month')[weather_metric_cols].mean()
    .reset_index().rename(columns={c: c + '_nclim' for c in weather_metric_cols})
)
price_baseline = (
    history[history['date'] >= last_actual_date - pd.DateOffset(years=1)]
    .groupby('calendar_month')[price_cols].mean()
    .reset_index().rename(columns={c: c + '_pbase' for c in price_cols})
)

future = future.merge(county_clim, on=['FIPS', 'calendar_month'], how='left')
future = future.merge(national_clim, on='calendar_month', how='left')
future = future.merge(price_baseline, on='calendar_month', how='left')

for c in weather_metric_cols:
    future[c] = future[c].fillna(future[c + '_cclim']).fillna(future[c + '_nclim'])
for c in price_cols:
    future[c] = future[c].fillna(future[c + '_pbase'])

future = future.drop(columns=[c + '_cclim' for c in weather_metric_cols]
                      + [c + '_nclim' for c in weather_metric_cols]
                      + [c + '_pbase' for c in price_cols])

fact = pd.concat([history, future], ignore_index=True)

# Crop acreage moves slowly year to year, so a climatology average isn't the right technique -
# carry forward each county's most recent ACTUAL acreage value instead. Sort so real (history)
# values come before estimated (future) rows within each FIPS+Family group, then forward-fill.
acreage_cols = ['CORN_Acres_prior_year', 'COTTON_Acres_prior_year',
                'SOYBEANS_Acres_prior_year', 'WHEAT_Acres_prior_year']
fact = fact.sort_values(['FIPS', 'Family', 'date'])
fact[acreage_cols] = fact.groupby(['FIPS', 'Family'])[acreage_cols].ffill()

# has_temperature_data is an internal data-quality flag (was this row's weather a real station
# reading?), not a value to estimate - drop it rather than fill it with a misleading 1/0.
fact = fact.drop(columns=['has_temperature_data'])
remaining_null = fact.loc[fact['date'] > last_actual_date, 'TAVG'].isna().sum()
print(f"Future rows still missing weather after climatology fill: {remaining_null} "
      f"(no station history available in either family's file for these counties)")

print("\n" + "=" * 80)
print("STEP 7: FINAL VALIDATION")
print("=" * 80)

print(f"\nFact table shape: {fact.shape}")
print(f"Unique FIPS: {fact['FIPS'].nunique()}")
print(f"Unique dates: {fact['date'].nunique()}")
print(f"Unique families: {fact['Family'].nunique()}")
print(f"Date range: {fact['date'].min()} to {fact['date'].max()}")

print("\nNull counts (top 10):")
null_counts = fact.isnull().sum().sort_values(ascending=False).head(10)
for col, count in null_counts.items():
    pct = count / len(fact) * 100
    print(f"  {col}: {count} ({pct:.1f}%)")

print("\n" + "=" * 80)
print("STEP 8: SAVE TO S3")
print("=" * 80)

# Build clean national table with historical + forecast
fore_national = forecast.groupby(['date', 'Family']).agg({
    'Predicted_QTY': 'sum'
}).reset_index()
fore_national = fore_national.rename(columns={'Predicted_QTY': 'Sales_QTY'})
fore_national['DataType'] = 'Forecast'

# Combine historical + forecast into single clean table
national_complete = pd.concat([hist_national, fore_national], ignore_index=True)
national_complete = national_complete.sort_values(['Family', 'date'])
national_complete['date'] = national_complete['date'].dt.strftime('%Y-%m-%d')
print(f"National table (historical + forecast): {len(national_complete)} rows")

fact['date'] = fact['date'].dt.strftime('%Y-%m-%d')  # consistent, unambiguous export format

fact.to_csv('/tmp/fact_table.csv', index=False)
s3.upload_file('/tmp/fact_table.csv', BUCKET, 'dashboard/fact_table.csv')
print(f"Uploaded fact_table.csv ({len(fact)} rows)")

# Ensure date is in YYYY-MM-DD format for QuickSight
if pd.api.types.is_datetime64_any_dtype(national_complete['date']):
    national_complete['date'] = national_complete['date'].dt.strftime('%Y-%m-%d')
national_complete.to_csv('/tmp/national_trailing.csv', index=False)
s3.upload_file('/tmp/national_trailing.csv', BUCKET, 'dashboard/national_trailing.csv')
print(f"Uploaded national_trailing.csv ({len(national_complete)} rows)")

print("\n" + "=" * 80)
print("STEP 9: BUILD feature_importance_long.csv (Feature Importance dashboard tab)")
print("=" * 80)
# Reshapes the wide Top1-5_Feature/Top1-5_SHAP columns into one row per feature rank --
# this is the format QuickSight needs to plot a ranked bar chart of feature importance,
# filterable by the same State/County_Name/Family/date/ModelRole controls already used
# on the main forecast sheet. Only Forecast rows carry real SHAP values (historical/
# validation rows never had a live SHAP computation run against them).
shap_source = fact[fact["DataType"] == "Forecast"].copy()
shap_id_cols = ["FIPS", "County_Name", "State", "date", "Family", "ModelRole", "ModelName",
                "External_Factors_Share_Pct"]

long_chunks = []
for rank in range(1, 6):
    feat_col, shap_col = f"Top{rank}_Feature", f"Top{rank}_SHAP"
    if feat_col not in shap_source.columns or shap_col not in shap_source.columns:
        continue
    chunk = shap_source[shap_id_cols + [feat_col, shap_col]].copy()
    chunk = chunk.rename(columns={feat_col: "Feature", shap_col: "SHAP_Value"})
    chunk["Rank"] = rank
    long_chunks.append(chunk)

feature_importance_long = pd.concat(long_chunks, ignore_index=True)
# BUG FIX: Top1-5_Feature is an empty string ('') for months where per-county SHAP
# wasn't computed (every month except each county's first forecast month, per direct
# instruction on cost -- see repack_evaluation.py), not NaN -- .dropna() alone doesn't
# catch an empty string, so this previously let blank/0-value rows through as if they
# were real. Now that per-county SHAP is genuinely computed for October 2025 (see
# repack_evaluation.py), this correctly keeps only those real rows.
feature_importance_long = feature_importance_long[feature_importance_long["Feature"].notna()
                                                    & (feature_importance_long["Feature"] != "")]
feature_importance_long = feature_importance_long.sort_values(["FIPS", "date", "Rank"]).reset_index(drop=True)
print(f"Reshaped {len(shap_source)} wide forecast rows -> {len(feature_importance_long)} long feature-importance rows")

feature_importance_long.to_csv('/tmp/feature_importance_long.csv', index=False)
s3.upload_file('/tmp/feature_importance_long.csv', BUCKET, 'dashboard/feature_importance_long.csv')
print(f"Uploaded feature_importance_long.csv ({len(feature_importance_long)} rows)")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
print(f"\nQuickSight tables ready:")
print(f"  - s3://{BUCKET}/dashboard/fact_table.csv")
print(f"  - s3://{BUCKET}/dashboard/national_trailing.csv")
print(f"  - s3://{BUCKET}/dashboard/feature_importance_long.csv")

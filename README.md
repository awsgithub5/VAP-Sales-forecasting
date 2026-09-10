# VAP Sales Forecasting Pipeline

Machine learning pipeline for forecasting VAP (Value-Added Product) sales across US counties for Barrage and Grounded product families.

## Overview

- **Models**: 4 models (Barrage/Grounded × Champion/Challenger)
  - Barrage Champion: XGBoost
  - Barrage Challenger: Ridge
  - Grounded Champion: LightGBM
  - Grounded Challenger: Ridge
- **Validation Period**: Oct 2024 - Sep 2025
- **Forecast Period**: Oct 2025 - Sep 2027
- **Quality Gate**: R² ≥ 0.7 for model registration

## Project Structure

```
VAPforecasting2/
├── pipeline.py                  # SageMaker pipeline definition
├── preprocess.py                # Data preprocessing
├── train_with_holdout.py        # Model training with validation
├── repack_evaluation.py         # Extract evaluation metrics
├── evaluate.py                  # Combine all evaluations
├── merge_predictions_step.py    # Merge validation predictions
├── inference.py                 # Endpoint inference logic
├── deploy.py                    # Endpoint deployment
├── dashboard_merge.py           # Dashboard data preparation
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Requirements

- Python 3.8+ (training container uses 3.8)
- AWS SageMaker access
- S3 bucket: `vap-sales-forecasting`

## Dependencies

```
xgboost==1.7.6
lightgbm==4.1.0
shap==0.42.1
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
```

## Quick Start

### 1. Train Models
```bash
cd /home/sagemaker-user/VAPforecasting2
python pipeline.py
```

Pipeline takes ~30-45 minutes and runs:
1. Data preprocessing
2. Train 4 models (with holdout validation)
3. Extract and evaluate metrics
4. Register models if R² ≥ 0.7

### 2. Deploy Endpoint
```bash
cd /home/sagemaker-user/VAPforecasting2
python deploy.py
```

Deployment takes ~5-10 minutes and:
1. Downloads 4 approved models
2. Bundles them with inference code
3. Deploys to endpoint: `VAPSales-endpoint`

### 3. Prepare Dashboard Data
```bash
cd /home/sagemaker-user/VAPforecasting2
python dashboard_merge.py
```

Creates QuickSight-ready tables in S3.

## Model Registry

Models are registered in 4 groups:
- `vap-barrage-champion`
- `vap-barrage-challenger`
- `vap-grounded-champion`
- `vap-grounded-challenger`

## Endpoint Usage

**Request:**
```json
{
  "family": "Barrage",
  "fips": "04013",
  "month": 10,
  "year": 2025
}
```

**Response:**
```json
{
  "family": "Barrage",
  "fips": "04013",
  "month": 10,
  "year": 2025,
  "champion": {"model_name": "XGBoost", "predicted_qty": 116.2},
  "challenger": {"model_name": "Ridge", "predicted_qty": 108.9},
  "feature_source": "model_forecast_recursive_guardrailed"
}
```

## S3 Structure

```
s3://vap-sales-forecasting/
├── raw-data/                    # Input data
├── pipeline-output/             # Training outputs
│   ├── preprocess/
│   ├── train/
│   └── evaluations/
├── models/combined/             # Bundled endpoint model
├── lookup/                      # Feature lookup tables
├── dashboard/                   # Dashboard data
└── comparison/                  # Forecast comparisons
```

## Monitoring

Check pipeline status:
```bash
aws sagemaker list-pipeline-executions \
  --pipeline-name vap-forecast-pipeline-v4 \
  --region us-east-1 \
  --max-results 1
```

Check endpoint status:
```bash
aws sagemaker describe-endpoint \
  --endpoint-name VAPSales-endpoint \
  --region us-east-1
```

## Troubleshooting

**Pipeline fails**: Check CloudWatch logs for the failed step
**Endpoint fails**: Verify requirements.txt matches training environment
**Low R²**: Check data quality and feature engineering



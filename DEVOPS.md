# VAP Sales Forecasting Pipeline - DevOps Guide

## Overview
ML pipeline for training and deploying 4 sales forecasting models (Barrage/Grounded × Champion/Challenger) using AWS SageMaker.

## Architecture

### Pipeline Flow
1. **PreprocessData** - Cleans and prepares raw data
2. **Train (4 steps)** - Trains models with validation holdout (Scenario 1)
3. **Repack (4 steps)** - Retrains on full data (Scenario 2) + generates forecasts
4. **EvaluateAllModels** - Aggregates metrics and validates quality
5. **MergeAndGenerateDashboard** - Combines forecasts + generates dashboard files
6. **Conditional Registration (4 steps)** - Registers models meeting R² ≥ 0.70 threshold

### Model Registry Groups
- `vap-barrage-champion`
- `vap-barrage-challenger`
- `vap-grounded-champion`
- `vap-grounded-challenger`

---

## S3 Bucket Structure

```
s3://vap-sales-forecasting/
│
├── raw-data/                          # Input data (Excel/CSV files)
│   └── [uploaded by data team]
│
├── pipeline-output/
│   ├── preprocess/                    # Cleaned training data
│   │   └── train/
│   │
│   ├── train/                         # Scenario 1 models (with holdout)
│   │   ├── barrage-champion/
│   │   │   └── model.tar.gz
│   │   ├── barrage-challenger/
│   │   ├── grounded-champion/
│   │   └── grounded-challenger/
│   │
│   ├── scenario2-models/              # Scenario 2 models (full data, for deployment)
│   │   ├── barrage-champion/
│   │   │   └── model.tar.gz
│   │   ├── barrage-challenger/
│   │   ├── grounded-champion/
│   │   └── grounded-challenger/
│   │
│   ├── evaluations/                   # Validation metrics (Scenario 1)
│   │   ├── barrage-champion/
│   │   │   └── evaluation.json
│   │   └── [other models]/
│   │
│   ├── final-evaluation/              # Aggregated metrics
│   │   └── evaluation.json
│   │
│   ├── quality-check/                 # Quality gate results
│   │   └── quality_check.json
│   │
│   └── merge-summary/                 # Dashboard generation logs
│
├── comparison/
│   ├── scenario2-chunks/              # Individual model forecasts
│   │   ├── barrage-champion/
│   │   │   └── forecast_chunk.csv
│   │   └── [other models]/
│   │
│   ├── scenario2-importance/          # Feature importance per model
│   │   └── [family-role]/
│   │       └── importance_chunk.csv
│   │
│   ├── forecast_2yr_all_models.csv    # Combined 2-year forecast
│   ├── county_accuracy.csv            # County-level accuracy metrics
│   └── feature_importance_global.csv  # Global feature importance
│
├── dashboard/                         # QuickSight dashboard files
│   ├── fact_forecast.csv
│   ├── dim_county.csv
│   ├── dim_date.csv
│   └── dim_model.csv
│
├── models/
│   └── combined/                      # Deployed endpoint model bundle
│       └── model.tar.gz               # All 4 approved models + inference.py
│
└── pipeline-artifacts/                # Code dependencies
    ├── requirements.txt
    ├── inference.py
    ├── repack_code_deps.tar.gz
    └── merge_code_deps.tar.gz
```

---

## Prerequisites

### AWS Resources Required
1. **IAM Role**: SageMaker execution role with permissions:
   - S3 read/write to bucket
   - SageMaker full access
   - VPC access (EC2 network interfaces)
   - Secrets Manager (optional)

2. **VPC Configuration**:
   - 2+ subnets (private recommended)
   - Security group allowing outbound HTTPS
   - NAT Gateway for internet access

3. **S3 Bucket**: `vap-sales-forecasting` (or custom name)

4. **Model Registry**: Auto-created on first pipeline run

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/awsgithub5/VAP-Sales-forecasting.git
cd VAP-Sales-forecasting
```

### 2. Configure AWS Resources
Copy the example config and fill in your AWS details:
```bash
cp config.json.example config.json
```

Edit `config.json`:
```json
{
  "region": "us-east-1",
  "s3_bucket": "your-bucket-name",
  "endpoint_name": "VAPSales-endpoint",
  "role_arn": "arn:aws:iam::ACCOUNT_ID:role/YOUR_ROLE",
  "vpc_subnets": ["subnet-xxxxx", "subnet-yyyyy"],
  "vpc_security_groups": ["sg-xxxxx"]
}
```

**IMPORTANT**: `config.json` is gitignored and contains sensitive data. Never commit this file.

### 3. Upload Raw Data
```bash
aws s3 cp your-data.xlsx s3://your-bucket-name/raw-data/
```

### 4. Install Dependencies (if running locally)
```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Option 1: From SageMaker Studio/Notebook
```python
python pipeline_vpc.py
```

### Option 2: Via AWS CLI
```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name vap-forecast-pipeline-v4 \
  --region us-east-1
```

### Monitor Execution
```bash
# Get latest execution ARN
aws sagemaker list-pipeline-executions \
  --pipeline-name vap-forecast-pipeline-v4 \
  --max-results 1

# Check status
aws sagemaker describe-pipeline-execution \
  --pipeline-execution-arn <ARN>
```

---

## Model Deployment

### 1. Approve Models in Registry
After pipeline completes successfully:
```bash
# List pending models
aws sagemaker list-model-packages \
  --model-package-group-name vap-barrage-champion \
  --model-approval-status PendingManualApproval

# Approve a model
aws sagemaker update-model-package \
  --model-package-arn <ARN> \
  --model-approval-status Approved
```

Or via Console: SageMaker → Model Registry → Select model → Update status → Approved

### 2. Deploy to Endpoint
```bash
python deploy.py
```

This script:
- Bundles all 4 approved models
- Creates/updates endpoint with zero-downtime (blue/green)
- Endpoint: `VAPSales-endpoint` on `ml.m5.xlarge`

### 3. Test Endpoint
```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')
response = runtime.invoke_endpoint(
    EndpointName='VAPSales-endpoint',
    ContentType='application/json',
    Body=json.dumps({
        'family': 'Barrage',
        'role': 'champion',
        'fips': '01001',
        'forecast_months': 24
    })
)
print(json.loads(response['Body'].read()))
```

---

## Environment Variables (Optional)

Override defaults via environment variables:
```bash
export VAP_REGION="us-west-2"
export VAP_S3_BUCKET="my-custom-bucket"
export VAP_MIN_R2_THRESHOLD="0.75"
export VAP_TRAIN_INSTANCE_TYPE="ml.m5.2xlarge"
export VAP_REPACK_INSTANCE_TYPE="ml.c5.9xlarge"
```

---

## Compute Resources

| Step | Instance Type | vCPUs | Memory | Notes |
|------|---------------|-------|--------|-------|
| Preprocess | ml.m5.large | 2 | 8 GB | Lightweight data cleaning |
| Train | ml.m5.xlarge | 4 | 16 GB | Model training with holdout |
| Repack | ml.c5.4xlarge | 16 | 32 GB | Full retrain + forecast generation |
| Evaluate | ml.m5.large | 2 | 8 GB | Metric aggregation only |
| Merge | ml.m5.xlarge | 4 | 16 GB | Dashboard file generation |
| Endpoint | ml.m5.xlarge | 4 | 16 GB | Real-time inference |

**Cost Optimization**: Adjust instance types via environment variables based on data volume.

---

## Troubleshooting

### Pipeline Fails at Training Step
- Check S3 bucket has preprocessed data: `s3://bucket/pipeline-output/preprocess/train/`
- Verify IAM role has S3 read permissions
- Check CloudWatch logs: `/aws/sagemaker/TrainingJobs`

### Model Not Registered
- Check R² score in `s3://bucket/pipeline-output/final-evaluation/evaluation.json`
- Ensure R² ≥ 0.70 threshold (configurable via `VAP_MIN_R2_THRESHOLD`)

### Deployment Fails
- Verify at least one model is Approved in each registry group
- Check VPC security group allows outbound HTTPS (443)
- Review CloudWatch logs: `/aws/sagemaker/Endpoints/VAPSales-endpoint`

### Dashboard Files Missing
- Check `s3://bucket/comparison/forecast_2yr_all_models.csv` exists
- Verify all 4 Repack steps completed successfully
- Review logs: `s3://bucket/pipeline-output/merge-summary/`

---

## Security Best Practices

1. **Never commit `config.json`** - Contains AWS account IDs and resource ARNs
2. **Use IAM roles** - No hardcoded credentials
3. **Enable S3 bucket encryption** - Server-side encryption (SSE-S3 or SSE-KMS)
4. **VPC isolation** - Run pipeline in private subnets
5. **Least privilege** - Grant only required IAM permissions
6. **Secrets Manager** (optional) - Store config in AWS Secrets Manager instead of local file

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Deploy Pipeline
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v1
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: us-east-1
      - name: Update Pipeline
        run: python pipeline_vpc.py
```

---

## Support

- **Pipeline Issues**: Check CloudWatch Logs → `/aws/sagemaker/ProcessingJobs`
- **Model Performance**: Review `evaluation.json` in S3
- **Endpoint Issues**: CloudWatch Logs → `/aws/sagemaker/Endpoints`

## License
Proprietary - Internal Use Only

# VAP Sales Forecasting Pipeline

ML pipeline for training and deploying sales forecasting models.

## Setup

1. Copy `config.json.example` to `config.json`
2. Update `config.json` with your AWS account details:
   - `role_arn`: Your SageMaker execution role
   - `vpc_subnets`: Your VPC subnet IDs
   - `vpc_security_groups`: Your security group IDs
   - `s3_bucket`: Your S3 bucket name

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline:
```bash
python pipeline_vpc.py
```

Deploy models:
```bash
python deploy.py
```

**Note:** `config.json` is gitignored and contains sensitive AWS resource IDs. Never commit this file.

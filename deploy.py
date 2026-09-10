"""VAP Model Deployment - Blue/Green Updates

Deploys approved models from Model Registry to inference endpoint.
Supports zero-downtime updates via blue/green deployment strategy.
"""
import boto3
import sagemaker
import os
import tarfile
import tempfile
import shutil
from config import get_config

# Load configuration from Secrets Manager
config = get_config()
REGION = config['region']
BUCKET = config['s3_bucket']
ENDPOINT_NAME = config['endpoint_name']
ROLE_ARN = config['role_arn']

ALL_GROUPS = ['vap-barrage-champion', 'vap-barrage-challenger', 'vap-grounded-champion', 'vap-grounded-challenger']

VPC_CONFIG = {
    "Subnets": config['vpc_subnets'],
    "SecurityGroupIds": config['vpc_security_groups'],
}

sm = boto3.client('sagemaker', region_name=REGION)
s3 = boto3.client('s3', region_name=REGION)

print("\nStep 2: Bundling all 4 models...")
tmpdir = tempfile.mkdtemp()
bundle_dir = os.path.join(tmpdir, 'bundle')
os.makedirs(bundle_dir)

for group in ALL_GROUPS:
    response = sm.list_model_packages(
        ModelPackageGroupName=group,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    if not response['ModelPackageSummaryList']:
        raise RuntimeError(
            f"'{group}' has NO approved model package at all -- nothing to deploy for this "
            f"group. Approve at least one version in the Model Registry before running this "
            f"script, or this endpoint would be missing a required model file."
        )

    summary = response['ModelPackageSummaryList'][0]
    pkg = sm.describe_model_package(ModelPackageName=summary['ModelPackageArn'])
    s3_uri = pkg['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    bucket = s3_uri.split('/')[2]
    key = '/'.join(s3_uri.split('/')[3:])
    local_tar = os.path.join(tmpdir, f"{group}.tar.gz")
    s3.download_file(bucket, key, local_tar)
    with tarfile.open(local_tar, 'r:gz') as tar:
        tar.extractall(bundle_dir, filter='data')
    print(f"  \u2713 {group}: using version approved on {summary['CreationTime']:%Y-%m-%d %H:%M} "
          f"({summary['ModelPackageArn'].split('/')[-1]})")


code_dir = os.path.join(bundle_dir, 'code')
os.makedirs(code_dir, exist_ok=True)
shutil.copy('inference.py', code_dir)


if os.path.exists(os.path.join(bundle_dir, 'requirements.txt')):
    shutil.copy(os.path.join(bundle_dir, 'requirements.txt'), code_dir)
    print("  ✓ Using requirements.txt from trained model")
else:
    shutil.copy('requirements.txt', code_dir)
    print("  ✓ Using local requirements.txt")

combined_tar = os.path.join(tmpdir, 'model.tar.gz')
with tarfile.open(combined_tar, 'w:gz') as tar:
    tar.add(bundle_dir, arcname='.')

s3.upload_file(combined_tar, BUCKET, 'models/combined/model.tar.gz')
shutil.rmtree(tmpdir)
print(f"  ✓ Uploaded combined model")


image_uri = '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3'

import time

print("\nStep 3: Deploying endpoint...")


model_name = f'vap-model-{int(time.time())}'
sm.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': image_uri,
        'ModelDataUrl': f's3://{BUCKET}/models/combined/model.tar.gz',
        'Environment': {
            'SAGEMAKER_PROGRAM': 'inference.py',
            'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
            'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
            'SAGEMAKER_REGION': REGION
        }
    },
    ExecutionRoleArn=ROLE_ARN,
    VpcConfig=VPC_CONFIG
)
print(f"  ✓ Created model: {model_name}")


try:
    sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    endpoint_exists = True
    print(f"  ✓ Endpoint exists, using blue/green update")
except:
    endpoint_exists = False
    print(f"  ✓ Endpoint doesn't exist, creating new")

if endpoint_exists:

    new_config_name = f'{ENDPOINT_NAME}-{int(time.time())}'
    sm.create_endpoint_config(
        EndpointConfigName=new_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.xlarge'
        }]
    )
    print(f"  ✓ Created new endpoint config: {new_config_name}")
    
    sm.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=new_config_name
    )
    print(f"  ✓ Updating endpoint (blue/green swap)...")
    
    waiter = sm.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=ENDPOINT_NAME)
    print(f"  ✓ Endpoint updated with zero downtime")
else:

    sm.create_endpoint_config(
        EndpointConfigName=ENDPOINT_NAME,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.xlarge'
        }]
    )
    print(f"  ✓ Created endpoint config")
    
    sm.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_NAME
    )
    print(f"  ✓ Creating endpoint...")
    
    waiter = sm.get_waiter('endpoint_in_service')
    print("  Waiting for endpoint to be InService (this may take 5-10 minutes)...")
    waiter.wait(EndpointName=ENDPOINT_NAME)

print(f"\n✓ Endpoint {ENDPOINT_NAME} deployed successfully!")

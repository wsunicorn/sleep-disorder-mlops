"""
Training — Launch AWS SageMaker Training Job.
Chạy script này từ local để kick off SageMaker training.
"""

import os
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET", "sleep-mlops-data")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
SAGEMAKER_ROLE = os.getenv("SAGEMAKER_ROLE_ARN")  # IAM role ARN


def launch_training_job(
    model_type: str = "xgboost",
    instance_type: str = "ml.m5.xlarge",
):
    """Launch SageMaker training job."""
    session = sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))
    job_name = f"sleep-{model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    logger.info(f"Starting SageMaker job: {job_name}")
    logger.info(f"Instance: {instance_type}")

    estimator = SKLearn(
        entry_point="train.py",
        source_dir="training/",
        role=SAGEMAKER_ROLE,
        instance_count=1,
        instance_type=instance_type,
        framework_version="1.2-1",
        py_version="py3",
        job_name=job_name,
        hyperparameters={
            "model-type": model_type,
            "test-size": "0.2",
        },
        environment={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        },
        sagemaker_session=session,
    )

    data_uri = f"s3://{S3_BUCKET}/features/"
    estimator.fit({"train": data_uri}, wait=False)

    logger.info(f"Job submitted: {job_name}")
    logger.info(f"Monitor at: https://{AWS_REGION}.console.aws.amazon.com/sagemaker")
    return job_name


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="xgboost")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    args = parser.parse_args()

    launch_training_job(args.model_type, args.instance_type)

from .utils import setup_aws_session
import boto3
from sagemaker.s3 import S3Uploader, S3Downloader

def check_resource_on_s3(bucket_name, resource_path):
    """Check if a resource exists on S3.""" 
    setup_aws_session()
    s3 = boto3.client('s3')

    try:
        s3.head_object(Bucket=bucket_name, Key=resource_path)
        return True
    except Exception as e:
        print(f"Resource {resource_path} not found on S3: {e}")
        return False

def upload_datasets_to_s3(bucket_name):
    setup_aws_session()

    try:
        # Upload to S3
        s3_train_path = S3Uploader.upload(
            "data/preprocessed/train_processed", f"s3://{bucket_name}/text-classifier/train"
        )
        s3_test_path = S3Uploader.upload(
            "data/preprocessed/test_processed", f"s3://{bucket_name}/text-classifier/test"
        )
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return

    print(f"Train data uploaded to: {s3_train_path}")
    print(f"Test data uploaded to: {s3_test_path}")

def download_datasets_from_s3(bucket_name):
    setup_aws_session()

    try:
        S3Downloader.download(f"s3://{bucket_name}/text-classifier/train", './data/preprocessed/train_processed')
        S3Downloader.download(f"s3://{bucket_name}/text-classifier/test", './data/preprocessed/test_processed')
    except Exception as e:
        print(f"Error downloading from S3: {e}")

def load_datasets_from_s3(bucket_name):
    setup_aws_session()

    try:
        s3 = boto3.client('s3')

        train_dataset = s3.get_object(Bucket=bucket_name, Key="text-classifier/train")['Body'].read()
        test_dataset = s3.get_object(Bucket=bucket_name, Key="text-classifier/test")['Body'].read()
    except Exception as e:
        print(f"Error loading from S3: {e}")

    return train_dataset, test_dataset

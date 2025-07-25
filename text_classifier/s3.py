import dotenv

import boto3
from sagemaker.s3 import S3Uploader, S3Downloader

def setup_aws_session():
    try:
        aws_access_key_id = dotenv.get_key(dotenv.find_dotenv(), "ACCESS_KEY")
        aws_secret_access_key = dotenv.get_key(dotenv.find_dotenv(), "SECRET_KEY")
        region_name = (
            dotenv.get_key(dotenv.find_dotenv(), "AWS_REGION") or "us-west-2"
        )  # Default to us-west-2

        # Configure the default session
        boto3.setup_default_session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
    except Exception as e:
        print(f"Error setting up AWS session: {e}")

def upload_datasets_to_s3(s3_bucket):
    setup_aws_session()

    try:
        # Upload to S3
        s3_train_path = S3Uploader.upload(
            "data/preprocessed/train_processed", f"s3://{s3_bucket}/text-classifier/train"
        )
        s3_test_path = S3Uploader.upload(
            "data/preprocessed/test_processed", f"s3://{s3_bucket}/text-classifier/test"
        )
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return

    print(f"Train data uploaded to: {s3_train_path}")
    print(f"Test data uploaded to: {s3_test_path}")

def download_datasets_from_s3(s3_bucket):
    setup_aws_session()

    try:
        S3Downloader.download(f"s3://{s3_bucket}/text-classifier/train", './data/preprocessed/train_processed')
        S3Downloader.download(f"s3://{s3_bucket}/text-classifier/test", './data/preprocessed/test_processed')
    except Exception as e:
        print(f"Error downloading from S3: {e}")

def load_datasets_from_s3(s3_bucket):
    setup_aws_session()

    try:
        s3 = boto3.client('s3')

        train_dataset = s3.get_object(Bucket=s3_bucket, Key="text-classifier/train")['Body'].read()
        test_dataset = s3.get_object(Bucket=s3_bucket, Key="text-classifier/test")['Body'].read()
    except Exception as e:
        print(f"Error loading from S3: {e}")

    return train_dataset, test_dataset

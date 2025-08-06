from .utils import setup_aws_session
import boto3
from sagemaker.s3 import S3Uploader

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

def upload_dataset_to_s3(dataset, local_dir, s3_path):
    setup_aws_session()

    dataset.save_to_disk(local_dir)

    try:
        s3_path = S3Uploader.upload(
            local_dir, 
            s3_path
        )
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return

    print(f"Data uploaded to: {s3_path}")

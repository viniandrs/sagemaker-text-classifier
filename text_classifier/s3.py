from .utils import setup_aws_session
import boto3
import os
from sagemaker.s3 import S3Uploader

def check_resource_on_s3(bucket_name, resource_path):
    """Check if a resource exists on S3.""" 
    setup_aws_session()
    s3 = boto3.client('s3')

    try:
        s3.head_object(Bucket=bucket_name, Key=resource_path)
        return True
    except Exception as e:
        print(f"Resource {resource_path} not found on S3")
        return False

def upload_dataset_to_s3(dataset, local_dir, s3_path):
    setup_aws_session()

    dataset.save_to_disk(local_dir)

    # Verify local files were created
    required_files = {"dataset_info.json", "state.json"}
    saved_files = set(os.listdir(local_dir))
    if not required_files.issubset(saved_files):
        missing = required_files - saved_files
        raise ValueError(f"Missing dataset files: {missing}")

    try:
        # Upload each file individually for reliability
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                S3Uploader.upload(local_path, s3_path)
        
        print(f"Successfully uploaded to: {s3_path}")
        print(f"Files uploaded: {os.listdir(local_dir)}")

    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return

    print(f"Data uploaded to: {s3_path}")

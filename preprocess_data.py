#!/usr/bin/env python3
import click
from text_classifier.s3 import check_resource_on_s3, upload_datasets_to_s3
from text_classifier.utils import tokenize_and_save_datasets


@click.command()
@click.option(
    "--padding_length", type=int, default=256, help="Padding length for tokenization"
)
@click.option(
    "--s3_bucket", type=str, required=False, help="S3 bucket to upload processed data"
)
def main(padding_length, s3_bucket):
    assert s3_bucket is not None, "S3 bucket must be provided for uploading datasets."
    
    if check_resource_on_s3("data/preprocessed/train") and check_resource_on_s3("data/preprocessed/test"):
        print("Train and test datasets already exists on S3. Nothing to do.")
        return
    
    tokenize_and_save_datasets(padding_length)
    upload_datasets_to_s3(s3_bucket)

if __name__ == "__main__":
    main()

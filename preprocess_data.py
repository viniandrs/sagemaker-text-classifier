#!/usr/bin/env python3
import os
import click
import tempfile

@click.command()
@click.option(
    "--padding_length", type=int, default=256, help="Padding length for tokenization"
)
@click.option(
    "--s3-bucket", type=str, required=True, help="S3 bucket to upload processed data"
)
def main(padding_length, s3_bucket):
    assert s3_bucket is not None, "S3 bucket must be provided for uploading datasets."

    from text_classifier.s3 import check_resource_on_s3, upload_dataset_to_s3

    if check_resource_on_s3("firstmlopsproj", "data/preprocessed/train") and \
       check_resource_on_s3("firstmlopsproj", "data/preprocessed/test"):
        print("Train and test datasets already exists on S3. Nothing to do.")
        return
    
    from text_classifier.utils import load_and_tokenize_datasets

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Using temp dir: {tmp_dir}")
            
        train_dataset, test_dataset = load_and_tokenize_datasets(padding_length)

        upload_dataset_to_s3(
            train_dataset, 
            os.path.join(tmp_dir, "train"), 
            f"s3://{s3_bucket}/data/train"
        )
        upload_dataset_to_s3(
            test_dataset, 
            os.path.join(tmp_dir, "test"), 
            f"s3://{s3_bucket}/data/test"
        )

    print("Data preprocessing and upload completed successfully.")

@click.command("replace")
@click.option("--s3-bucket", type=str, required=True, help="S3 bucket to upload processed data")
def replace(s3_bucket):
    assert s3_bucket is not None, "S3 bucket must be provided for uploading datasets."

    from text_classifier.s3 import check_resource_on_s3, upload_dataset_to_s3

    if check_resource_on_s3("firstmlopsproj", "data/preprocessed/train") and \
       check_resource_on_s3("firstmlopsproj", "data/preprocessed/test"):
        print("Train and test datasets already exists on S3. Nothing to do.")
        return
    

if __name__ == "__main__":
    main()

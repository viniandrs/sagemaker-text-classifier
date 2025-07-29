#!/usr/bin/env python3
import dotenv
import click


@click.command()
@click.option(
    "--padding_length", type=int, default=256, help="Padding length for tokenization"
)
@click.option(
    "--s3_bucket", type=str, required=False, help="S3 bucket to upload processed data"
)
def main(padding_length, s3_bucket):
    from datasets import load_from_disk

    try:
        load_from_disk("data/preprocessed/train")
        load_from_disk("data/preprocessed/test")
        print("Train and test datasets already exists. Skipping preprocessing.")
    except FileNotFoundError:
        tokenize_and_save_datasets(padding_length)

    if not s3_bucket:
        print("No S3 bucket specified. Skipping upload.")
        return

    from text_classifier.utils import tokenize_and_save_datasets
    from text_classifier.s3 import upload_datasets_to_s3

    upload_datasets_to_s3(s3_bucket)


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()

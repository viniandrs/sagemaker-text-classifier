#!/usr/bin/env python3
import dotenv
import click

@click.command()
@click.option("--padding_length", type=int, default=256, help="Padding length for tokenization")
@click.option("--s3_bucket", type=str, required=False, help="S3 bucket to upload processed data")
def main(padding_length, s3_bucket):    
    from datasets import load_dataset, load_from_disk
    
    try:
        load_from_disk("data/preprocessed/train_processed")
        load_from_disk("data/preprocessed/test_processed")
        print("Train and test datasets already exists. Skipping preprocessing.")
    except FileNotFoundError:
        from transformers import AutoTokenizer

        dataset = load_dataset("imdb")
        train_data, test_data = dataset["train"], dataset["test"]

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        def tokenize(batch):
            return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=padding_length)

        # Tokenize datasets
        train_dataset = train_data.map(tokenize, batched=True)
        test_dataset = test_data.map(tokenize, batched=True)

        # Save processed datasets to disk
        train_dataset.save_to_disk("data/preprocessed/train_processed")
        test_dataset.save_to_disk("data/preprocessed/test_processed")
        print("Train and test datasets processed and saved to disk.")

    if not s3_bucket:
        print("No S3 bucket specified. Skipping upload.")
        return
    
    import boto3
    from sagemaker.s3 import S3Uploader

    aws_access_key_id = dotenv.get_key(dotenv.find_dotenv(), "ACCESS_KEY")  
    aws_secret_access_key = dotenv.get_key(dotenv.find_dotenv(), "SECRET_KEY") 
    region_name = dotenv.get_key(dotenv.find_dotenv(), "AWS_REGION") or "us-west-2"  # Default to us-west-2  

    # Configure the default session
    boto3.setup_default_session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )     

    # Upload to S3
    s3_train_path = S3Uploader.upload("data/preprocessed/train_processed", f"s3://{s3_bucket}/text-classifier/train")
    s3_test_path = S3Uploader.upload("data/preprocessed/test_processed", f"s3://{s3_bucket}/text-classifier/test")

    print(f"Train data uploaded to: {s3_train_path}")
    print(f"Test data uploaded to: {s3_test_path}")

if __name__ == "__main__":
    dotenv.load_dotenv()
    main()
import dotenv
import boto3
from transformers import AutoTokenizer
from datasets import load_dataset

def tokenize_and_save_datasets(padding_length=256):
    dataset = load_dataset("imdb")
    train_data, test_data = dataset["train"], dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=padding_length,
        )

    # Tokenize datasets
    train_dataset = train_data.map(tokenize, batched=True)
    test_dataset = test_data.map(tokenize, batched=True)

    # Save processed datasets to disk
    train_dataset.save_to_disk("data/preprocessed/train_processed")
    test_dataset.save_to_disk("data/preprocessed/test_processed")
    print("Train and test datasets processed and saved to disk.")

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
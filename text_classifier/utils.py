import dotenv
import boto3
from transformers import AutoTokenizer
from datasets import DatasetDict, load_dataset

def load_and_tokenize_datasets(padding_length=256):
    dataset = load_dataset("imdb")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=padding_length,
        )

    # Tokenized dataset dict
    tokenized_dataset = DatasetDict({
        "train": dataset["train"].map(tokenize, batched=True),
        "test": dataset["test"].map(tokenize, batched=True)
    })

    # print("Sample processed item:", tokenized_dataset["train"][0])
    return tokenized_dataset["train"], tokenized_dataset["test"]

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
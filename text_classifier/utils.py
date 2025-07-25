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
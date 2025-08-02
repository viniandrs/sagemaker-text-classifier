import os
import argparse

def train_sagemaker(
    epochs, batch_size, learning_rate, s3_bucket
):
    from transformers import (
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    from datasets import load_from_disk

    try:
        train_dataset = load_from_disk(os.environ["SM_CHANNEL_TRAIN"])
        test_dataset = load_from_disk(os.environ["SM_CHANNEL_TEST"])
    except Exception:
        print("Error loading datasets. Ensure they are available either in S3 or locally.")
        return

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Metrics
    import evaluate
    import numpy as np

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.environ["SM_MODEL_DIR"],  # SageMaker saves model artifacts here
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        logging_dir=f"results/logs",
        logging_steps=100,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train and save model
    trainer.train()
    trainer.save_model("models/fine-tuned")

    return trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sagemaker(**vars(args))
# .venv/bin/python
import click
import dotenv


def train_local(epochs, s3_bucket, batch_size, learning_rate):
    if s3_bucket is not None:
        from text_classifier.s3 import load_datasets_from_s3

        train_dataset, test_dataset = load_datasets_from_s3(s3_bucket)
    else:
        try:
            from datasets import load_from_disk

            train_dataset = load_from_disk("data/preprocessed/train")
            test_dataset = load_from_disk("data/preprocessed/test")
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return

    from transformers import (
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    import numpy as np

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    # Metrics
    import evaluate

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="results",  # SageMaker saves model artifacts here
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        logging_dir="results/logs",
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

    trainer.train()
    trainer.save_model("models/fine_tuned")


def train_sagemaker(epochs, s3_bucket, batch_size, learning_rate):  # TODO
    import sagemaker
    from sagemaker.huggingface import HuggingFace

    estimator = HuggingFace(
        entry_point="text_classifier/train/sagemaker_train.py",
        source_dir=".",
        dependencies=["text_classifier"],  # Include other modules
        instance_type="ml.g4dn.xlarge",
        role=sagemaker.get_execution_role(),
        pytorch_version="1.13",
        transformers_version="4.26",
        py_version="py39",
        hyperparameters={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "s3_bucket": s3_bucket
        },
    )

    if s3_bucket is not None:
        estimator.fit({
            "train": f"s3://{s3_bucket}/data/train",
            "test": f"s3://{s3_bucket}/data/test"
        })
    else:
        estimator.fit({
            "train": "data/preprocessed/train",
            "test": "data/preprocessed/test"
        })


@click.command()
@click.option("--epochs", type=int, default=3, help="Number of training epochs. Default is 3.")
@click.option("--batch-size", type=int, default=8, help="Batch size for training. Default is 8.")
@click.option(
    "--learning-rate", type=float, default=5e-5, help="Learning rate for training. Default is 5e-5."
)
@click.option("--s3-bucket", default=None, help="S3 bucket name for datasets. If not provided, local datasets will be used.")
@click.option(
    "--train-on-local", is_flag=True, help="Train on local machine instead of SageMaker"
)
def main(epochs, batch_size, learning_rate, s3_bucket, train_on_local):
    """Main function to start training."""

    if train_on_local:
        train_local(
            epochs=epochs,
            s3_bucket=s3_bucket,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    else:
        train_sagemaker()


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()

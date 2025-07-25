import click
import dotenv

def train_local(
    model_name, epochs, data_from_s3, train_dir, test_dir, batch_size, learning_rate
):
    from transformers import (
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    import numpy as np

    if data_from_s3:
        from text_classifier.s3 import load_datasets_from_s3
        train_dataset, test_dataset = load_datasets_from_s3(train_dir, test_dir)
    else:
        from datasets import load_from_disk
        train_dataset = load_from_disk("data/preprocessed/train")
        test_dataset = load_from_disk("data/preprocessed/test")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Metrics
    import evaluate
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=dotenv.get("SM_OUTPUT_DATA_DIR"),  # SageMaker saves model artifacts here
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        logging_dir=f"{dotenv.get('SM_OUTPUT_DATA_DIR')}/logs",
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
    trainer.save_model(
        dotenv.get("SM_MODEL_DIR")
    )  # SageMaker compresses this for deployment

def train_sagemaker():
    import sagemaker
    from sagemaker.huggingface import HuggingFace

    estimator = HuggingFace(
        entry_point="text_classifier/train/sagemaker_train.py",
        source_dir="src/train",
        dependencies=["src"],  # Include other modules
        instance_type="ml.g4dn.xlarge",
        role=sagemaker.get_execution_role(),
        pytorch_version="1.13",
        transformers_version="4.26",
        py_version="py39",
        hyperparameters={
            "model_name": "distilbert-base-uncased"
        }
    )

    estimator.fit({
        "train": "s3://firstmlopsproj/data/train",
        "test": "s3://firstmlopsproj/data/test"
    })

@click.command()
@click.option("--epochs", type=int, default=3, help="Number of training epochs")
@click.option("--batch-size", type=int, default=8, help="Batch size for training")
@click.option("--learning-rate", type=float, default=5e-5, help="Learning rate for training")
@click.option("--s3-bucket", default="firstmlopsproj", help="S3 bucket name for datasets")
@click.option("--train-on-local", is_flag=True, help="Train on local machine instead of SageMaker")
def main(epochs, batch_size, learning_rate, s3_bucket, train_on_local):
    """Main function to start training."""
    dotenv.load_dotenv()

    if train_on_local:
        train_local(
            model_name="distilbert-base-uncased",
            epochs=epochs,
            data_from_s3=True,
            train_dir=f"s3://{s3_bucket}/data/train",
            test_dir=f"s3://{s3_bucket}/data/test",
            batch_size=batch_size,
            learning_rate=learning_rate
        )
    else:
        train_sagemaker()


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()

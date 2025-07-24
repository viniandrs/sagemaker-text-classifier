import click
import dotenv

@click.command()
@click.option("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained model name")
@click.option("--epochs", type=int, default=3, help="Number of training epochs")
@click.option("--data-from-s3", is_flag=True, help="Load data from S3 instead of local disk")
@click.option("--train_dir", type=str, default=dotenv.get("SM_CHANNEL_TRAIN", "train"), help="Directory for training data")
@click.option("--test_dir", type=str, default=dotenv.get("SM_CHANNEL_TEST", "test"), help="Directory for test data")
@click.option("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
@click.option("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
def main(model_name, epochs, data_from_s3, train_dir, test_dir, batch_size, learning_rate):
    from datasets import load_from_disk
    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
    import numpy as np
    import evaluate

    if data_from_s3:
        train_dataset = load_from_disk(train_dir)
        test_dataset = load_from_disk(test_dir)
    else:
        train_dataset = load_from_disk("data/preprocessed/train")
        test_dataset = load_from_disk("data/preprocessed/test")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Metrics
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Training arguments
    training_args= TrainingArguments(
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
    trainer.save_model(dotenv.get("SM_MODEL_DIR"))  # SageMaker compresses this for deployment

if __name__ == "__main__":
    dotenv.load_dotenv()
    main()
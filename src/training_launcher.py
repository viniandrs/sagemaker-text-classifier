import logging
import os
import argparse

def train_sagemaker(
    epochs, batch_size, learning_rate
):
    from transformers import (
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments
    )
    from datasets import load_from_disk
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        train_path = os.environ["SM_CHANNEL_TRAIN"]
        test_path = os.environ["SM_CHANNEL_TEST"]
        logger.info(f"Looking for data in:\nTrain: {train_path}\nTest: {test_path}")

        train_dataset = load_from_disk(train_path)
        test_dataset = load_from_disk(test_path)
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise  # Re-raise to see full traceback
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Test dataset: {len(test_dataset)} samples")
    logger.info(f"Sample train item: {train_dataset[0]}")

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Metrics
    import evaluate
    import numpy as np
    from text_classifier.logs import CloudWatchMetricsCallback

    # Define metric for evaluation
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'accuracy': metric.compute(predictions=predictions, references=labels)['accuracy']
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.environ["SM_MODEL_DIR"],  # SageMaker saves model artifacts here
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        logging_dir=os.environ["SM_OUTPUT_DATA_DIR"] + "/logs",  # SageMaker logs here
        logging_steps=10,
        report_to="all",   # Send logs to SageMaker + console
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[CloudWatchMetricsCallback(logger)]
    )

     # Training with enhanced logging
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
        
    trainer.save_model(os.environ["SM_MODEL_DIR"])
    logger.info(f"Model saved to {os.environ['SM_MODEL_DIR']}")

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
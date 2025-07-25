def train(
    epochs, batch_size, learning_rate, s3_bucket
):
    from transformers import (
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    import numpy as np

    if s3_bucket is not None:
        from text_classifier.s3 import load_datasets_from_s3
        train_dataset, test_dataset = load_datasets_from_s3(s3_bucket)
    else:
        from datasets import load_from_disk
        train_dataset = load_from_disk("data/preprocessed/train")
        test_dataset = load_from_disk("data/preprocessed/test")

    try:
        model = AutoModelForSequenceClassification.from_pretrained("models/distilbert-base-uncased", num_labels=2)
    except FileNotFoundError:
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Metrics
    import evaluate
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/fine-tuned",  # SageMaker saves model artifacts here
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

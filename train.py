#!/usr/bin/env python
import logging
import click
import dotenv


@click.command()
@click.option("--epochs", type=int, default=3, help="Number of training epochs. Default is 3.")
@click.option("--batch-size", type=int, default=8, help="Batch size for training. Default is 8.")
@click.option(
    "--learning-rate", type=float, default=5e-5, help="Learning rate for training. Default is 5e-5."
)
@click.option("--s3-bucket", default=None, help="S3 bucket name for datasets. If not provided, local datasets will be used.")
def main(epochs, s3_bucket, batch_size, learning_rate):  
    assert s3_bucket is not None, "S3 bucket must be provided for SageMaker training."

    from sagemaker.huggingface import HuggingFace
    from text_classifier.utils import setup_aws_session
    
    setup_aws_session()

    estimator = HuggingFace(
        entry_point="src/train.py",
        source_dir="src",
        dependencies=["text_classifier"],  # Include other modules
        instance_type="ml.g4dn.xlarge",
        role=dotenv.get_key(dotenv.find_dotenv(), "SAGEMAKER_ROLE"),
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

    estimator.fit({
        "train": f"s3://{s3_bucket}/data/train",
        "test": f"s3://{s3_bucket}/data/test"
    })


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()

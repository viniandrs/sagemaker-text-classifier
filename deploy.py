import click
import sagemaker
from sagemaker.huggingface import HuggingFaceModel


def deploy_model_from_s3(s3_bucket):
    sess = sagemaker.Session()

    # Create HuggingFace model object
    huggingface_model = HuggingFaceModel(
        model_data=f"s3://{s3_bucket}/models/model.tar.gz",  # Points to your existing model
        role=sagemaker.get_execution_role(),
        transformers_version="4.26",  # Match your training environment
        pytorch_version="1.13",
        py_version="py39",
        entry_point="inference.py",  # Your inference script
        source_dir=".",  # Directory containing inference.py
    )

    # Deploy the model
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",  # Cheaper CPU instance
        endpoint_name="text-classifier-api",  # Custom endpoint name
    )

    print(f"Model deployed locally at endpoint '{predictor.endpoint_name}'")


def deploy_model_local():
    """Deploys the model locally using SageMaker Local Mode."""
    sess = sagemaker.Session()

    huggingface_model = HuggingFaceModel(
        model_data="file://models/fine_tuned",  # Local path to your model
        role=sagemaker.get_execution_role(),
        transformers_version="4.26",
        pytorch_version="1.13",
        py_version="py39",
        entry_point="inference.py",
        source_dir=".",
    )

    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="local",  # Use local mode
        endpoint_name="text-classifier-local",
    )

    print(f"Model deployed locally at endpoint '{predictor.endpoint_name}'")


@click.command()
@click.option("--s3_bucket", type=str, help="S3 bucket for model artifacts")
def main(s3_bucket):
    """Deploys the model either from S3 or locally."""
    if s3_bucket:
        deploy_model_from_s3(s3_bucket)
    else:
        deploy_model_local()


if __name__ == "__main__":
    main()

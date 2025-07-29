import click
import boto3


@click.command()
@click.option(
    "--endpoint-name",
    default="text-classifier-api",
    help="Name of the SageMaker endpoint to delete",
)
def main(endpoint_name):
    """Deletes the SageMaker endpoint."""
    try:
        client = boto3.client("sagemaker")
        client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint '{endpoint_name}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting endpoint: {e}")


if __name__ == "__main__":
    main()

import click
import boto3
import json

@click.command()
@click.argument("text", type=str, required=True, help="Input text to analyze sentiment")
@click.option("--endpoint-name", default="text-classifier-api", help="SageMaker endpoint name")
def predict(text, endpoint_name):
    """Predict sentiment using a deployed SageMaker endpoint."""
    runtime = boto3.client("runtime.sagemaker")
    payload = {"text": text}

    # Call endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    # Parse and print result
    result = json.loads(response["Body"].read())
    sentiment = "Positive" if result["prediction"] == 1 else "Negative"
    confidence = max(result["probabilities"]) * 100
    
    click.echo(f"Text: {text}")
    click.echo(f"Sentiment: {sentiment} ({confidence:.1f}% confidence)")

if __name__ == "__main__":
    predict()
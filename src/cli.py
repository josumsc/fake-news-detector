# Description: Command line interface for the fake news detector.
import click
from detector import DetectorPipeline


@click.group()
def cli():
    pass


@click.command(name="train")
@click.option(
    "--dataset",
    default="GonzaloA/fake_news",
    help="Dataset to download from HuggingFace datasets library.",
)
@click.option(
    "--checkpoint",
    default="distilbert-base-uncased-finetuned-sst-2-english",
    help="Model to fine-tune.",
)
@click.option(
    "--output", default="fake-news-detector", help="Name of the model to save."
)
@click.option("--lr", default=2e-5, help="Learning rate.")
@click.option("--batch_size", default=16, help="Batch size.")
@click.option("--epochs", default=3, help="Number of epochs.")
def train(
    dataset: str, checkpoint: str, output: str, lr: float, batch_size: int, epochs: int
):
    pipeline = DetectorPipeline(
        dataset_name=dataset, checkpoint=checkpoint, model_name=output
    )
    pipeline.train_pipeline(epochs=epochs, lr=lr, batch_size=batch_size)
    click.echo(f"Model saved into directory ./models/{output}.")


@click.command(name="predict")
@click.option(
    "--model", default="fake-news-detector", help="Model to use for prediction."
)
@click.option(
    "--checkpoint",
    default="distilbert-base-uncased-finetuned-sst-2-english",
    help="Tokenizer used to tokenize the text.",
)
@click.option("--text", default="This is a fake news", help="Text to predict.")
def predict(model: str, checkpoint: str, text: str):
    pipeline = DetectorPipeline(model_name=model, checkpoint=checkpoint)
    tokenizer, model = pipeline.load_model_from_directory()
    prediction, logits = pipeline.predict(tokenizer, model, text)
    click.echo(f"Prediction: {prediction}")
    click.echo(f"Logits: {logits}")


@click.command(name="publish")
@click.option("--model", default="fake-news-detector", help="Model to publish.")
def publish(model: str):
    pipeline = DetectorPipeline(model_name=model)
    pipeline.publish_model_from_directory()
    click.echo("Model published into HuggingFace Hub.")


cli.add_command(train)
cli.add_command(predict)
cli.add_command(publish)

if __name__ == "__main__":
    cli()

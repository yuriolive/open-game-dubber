import logging

import typer

from src.utils.model_manager import download_all_models


app = typer.Typer(help="Open Game Dubber CLI")


@app.command()
def download(
    output_dir: str = typer.Option("models", help="Directory to store downloaded models"),
    model_size: str = typer.Option("large-v3-turbo", help="Faster-Whisper model size"),
):
    """
    Download all required AI models (Faster-Whisper, Demucs, etc.)
    """
    typer.echo(f"Starting download of models to {output_dir}...")
    download_all_models(output_dir=output_dir, model_size=model_size)
    typer.echo("Download process completed.")


@app.command()
def hello():
    """
    Test command to verify CLI is working.
    """
    typer.echo("Hello from Open Game Dubber!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app()

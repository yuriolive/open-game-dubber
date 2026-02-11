import logging

import typer

from src.utils.model_manager import download_all_models
from src.core.pipeline import DubbingPipeline
import os
import glob
from tqdm import tqdm

app = typer.Typer(help="Open Game Dubber CLI")


@app.callback()
def main():
    """
    Open Game Dubber CLI
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


@app.command()
def dub_batch(
    input_dir: str = typer.Option("samples", help="Directory containing source WAV files"),
    output_dir: str = typer.Option("output", help="Directory to save dubbed files"),
    target_lang: str = typer.Option("Portuguese", help="Target language for dubbing"),
    limit: int = typer.Option(None, help="Limit the number of files to process"),
):
    """
    Batch process all WAV files in a directory.
    """
    if not os.path.exists(input_dir):
        typer.echo(f"Error: Input directory {input_dir} does not exist.", err=True)
        raise typer.Exit(1)

    os.makedirs(output_dir, exist_ok=True)
    
    # Get all WAV files
    files = glob.glob(os.path.join(input_dir, "*.wav"))
    if not files:
        typer.echo(f"No WAV files found in {input_dir}")
        return

    if limit:
        files = files[:limit]

    typer.echo(f"Starting batch process for {len(files)} files...")
    pipeline = DubbingPipeline(output_dir, target_lang)

    for file_path in tqdm(files, desc="Dubbing Clips"):
        pipeline.process_file(file_path)

    typer.echo(f"Batch processing completed. Results saved in {output_dir}")


if __name__ == "__main__":
    app()

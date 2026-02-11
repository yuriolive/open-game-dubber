from unittest.mock import patch

from typer.testing import CliRunner

from src.interface.cli import app

runner = CliRunner()


def test_hello():
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Hello from Open Game Dubber!" in result.stdout


@patch("src.interface.cli.download_all_models")
def test_download_defaults(mock_download):
    result = runner.invoke(app, ["download"])
    assert result.exit_code == 0
    assert "Starting download of models to models..." in result.stdout
    mock_download.assert_called_once_with(output_dir="models", model_size="large-v3-turbo")


@patch("src.interface.cli.download_all_models")
def test_download_custom_args(mock_download):
    result = runner.invoke(app, ["download", "--output-dir", "custom_models", "--model-size", "tiny"])
    assert result.exit_code == 0
    assert "Starting download of models to custom_models..." in result.stdout
    mock_download.assert_called_once_with(output_dir="custom_models", model_size="tiny")

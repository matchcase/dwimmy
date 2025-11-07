import sys
import pytest
from pathlib import Path
import pickle

from dwimmy.cli import generate_embeddings


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create a sample project structure in a temporary directory."""
    project_dir = tmp_path / "my-cli-project"
    project_dir.mkdir()

    # Create a dummy pyproject.toml
    pyproject_content = """
[project]
name = "my_cli"
version = "0.1.0"

[tool.setuptools.packages.find]
where = ["."]
include = ["my_cli*"]
"""
    (project_dir / "pyproject.toml").write_text(pyproject_content)

    # Create the CLI module
    cli_content = """
import argparse
from dwimmy.argparse import ArgumentParser

parser = ArgumentParser(description='A test CLI.')
parser.add_argument('--input-file', help='Path to the input file.')
parser.add_argument('--output-format', choices=['json', 'csv', 'xml'], help='The output format.')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
"""
    (project_dir / "my_cli.py").write_text(cli_content)

    return project_dir


def test_generate_command_end_to_end(sample_project: Path, monkeypatch):
    """Test the full 'generate' command pipeline from end to end."""
    # Change into the sample project directory to mimic real-world usage
    monkeypatch.chdir(sample_project)
    sys.path.insert(0, str(sample_project))

    # Run the generate command programmatically
    success = generate_embeddings(
        parser_modules=["my_cli.py"],
        output_file=".dwimmy-embeddings",
        pyproject_path="pyproject.toml"
    )

    assert success, "generate_embeddings should return True on success"

    # 1. Verify embeddings file was created
    embeddings_file = sample_project / ".dwimmy-embeddings"
    assert embeddings_file.exists(), "Embeddings file should be created"
    with open(embeddings_file, 'rb') as f:
        embeddings_data = pickle.load(f)
    assert isinstance(embeddings_data, dict)
    assert "--input-file" in embeddings_data
    assert "json" in embeddings_data

    # 2. Verify ONNX model directory and files were created
    onnx_dir = sample_project / ".dwimmy"
    assert onnx_dir.is_dir(), "ONNX model directory should be created"
    assert (onnx_dir / "model.onnx").exists(), "model.onnx should exist"
    assert (onnx_dir / "tokenizer.json").exists(), "tokenizer.json should exist"
    assert (onnx_dir / "config.json").exists(), "config.json should exist"

    # 3. Verify pyproject.toml was updated
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    
    with open(sample_project / "pyproject.toml", "rb") as f:
        pyproject_data = tomllib.load(f)

    dwimmy_config = pyproject_data.get("tool", {}).get("dwimmy", {})
    assert dwimmy_config.get("embeddings-file") == ".dwimmy-embeddings"
    assert dwimmy_config.get("model-dir") == ".dwimmy"

    # Check that package_data was updated correctly
    package_data = pyproject_data.get("tool", {}).get("setuptools", {}).get("package-data", {}).get("my_cli", [])
    assert ".dwimmy-embeddings" in package_data
    assert ".dwimmy/**" in package_data

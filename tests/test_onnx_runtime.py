
import sys
import pytest
from pathlib import Path
import pickle
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the path to allow importing 'dwimmy'
sys.path.insert(0, str(Path(__file__).parent.parent))

from dwimmy.core.matcher import SemanticMatcher

@pytest.fixture
def mock_onnx_project(tmp_path: Path) -> Path:
    """Creates a mock project with pre-generated ONNX model and embeddings."""
    project_dir = tmp_path / "my-onnx-cli"
    project_dir.mkdir()

    # 1. Create pyproject.toml
    pyproject_content = """
[project]
name = "my_onnx_cli"
version = "0.1.0"

[tool.dwimmy]
embeddings-file = "data/embeddings.pkl"
model-dir = "data/model"
"""
    (project_dir / "pyproject.toml").write_text(pyproject_content)

    # 2. Create the package data directory
    package_data_dir = project_dir / "my_onnx_cli" / "data"
    package_data_dir.mkdir(parents=True)
    
    # 3. Create dummy ONNX model and tokenizer files
    model_dir = package_data_dir / "model"
    model_dir.mkdir()
    (model_dir / "model.onnx").touch()
    (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "config.json").write_text("{}")

    # 4. Create dummy embeddings file
    embeddings = {
        "--help": np.random.rand(384),
        "--version": np.random.rand(384),
        "show": np.random.rand(384),
        "config": np.random.rand(384),
    }
    # Normalize embeddings to mimic the real ones
    for key in embeddings:
        embeddings[key] /= np.linalg.norm(embeddings[key])

    with open(package_data_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    return project_dir

def test_onnx_runtime_without_sentence_transformers(mock_onnx_project, monkeypatch):
    """
    Verify SemanticMatcher works using ONNX when sentence-transformers is not installed.
    """
    monkeypatch.chdir(mock_onnx_project)
    sys.path.insert(0, str(mock_onnx_project))

    # --- Mock the environment ---
    
    # 1. Mock importlib.resources to find our mock package files
    # This is a simplified mock; a real test might need a more robust one.
    def mock_files(package_name):
        return Path(mock_onnx_project) / package_name

    # 2. Mock the ONNX runtime and tokenizer to prevent real model loading
    mock_session = MagicMock()
    mock_tokenizer = MagicMock()

    # Define a fake embedding for our query "conf"
    # The shape is (batch_size, sequence_length, hidden_size)
    fake_conf_embedding = np.random.rand(1, 1, 384)
    fake_conf_embedding /= np.linalg.norm(fake_conf_embedding)

    mock_tokenizer.return_value = {'input_ids': np.array([[0]]), 'attention_mask': np.array([[1]])}
    mock_session.run.return_value = [fake_conf_embedding] # This is last_hidden_state

    # 3. CRITICAL: Simulate sentence-transformers NOT being installed
    # We patch the module in sys.modules, replacing it with None.
    with patch.dict(sys.modules, {"sentence_transformers": None}), \
         patch("importlib.resources.files", mock_files), \
         patch("onnxruntime.InferenceSession", return_value=mock_session), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):

        # --- Run the test ---
        
        matcher = SemanticMatcher()
        
        # This should succeed using the ONNX path and not raise an ImportError
        result = matcher.find_closest_runtime(
            query="conf",
            candidates=["show", "config", "--help"]
        )

        # --- Assertions ---

        # We can't easily assert the "correct" answer because the embeddings are random.
        # Instead, we assert that the machinery worked:
        # - The ONNX session was called.
        # - The tokenizer was called.
        # - A result was returned (or not, which is also a valid outcome).
        
        mock_tokenizer.assert_called_once_with(
            ["conf"], padding=True, truncation=True, return_tensors='np'
        )
        mock_session.run.assert_called_once()
        
        # In this controlled test, since we can't guarantee which random vector is
        # closest, we just check that the logic ran and produced a plausible result.
        # A more advanced test could inject known vectors and assert the exact match.
        assert result is None or isinstance(result, tuple)

def test_import_error_if_sentence_transformers_is_needed(mock_onnx_project, monkeypatch):
    """
    Verify that an ImportError is raised if we try to use a dev feature
    (like embed_search_space) without sentence-transformers.
    """
    monkeypatch.chdir(mock_onnx_project)
    
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        matcher = SemanticMatcher()
        with pytest.raises(ImportError, match="sentence-transformers is required"):
            # This method is for developers and requires the full library
            matcher.embed_search_space(search_space={"flags": ["--new-flag"]})

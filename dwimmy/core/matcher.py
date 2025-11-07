"""Semantic matching engine using sentence transformers"""

import sys
import time
import threading
import pickle
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import hashlib


class SemanticMatcher:
    """Lightweight semantic matcher using sentence transformers and ONNX"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._model = None  # For sentence-transformers model (dev time)
        self._onnx_session = None  # For ONNX model (runtime)
        self._tokenizer = None # For ONNX model (runtime)
        self._embeddings_cache: Dict[str, np.ndarray] = {}

    def _ensure_onnx_model_loaded(self):
        """Lazy load the ONNX model and tokenizer from the package."""
        if self._onnx_session is not None:
            return

        try:
            from importlib.resources import files
            import onnxruntime
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "onnxruntime and transformers are required for semantic matching.\n"
                "Please ensure your package includes them as dependencies."
            ) from None

        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                raise ImportError("Could not find a TOML parser library.")

        # Find and parse pyproject.toml
        pyproject_path = self._find_pyproject_toml()
        if not pyproject_path:
            raise FileNotFoundError("Could not find pyproject.toml in parent directories.")

        with open(pyproject_path, 'rb') as f:
            pyproject = tomllib.load(f)

        dwimmy_config = pyproject.get('tool', {}).get('dwimmy', {})
        model_dir_name = dwimmy_config.get('model-dir')
        project_name = pyproject.get('project', {}).get('name')

        if not model_dir_name or not project_name:
            raise ValueError("Dwimmy not configured in pyproject.toml. Run 'dwimmy generate'.")

        # Load model and tokenizer from the package resources
        model_path = files(project_name).joinpath(model_dir_name, 'model.onnx')
        tokenizer_path = files(project_name).joinpath(model_dir_name)

        with model_path.open('rb'): # Check existence
            self._onnx_session = onnxruntime.InferenceSession(str(model_path))
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        
        print("Loaded ONNX model and tokenizer from package.", file=sys.stderr)

    def _find_pyproject_toml(self) -> Optional[Path]:
        """Walk up from cwd to find pyproject.toml."""
        current = Path.cwd()
        for _ in range(10):
            pyproject_path = current / 'pyproject.toml'
            if pyproject_path.exists():
                return pyproject_path
            parent = current.parent
            if parent == current:
                return None
            current = parent
        return None

    def _embed_with_onnx(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for a list of sentences using the ONNX model."""
        if not self._tokenizer or not self._onnx_session:
            raise RuntimeError("ONNX model is not loaded.")

        # Tokenize sentences
        encoded_input = self._tokenizer(
            sentences, padding=True, truncation=True, return_tensors='np'
        )
        
        # Run inference
        model_inputs = {name: encoded_input[name] for name in self._onnx_session.get_inputs()}
        model_output = self._onnx_session.run(None, model_inputs)
        
        # Perform pooling (mean pooling)
        last_hidden_state = model_output[0]
        attention_mask = encoded_input['attention_mask']
        
        mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.maximum(np.sum(mask_expanded, axis=1), 1e-9)
        
        mean_pooled = sum_embeddings / sum_mask
        
        # Normalize embeddings
        norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        normalized_embeddings = mean_pooled / norm
        
        return normalized_embeddings
    
    def _ensure_model_loaded(self):
        """Lazy load the sentence transformer model
        
        Raises:
            ImportError: If sentence-transformers is not installed
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for generating embeddings.\n"
                    "\n"
                    "For CLI developers using 'dwimmy init':\n"
                    "  pip install dwimmy[dev]\n"
                    "\n"
                    "Note: End users installing CLIs with pre-computed embeddings\n"
                    "do not need this dependency."
                ) from None
            
            from dwimmy.core.utils import get_cache_dir
            
            cache_dir = get_cache_dir()
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(cache_dir)
            )
    
    # ... rest of the methods remain the same ...
    
        def _load_embeddings_from_package(self) -> bool:
            """Load pre-computed embeddings from the package if available"""
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    return False
    
            try:
                from importlib.resources import files, as_file
            except ImportError:
                return False
    
            pyproject_path = self._find_pyproject_toml()
            if not pyproject_path:
                return False
    
            with open(pyproject_path, 'rb') as f:
                pyproject = tomllib.load(f)
    
            dwimmy_config = pyproject.get('tool', {}).get('dwimmy', {})
            embeddings_file = dwimmy_config.get('embeddings-file')
            project_name = pyproject.get('project', {}).get('name')
    
            if not embeddings_file or not project_name:
                return False
    
            try:
                traversable = files(project_name).joinpath(embeddings_file)
                with as_file(traversable) as path:
                    with open(path, 'rb') as f:
                        cached_data = pickle.load(f)
                        self._embeddings_cache.update(cached_data)
                        print(f"Loaded {len(cached_data)} pre-computed embeddings from package", file=sys.stderr)
                        return True
            except (ModuleNotFoundError, FileNotFoundError):
                return False
            except Exception as e:
                print(f"Warning: Failed to load package embeddings: {e}", file=sys.stderr)
                return False
        
        def find_closest(
            self, 
            query: str, 
            candidates: List[str], 
            threshold: float = 0.50,
            exclude: Optional[List[str]] = None
        ) -> Optional[Tuple[str, float]]:
            """Find the most similar candidate using cosine similarity with ONNX."""
            if not candidates:
                return None
            
            if exclude:
                candidates = [c for c in candidates if c not in exclude]
            
            if not candidates:
                return None
            
            self._ensure_onnx_model_loaded()
            self.load_cached_embeddings()
            self._load_embeddings_from_package()
    
            query_embedding = self._embed_with_onnx([query])[0]
            
            candidate_embeddings = np.array([self._embeddings_cache[c] for c in candidates if c in self._embeddings_cache])
            if candidate_embeddings.shape[0] == 0:
                return None
    
            similarities = np.dot(candidate_embeddings, query_embedding)
            
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score >= threshold:
                return (candidates[best_idx], float(best_score))
            
            return None
    
        def find_top_n(
            self, 
            query: str, 
            candidates: List[str], 
            n: int = 3,
            threshold: float = 0.50,
            exclude: Optional[List[str]] = None
        ) -> List[Tuple[str, float]]:
            """Find top N most similar candidates using ONNX."""
            if not candidates:
                return []
    
            if exclude:
                candidates = [c for c in candidates if c not in exclude]
    
            if not candidates:
                return []
    
            self._ensure_onnx_model_loaded()
            self.load_cached_embeddings()
            self._load_embeddings_from_package()
    
            query_embedding = self._embed_with_onnx([query])[0]
    
            # Ensure all candidates are in the cache before creating the array
            valid_candidates = [c for c in candidates if c in self._embeddings_cache]
            if not valid_candidates:
                return []
    
            candidate_embeddings = np.array([self._embeddings_cache[c] for c in valid_candidates])
    
            similarities = np.dot(candidate_embeddings, query_embedding)
    
            top_indices = np.argsort(similarities)[::-1][:n]
            
            results = [
                (valid_candidates[idx], float(similarities[idx]))
                for idx in top_indices
                if similarities[idx] >= threshold
            ]
            
            return results

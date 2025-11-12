"""Semantic matching engine using sentence transformers"""

import sys
import time
import threading
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class SemanticMatcher:
    """Lightweight semantic matcher using sentence transformers and ONNX"""
    
    def __init__(self, model_name: Optional[str] = None):
        if model_name is None:
            self.model_name = self._get_model_name_from_pyproject()
        else:
            self.model_name = model_name
        
        self._model = None  # For sentence-transformers model (dev time)
        self._onnx_session = None  # For ONNX model (runtime)
        self._tokenizer = None # For ONNX model (runtime)
        self._embeddings_cache: Dict[str, np.ndarray] = {}

    def _get_model_name_from_pyproject(self) -> str:
        """Get the model name from pyproject.toml, with a default fallback."""
        default_model = 'sentence-transformers/all-MiniLM-L6-v2'
        
        pyproject_path = self._find_pyproject_toml()
        if not pyproject_path:
            return default_model
            
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return default_model
        
        try:
            with open(pyproject_path, 'rb') as f:
                pyproject = tomllib.load(f)
            
            dwimmy_config = pyproject.get('tool', {}).get('dwimmy', {})
            return dwimmy_config.get('model', default_model)
        except Exception:
            return default_model

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
    
    def _get_cache_path(self, cache_name: str = 'dwimmy_embeddings') -> Path:
        """Get the cache file path in the user's project or home directory"""
        # Try to find .dwimmy-embeddings in project root (walking up from cwd)
        current = Path.cwd()
        for _ in range(10):  # Don't go more than 10 levels up
            dwimmy_cache_file = current / '.dwimmy-embeddings'
            if dwimmy_cache_file.exists() and dwimmy_cache_file.is_file():
                return dwimmy_cache_file
            
            parent = current.parent
            if parent == current:  # Reached filesystem root
                break
            current = parent
            
        # Fallback to user's home cache directory
        from dwimmy.core.utils import get_cache_dir
        return get_cache_dir() / f'{cache_name}.pkl'
    
    def load_cached_embeddings(self, cache_name: str = 'dwimmy_embeddings') -> Dict[str, np.ndarray]:
        """Load embeddings from cache file if it exists"""
        cache_path = self._get_cache_path(cache_name)
        
        if not cache_path.exists():
            return {}
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self._embeddings_cache.update(cached_data)
                return cached_data
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}", file=sys.stderr)
            return {}
        
    def save_embeddings_cache(
            self, 
            embeddings: Dict[str, np.ndarray],
            cache_name: str = 'dwimmy_embeddings'
    ) -> Path:
        """Save embeddings to a cache file for future use"""
        cache_path = self._get_cache_path(cache_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
            
        print(f"Saved {len(embeddings)} embeddings to {cache_path}", file=sys.stderr)
        return cache_path
    
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
            
    def embed_search_space(
            self, 
            search_space: Dict[str, any],
            show_spinner: bool = True,
            cache_name: str = 'dwimmy_embeddings',
            use_cache: bool = True,
            save_cache: bool = True
    ) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all searchable items in batch"""
        self._ensure_model_loaded()
        
        # Try to load from cache first
        if use_cache:
            cached = self.load_cached_embeddings(cache_name)
            if cached:
                print(f"Loaded {len(cached)} cached embeddings from {self._get_cache_path(cache_name)}", 
                      file=sys.stderr)
                
        # Collect all items to embed
        all_items = []
        
        # Add flags (exclude help flags like -h, --help)
        user_flags = [f for f in search_space.get('flags', []) 
                      if f not in ['-h', '--help', '--version']]
        all_items.extend(user_flags)
        
        # Add subcommands
        all_items.extend(search_space.get('subcommands', []))
        
        # Add all choice values
        for choices in search_space.get('choices', {}).values():
            all_items.extend(choices)
            
        if not all_items:
            return {}
        
        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in all_items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
                
        # Find items that need embedding
        items_to_embed = [item for item in unique_items if item not in self._embeddings_cache]
        
        if not items_to_embed:
            # All items already cached
            return self._embeddings_cache
        
        # Show spinner during embedding
        spinner = None
        if show_spinner and items_to_embed:
            from dwimmy.core.utils import SimpleSpinner, is_interactive
            if is_interactive():
                spinner = SimpleSpinner(
                    text=f'Embedding {len(items_to_embed)} CLI components',
                    spinner='dots',
                    stream=sys.stderr
                )
                spinner.start()
                
                # Run embedding in a thread so we can animate the spinner
                embedding_done = threading.Event()
                exception_holder = {'exc': None, 'embeddings': None}
                
                def do_embedding():
                    try:
                        embeddings = self._model.encode(
                            items_to_embed, 
                            batch_size=64, 
                            show_progress_bar=False
                        )
                        exception_holder['embeddings'] = embeddings
                    except Exception as e:
                        exception_holder['exc'] = e
                    finally:
                        embedding_done.set()
                        
                thread = threading.Thread(target=do_embedding, daemon=True)
                thread.start()
                
                # Animate spinner while embedding
                while not embedding_done.is_set():
                    spinner._update()
                    time.sleep(0.1)
                    
                spinner.succeed(f'Embedded {len(items_to_embed)} CLI components')
                
                if exception_holder['exc']:
                    raise exception_holder['exc']
                
                embeddings = exception_holder['embeddings']
            else:
                # Non-interactive: just embed without spinner
                embeddings = self._model.encode(items_to_embed, batch_size=64, show_progress_bar=False)
        else:
            # No spinner: just embed
            embeddings = self._model.encode(items_to_embed, batch_size=64, show_progress_bar=False)
            
        # Cache embeddings
        for item, embedding in zip(items_to_embed, embeddings):
            self._embeddings_cache[item] = embedding
            
        # Save cache if requested
        if save_cache and items_to_embed:
            self.save_embeddings_cache(self._embeddings_cache, cache_name)
            
        return self._embeddings_cache
    
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
        """Find the most similar candidate using cosine similarity (for dev/testing)."""
        if not candidates:
            return None

        if exclude:
            candidates = [c for c in candidates if c not in exclude]

        if not candidates:
            return None

        self._ensure_model_loaded()

        query_embedding = self._model.encode([query], show_progress_bar=False)[0]

        candidate_embeddings = []
        for candidate in candidates:
            if candidate in self._embeddings_cache:
                candidate_embeddings.append(self._embeddings_cache[candidate])
            else:
                emb = self._model.encode([candidate], show_progress_bar=False)[0]
                self._embeddings_cache[candidate] = emb
                candidate_embeddings.append(emb)

        candidate_embeddings = np.array(candidate_embeddings)
        query_norm = np.linalg.norm(query_embedding)
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
        
        similarities = np.dot(candidate_embeddings, query_embedding) / (candidate_norms * query_norm + 1e-8)

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
        """Find top N most similar candidates (for dev/testing)."""
        if not candidates:
            return []

        if exclude:
            candidates = [c for c in candidates if c not in exclude]

        if not candidates:
            return []

        self._ensure_model_loaded()

        query_embedding = self._model.encode([query], show_progress_bar=False)[0]

        candidate_embeddings = []
        for candidate in candidates:
            if candidate in self._embeddings_cache:
                candidate_embeddings.append(self._embeddings_cache[candidate])
            else:
                emb = self._model.encode([candidate], show_progress_bar=False)[0]
                self._embeddings_cache[candidate] = emb
                candidate_embeddings.append(emb)
                
        candidate_embeddings = np.array(candidate_embeddings)
        query_norm = np.linalg.norm(query_embedding)
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)

        similarities = np.dot(candidate_embeddings, query_embedding) / (candidate_norms * query_norm + 1e-8)

        top_indices = np.argsort(similarities)[::-1][:n]
        results = [
            (candidates[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] >= threshold
        ]
        return results

    def find_closest_runtime(
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
        self._load_embeddings_from_package()

        query_embedding = self._embed_with_onnx([query])[0]

        valid_candidates = [c for c in candidates if c in self._embeddings_cache]
        if not valid_candidates:
            return None

        candidate_embeddings = np.array([self._embeddings_cache[c] for c in valid_candidates])
        similarities = np.dot(candidate_embeddings, query_embedding)
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score >= threshold:
            return (valid_candidates[best_idx], float(best_score))

        return None

    def find_top_n_runtime(
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
        self._load_embeddings_from_package()

        query_embedding = self._embed_with_onnx([query])[0]

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

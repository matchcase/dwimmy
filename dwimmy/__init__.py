"""
dwimmy - Do What I Mean for Python CLI libraries
Semantic argument correction using lightweight embeddings
"""

__version__ = "0.1.0"

# Make core utilities available at package level
from dwimmy.core.matcher import SemanticMatcher
from dwimmy.core.utils import is_interactive, get_cache_dir

__all__ = [
    "SemanticMatcher",
    "is_interactive",
    "get_cache_dir",
]

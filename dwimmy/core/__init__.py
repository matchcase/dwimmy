"""Core functionality for semantic matching and utilities"""

from dwimmy.core.matcher import SemanticMatcher
from dwimmy.core.utils import is_interactive, get_cache_dir

__all__ = [
    "SemanticMatcher",
    "is_interactive",
    "get_cache_dir",
]

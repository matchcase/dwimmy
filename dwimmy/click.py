"""Lazy-loaded Click integration with DWIM capabilities"""

import sys


class _LazyClickModule:
    """Lazy loader for Click with DWIM enhancements"""
    
    def __init__(self):
        self._click = None
        self._wrapped_classes = {}
    
    def _ensure_click_loaded(self):
        """Load Click or raise helpful error"""
        if self._click is None:
            try:
                import click
                self._click = click
            except ImportError:
                raise ImportError(
                    "Click is not installed. Install with: pip install dwimmy[click]"
                ) from None
    
    def __getattr__(self, name):
        """Intercept attribute access to wrap Click classes"""
        self._ensure_click_loaded()
        
        # Get the original Click attribute
        original = getattr(self._click, name)
        
        # For now, just pass through - full Click integration TBD
        # TODO: Wrap Group, Command with DWIM error handling
        return original


# Replace module with lazy loader
sys.modules[__name__] = _LazyClickModule()

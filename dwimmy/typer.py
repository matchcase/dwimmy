"""Lazy-loaded Typer integration with DWIM capabilities"""

import sys


class _LazyTyperModule:
    """Lazy loader for Typer with DWIM enhancements"""
    
    def __init__(self):
        self._typer = None
        self._wrapped_classes = {}
    
    def _ensure_typer_loaded(self):
        """Load Typer or raise helpful error"""
        if self._typer is None:
            try:
                import typer
                self._typer = typer
            except ImportError:
                raise ImportError(
                    "Typer is not installed. Install with: pip install dwimmy[typer]"
                ) from None
    
    def __getattr__(self, name):
        """Intercept attribute access to wrap Typer classes"""
        self._ensure_typer_loaded()
        
        # Get the original Typer attribute
        original = getattr(self._typer, name)
        
        # For now, just pass through - full Typer integration TBD
        # TODO: Wrap Typer class with DWIM error handling
        return original


# Replace module with lazy loader
sys.modules[__name__] = _LazyTyperModule()

"""Utility functions for dwimmy"""

import os
import sys
from pathlib import Path
from typing import Optional


def is_interactive():
    """Check if we're in an interactive terminal"""
    return sys.stdin.isatty() and sys.stdout.isatty()


def get_cache_dir():
    """Get the cache directory for storing models"""
    # Follow XDG Base Directory specification
    xdg_cache = os.environ.get('XDG_CACHE_HOME')
    if xdg_cache:
        cache_dir = Path(xdg_cache) / 'dwimmy'
    else:
        cache_dir = Path.home() / '.cache' / 'dwimmy'
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_config():
    """Load dwimmy configuration from user's config file"""
    config_paths = [
        Path.home() / '.dwimmyrc',
        Path.home() / '.config' / 'dwimmy' / 'config.toml',
    ]
    
    # Default configuration
    config = {
        'enabled': True,
        'confidence_threshold': 0.75,
        'interactive': 'auto',
        'auto_correct': False,
        'max_suggestions': 3,
    }
    
    # Try to load from config files
    for config_path in config_paths:
        if config_path.exists():
            try:
                import tomli
                with open(config_path, 'rb') as f:
                    user_config = tomli.load(f)
                    if 'dwimmy' in user_config:
                        config.update(user_config['dwimmy'])
            except ImportError:
                # tomli not available, skip config file
                pass
            break
    
    # Environment variable overrides
    if os.environ.get('DWIMMY_ENABLED') == '0':
        config['enabled'] = False
    if os.environ.get('DWIMMY_INTERACTIVE'):
        config['interactive'] = os.environ['DWIMMY_INTERACTIVE']
    if os.environ.get('DWIMMY_AUTO_CORRECT') == '1':
        config['auto_correct'] = True
    
    return config


class SimpleSpinner:
    """Lightweight spinner with no external dependencies"""
    
    SPINNERS = {
        'dots': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
        'line': ['-', '\\', '|', '/'],
        'arrow': ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
        'box': ['◰', '◳', '◲', '◱'],
        'dots2': ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'],
    }
    
    def __init__(self, text: str = '', spinner: str = 'dots', stream=None):
        """Initialize spinner
        
        Args:
            text: Message to display
            spinner: Spinner style ('dots', 'line', 'arrow', 'box', 'dots2')
            stream: Output stream (default: sys.stderr)
        """
        self.text = text
        self.spinner_style = self.SPINNERS.get(spinner, self.SPINNERS['dots'])
        self.stream = stream or sys.stderr
        self.running = False
        self.frame = 0
    
    def start(self):
        """Start the spinner"""
        if not is_interactive():
            return
        
        self.running = True
        self.frame = 0
        self._update()
    
    def _update(self):
        """Update spinner frame"""
        if not self.running or not is_interactive():
            return
        
        spinner_char = self.spinner_style[self.frame % len(self.spinner_style)]
        self.stream.write(f'\r{spinner_char} {self.text}')
        self.stream.flush()
        self.frame += 1
    
    def stop(self, final_text: Optional[str] = None):
        """Stop the spinner
        
        Args:
            final_text: Text to display when stopped (optional)
        """
        if not is_interactive():
            if final_text:
                print(final_text, file=self.stream)
            return
        
        self.running = False
        
        if final_text:
            self.stream.write(f'\r✓ {final_text}\n')
        else:
            self.stream.write('\r')
        
        self.stream.flush()
    
    def succeed(self, final_text: Optional[str] = None):
        """Stop spinner with success message"""
        if not is_interactive():
            if final_text:
                print(f'✓ {final_text}', file=self.stream)
            return
        
        self.stop(final_text)
    
    def fail(self, final_text: Optional[str] = None):
        """Stop spinner with failure message"""
        if not is_interactive():
            if final_text:
                print(f'✗ {final_text}', file=self.stream)
            return
        
        self.running = False
        
        if final_text:
            self.stream.write(f'\r✗ {final_text}\n')
        else:
            self.stream.write('\r')
        
        self.stream.flush()
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type:
            self.fail(f'Failed: {exc_val}')
        else:
            self.succeed()

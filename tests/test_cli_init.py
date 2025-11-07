"""Tests for dwimmy CLI init command"""

import pytest
import sys
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from dwimmy.cli import (
    init_embeddings,
    _extract_cli_components,
    _discover_modules,
    _update_pyproject,
)


@pytest.fixture
def temp_project():
    """Create a temporary project structure"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        
        # Create package directory
        pkg_dir = project_dir / 'myapp'
        pkg_dir.mkdir()
        
        # Create __init__.py
        (pkg_dir / '__init__.py').write_text('')
        
        # Create cli.py with a simple parser
        cli_code = '''
from dwimmy.argparse import ArgumentParser

parser = ArgumentParser(prog='myapp')
parser.add_argument('--config', help='configuration file')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--environment', 
                   choices=['dev', 'staging', 'production'])

def main():
    args = parser.parse_args()
    print(f"Config: {args.config}")

if __name__ == '__main__':
    main()
'''
        (pkg_dir / 'cli.py').write_text(cli_code)
        
        # Create pyproject.toml
        pyproject_content = '''[project]
name = "myapp"
version = "0.1.0"
dependencies = ["dwimmy"]
'''
        (project_dir / 'pyproject.toml').write_text(pyproject_content)
        
        yield project_dir


class TestExtractCliComponents:
    """Test extraction of CLI components from modules"""
    
    def test_extract_from_simple_parser(self, temp_project):
        """Test extracting components from a simple ArgumentParser"""
        sys.path.insert(0, str(temp_project))
        
        try:
            components = _extract_cli_components(['myapp.cli'])
            
            assert '--config' in components
            assert '--verbose' in components
            assert '--environment' in components
            assert 'dev' in components
            assert 'staging' in components
            assert 'production' in components
            
            # Should exclude help flags
            assert '-h' not in components
            assert '--help' not in components
        finally:
            sys.path.remove(str(temp_project))
            # Clean up module cache
            if 'myapp.cli' in sys.modules:
                del sys.modules['myapp.cli']
            if 'myapp' in sys.modules:
                del sys.modules['myapp']
    
    def test_extract_empty_module_list(self):
        """Test with empty module list"""
        components = _extract_cli_components([])
        assert components == []
    
    def test_extract_nonexistent_module(self):
        """Test with nonexistent module"""
        components = _extract_cli_components(['nonexistent.module'])
        assert components == []


class TestDiscoverModules:
    """Test module discovery from pyproject.toml"""
    
    def test_discover_explicit_modules(self, temp_project):
        """Test discovering explicitly specified modules"""
        pyproject_content = '''[project]
name = "myapp"

[tool.dwimmy]
cli-modules = ["myapp.cli", "myapp.commands"]
'''
        (temp_project / 'pyproject.toml').write_text(pyproject_content)
        
        modules = _discover_modules(str(temp_project / 'pyproject.toml'))
        
        assert 'myapp.cli' in modules
        assert 'myapp.commands' in modules
    
    def test_discover_no_config(self, temp_project):
        """Test discovery with no dwimmy config"""
        modules = _discover_modules(str(temp_project / 'pyproject.toml'))
        
        # Should return empty or default modules
        assert isinstance(modules, list)
    
    def test_discover_nonexistent_pyproject(self):
        """Test with nonexistent pyproject.toml"""
        modules = _discover_modules('/nonexistent/pyproject.toml')
        
        assert modules == []


class TestUpdatePyproject:
    """Test updating pyproject.toml"""
    
    def test_update_adds_dwimmy_config(self, temp_project):
        """Test that update adds dwimmy config section"""
        pyproject_path = temp_project / 'pyproject.toml'
        
        _update_pyproject(str(pyproject_path), '.dwimmy-embeddings')
        
        content = pyproject_path.read_text()
        
        assert '[tool.dwimmy]' in content
        assert 'embeddings-file' in content
        assert '.dwimmy-embeddings' in content
    
    def test_update_preserves_existing_config(self, temp_project):
        """Test that update preserves existing configuration"""
        pyproject_path = temp_project / 'pyproject.toml'
        
        _update_pyproject(str(pyproject_path), '.dwimmy-embeddings')
        
        new_content = pyproject_path.read_text()
        
        # Original project section should be preserved
        assert '[project]' in new_content
        assert 'name' in new_content


class TestInitEmbeddings:
    """Test the main init_embeddings function"""
    
    def test_init_fails_without_parsers(self, temp_project):
        """Test that init fails gracefully without CLI modules"""
        embeddings_file = temp_project / '.dwimmy-embeddings'
        
        success = init_embeddings(
            parser_modules=['nonexistent.module'],
            output_file=str(embeddings_file),
            pyproject_path=str(temp_project / 'pyproject.toml')
        )
        
        assert not success
        assert not embeddings_file.exists()
    
    def test_init_extracts_correct_components(self, temp_project):
        """Test that init correctly identifies CLI components"""
        sys.path.insert(0, str(temp_project))
        
        try:
            components = _extract_cli_components(['myapp.cli'])
            
            # Verify we found the right components
            assert '--config' in components
            assert '--verbose' in components
            assert '--environment' in components
            assert 'dev' in components
            assert 'production' in components
        finally:
            sys.path.remove(str(temp_project))
            if 'myapp.cli' in sys.modules:
                del sys.modules['myapp.cli']
            if 'myapp' in sys.modules:
                del sys.modules['myapp']


class TestCliIntegration:
    """Integration tests for the full CLI"""
    
    def test_cli_help_command(self):
        """Test that help command works"""
        original_argv = sys.argv
        sys.argv = ['dwimmy', '--help']
        
        from dwimmy.cli import main
        
        try:
            main()
        except SystemExit as e:
            # Help exits with 0
            assert e.code == 0
        finally:
            sys.argv = original_argv
    
    def test_cli_init_help(self):
        """Test that init help command works"""
        original_argv = sys.argv
        sys.argv = ['dwimmy', 'init', '--help']
        
        from dwimmy.cli import main
        
        try:
            main()
        except SystemExit as e:
            # Help exits with 0
            assert e.code == 0
        finally:
            sys.argv = original_argv
    
    def test_cli_init_no_modules(self):
        """Test that init fails gracefully with no modules"""
        original_argv = sys.argv
        sys.argv = ['dwimmy', 'init', '--modules', 'nonexistent.module']
        
        from dwimmy.cli import main
        
        try:
            main()
        except SystemExit as e:
            # Should exit with error code 1
            assert e.code == 1
        finally:
            sys.argv = original_argv


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

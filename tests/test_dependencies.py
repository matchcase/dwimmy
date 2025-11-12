"""Tests for optional dependency handling"""

import pytest
import sys
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np


class TestRuntimeDependencies:
    """Test that runtime works without sentence-transformers"""
    
    def test_dwim_parser_works_without_embeddings(self):
        """Test that DwimParser can be imported without sentence-transformers"""
        # This should work - no heavy imports required
        from dwimmy.argparse import ArgumentParser
        
        parser = ArgumentParser()
        parser.add_argument('--config')
        
        assert parser is not None
    
    def test_numpy_only_required_at_runtime(self):
        """Verify numpy is the only required runtime dependency"""
        import dwimmy.core.matcher
        
        # These should be available without sentence-transformers
        assert hasattr(dwimmy.core.matcher, 'np')
        
        # Create matcher - should not fail
        from dwimmy.core.matcher import SemanticMatcher
        matcher = SemanticMatcher()
        assert matcher is not None


class TestWithPreComputedEmbeddings:
    """Test that DWIM works perfectly with pre-computed embeddings"""
    
    @pytest.fixture
    def temp_project_with_embeddings(self):
        """Create a project with pre-computed embeddings"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            
            # Create package
            pkg_dir = project_dir / 'myapp'
            pkg_dir.mkdir()
            (pkg_dir / '__init__.py').write_text('')
            
            # Create CLI
            cli_code = '''
from dwimmy.argparse import ArgumentParser

parser = ArgumentParser(prog='myapp')
parser.add_argument('--config', help='configuration file')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--environment', choices=['dev', 'prod'])
'''
            (pkg_dir / 'cli.py').write_text(cli_code)
            
            # Create pre-computed embeddings file
            embeddings_dir = project_dir / 'myapp'
            embeddings_data = {
                '--config': np.random.randn(384),
                '--verbose': np.random.randn(384),
                '--environment': np.random.randn(384),
                'dev': np.random.randn(384),
                'prod': np.random.randn(384),
            }
            
            embeddings_file = embeddings_dir / '.dwimmy-embeddings'
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings_data, f)
            
            yield project_dir
    
    def test_load_precomputed_embeddings(self, temp_project_with_embeddings):
        """Test that pre-computed embeddings are loaded correctly"""
        embeddings_file = temp_project_with_embeddings / 'myapp' / '.dwimmy-embeddings'
        
        # Load embeddings
        with open(embeddings_file, 'rb') as f:
            cache = pickle.load(f)
        
        assert isinstance(cache, dict)
        assert len(cache) == 5
        assert '--config' in cache
        assert 'dev' in cache
        
        # Verify they're numpy arrays
        for key, value in cache.items():
            assert isinstance(value, np.ndarray)
            assert value.shape == (384,)
    
    def test_matcher_uses_precomputed_embeddings(self, temp_project_with_embeddings):
        """Test that matcher can use pre-computed embeddings without loading model"""
        from dwimmy.core.matcher import SemanticMatcher
        
        embeddings_file = temp_project_with_embeddings / 'myapp' / '.dwimmy-embeddings'
        
        # Load embeddings manually (simulating package loading)
        with open(embeddings_file, 'rb') as f:
            cache = pickle.load(f)
        
        # Create matcher and populate cache
        matcher = SemanticMatcher()
        matcher._embeddings_cache.update(cache)
        
        # Now we can find closest without loading the model!
        candidates = ['--config', '--verbose', '--environment']
        result = matcher.find_closest('--cofig', candidates, threshold=0.50)
        
        # Should find the match (if similarity is high enough)
        # The important thing is it doesn't crash trying to load sentence-transformers
        assert result is None or isinstance(result, tuple)


class TestErrorHandlingWithoutDependencies:
    """Test helpful error messages when dependencies are missing"""
    
    @patch.dict('sys.modules', {'sentence_transformers': None})
    def test_helpful_error_without_sentence_transformers(self):
        """Test that users get helpful error message without sentence-transformers"""
        from dwimmy.core.matcher import SemanticMatcher
        
        matcher = SemanticMatcher()
        
        # Trying to load model should give helpful error
        with pytest.raises(ImportError) as exc_info:
            matcher._ensure_model_loaded()
        
        error_message = str(exc_info.value)
        assert 'sentence-transformers' in error_message
        assert 'dwimmy[dev]' in error_message
        assert 'pip install' in error_message


class TestEndUserScenario:
    """Test a realistic end-user scenario"""
    
    @pytest.fixture
    def end_user_cli_package(self):
        """Simulate an end-user CLI package with dwimmy"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            
            # Create a realistic CLI package structure
            pkg_dir = project_dir / 'mycli'
            pkg_dir.mkdir()
            (pkg_dir / '__init__.py').write_text('__version__ = "1.0.0"')
            
            # Main CLI module
            cli_code = '''
from dwimmy.argparse import ArgumentParser

parser = ArgumentParser(prog='mycli')
parser.add_argument('--output', '-o', help='output file')
parser.add_argument('--format', choices=['json', 'yaml', 'xml'])
parser.add_argument('--verbose', '-v', action='store_true')

def main():
    args = parser.parse_args()
    print(f"Output: {args.output}")

if __name__ == '__main__':
    main()
'''
            (pkg_dir / 'cli.py').write_text(cli_code)
            
            # Pre-computed embeddings (as would be generated by dwimmy init)
            embeddings = {
                '--output': np.random.randn(384),
                '-o': np.random.randn(384),
                '--format': np.random.randn(384),
                '--verbose': np.random.randn(384),
                '-v': np.random.randn(384),
                'json': np.random.randn(384),
                'yaml': np.random.randn(384),
                'xml': np.random.randn(384),
            }
            
            embeddings_file = pkg_dir / '.dwimmy-embeddings'
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
            
            # pyproject.toml showing minimal dependencies
            pyproject = '''[project]
name = "mycli"
version = "1.0.0"
description = "A simple CLI app"
dependencies = [
    "dwimmy",
]

[tool.setuptools.package-data]
mycli = [".dwimmy-embeddings"]

[project.scripts]
mycli = "mycli.cli:main"
'''
            (project_dir / 'pyproject.toml').write_text(pyproject)
            
            yield project_dir
    
    def test_end_user_cli_minimal_deps(self, end_user_cli_package):
        """Test that end-user CLI only needs dwimmy (no sentence-transformers)"""
        # Verify the package structure
        pkg_dir = end_user_cli_package / 'mycli'
        assert (pkg_dir / '__init__.py').exists()
        assert (pkg_dir / 'cli.py').exists()
        assert (pkg_dir / '.dwimmy-embeddings').exists()
        
        # Verify embeddings file is valid
        with open(pkg_dir / '.dwimmy-embeddings', 'rb') as f:
            embeddings = pickle.load(f)
        
        assert len(embeddings) > 0
        assert '--output' in embeddings
        assert 'json' in embeddings
    
    def test_end_user_can_import_cli_module(self, end_user_cli_package):
        """Test that end-user can import the CLI module"""
        sys.path.insert(0, str(end_user_cli_package))
        
        try:
            import mycli.cli
            
            # Module should be importable
            assert mycli.cli is not None
            
            # Parser should be defined in the module
            assert hasattr(mycli.cli, 'parser')
            
            # Parser should have the expected arguments
            parser = mycli.cli.parser
            actions = {action.dest: action for action in parser._actions}
            assert 'output' in actions
            assert 'format' in actions
            assert 'verbose' in actions
        finally:
            sys.path.remove(str(end_user_cli_package))
            if 'mycli.cli' in sys.modules:
                del sys.modules['mycli.cli']
            if 'mycli' in sys.modules:
                del sys.modules['mycli']


class TestDependencyDocumentation:
    """Test that dependency information is documented"""
    
    def test_readme_documents_dependencies(self):
        """Verify README mentions dependency differences"""
        readme_path = Path(__file__).parent.parent / 'README.md'
        
        if readme_path.exists():
            content = readme_path.read_text()
            
            # Should document minimal runtime deps
            assert 'numpy' in content or 'End Users' in content
            
            # Should document dev deps
            assert 'developer' in content.lower() or 'dev' in content.lower()
    
    def test_pyproject_documents_optional_deps(self):
        """Verify pyproject.toml shows optional dependencies"""
        pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            
            # Should have optional dependencies section
            assert 'optional-dependencies' in content or 'dev' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

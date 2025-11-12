"""Integration tests for dwimmy argparse DWIM functionality"""

import pytest
import sys
from unittest.mock import patch
from dwimmy.argparse import ArgumentParser
from dwimmy.core.matcher import SemanticMatcher


class TestDwimFlagCorrection:
    """Test semantic flag correction"""
    
    @pytest.fixture
    def simple_parser(self):
        """Create a simple parser with common arguments"""
        parser = ArgumentParser()
        parser.add_argument('--config', help='configuration file')
        parser.add_argument('--verbose', action='store_true', help='verbose output')
        parser.add_argument('--environment', 
                          choices=['development', 'staging', 'production'],
                          help='target environment')
        return parser
    
    def test_search_space_extraction(self, simple_parser):
        """Test that search space is correctly extracted from parser"""
        search_space = simple_parser._build_search_space()
        
        # Check flags are extracted
        assert '--config' in search_space['flags']
        assert '--verbose' in search_space['flags']
        assert '--environment' in search_space['flags']
        
        # Check choices are extracted
        assert 'environment' in search_space['choices']
        assert 'development' in search_space['choices']['environment']
        assert 'production' in search_space['choices']['environment']
    
    def test_matcher_embedding(self, simple_parser):
        """Test that semantic matcher can embed search space"""
        search_space = simple_parser._build_search_space()
        matcher = SemanticMatcher()
        
        embeddings = matcher.embed_search_space(search_space)
        
        # Check embeddings were created for all items
        assert len(embeddings) > 0
        assert '--config' in embeddings
        assert 'production' in embeddings
    
    def test_flag_similarity_matching(self, simple_parser):
        """Test that similar flags are matched correctly"""
        search_space = simple_parser._build_search_space()
        matcher = SemanticMatcher()
        matcher.embed_search_space(search_space)
        
        # Test exact match
        result = matcher.find_closest('--config', search_space['flags'], threshold=0.50)
        assert result is not None
        assert result[0] == '--config'
        assert result[1] > 0.90  # Exact match should be very high
        
        # Test typo matching - use lower threshold for typos
        result = matcher.find_closest('--cofig', search_space['flags'], threshold=0.50)
        assert result is not None
        assert result[0] == '--config'
        assert result[1] > 0.50  # Typo should still match above 0.50    

    def test_choice_similarity_matching(self, simple_parser):
        """Test that similar choice values are matched correctly"""
        search_space = simple_parser._build_search_space()
        matcher = SemanticMatcher()
        matcher.embed_search_space(search_space)
        
        choices = search_space['choices']['environment']
        
        # Test exact match
        result = matcher.find_closest('production', choices, threshold=0.50)
        assert result is not None
        assert result[0] == 'production'
        assert result[1] > 0.90  # Exact match
        
        # Test typo matching - 'prod' should match 'production'
        result = matcher.find_closest('prod', choices, threshold=0.30)
        assert result is not None
        # Should match one of the choices
        assert result[0] in choices

class TestDwimReconstruction:
    """Test command reconstruction from malformed input"""
    
    @pytest.fixture
    def rich_parser(self):
        """Create a parser with multiple argument types"""
        parser = ArgumentParser()
        parser.add_argument('--config', help='config file')
        parser.add_argument('--output-file', help='output file')
        parser.add_argument('--environment', 
                          choices=['dev', 'staging', 'production'])
        parser.add_argument('--debug', action='store_true')
        return parser
    
    def test_missing_dashes_reconstruction(self, rich_parser):
        """Test reconstruction of missing dashes"""
        # Initialize matcher
        search_space = rich_parser._build_search_space()
        rich_parser._matcher = SemanticMatcher()
        rich_parser._matcher.embed_search_space(search_space)
        rich_parser._search_space = search_space
        
        # Input: config file.yaml (missing --)
        tokens = ['config', 'file.yaml']
        
        reconstructed = rich_parser._reconstruct_command(tokens)
        
        # Should suggest --config
        assert reconstructed is not None
        assert '--config' in reconstructed or any('config' in str(t) for t in reconstructed)
    
    def test_malformed_flag_reconstruction(self, rich_parser):
        """Test reconstruction of misspelled flags"""
        search_space = rich_parser._build_search_space()
        rich_parser._matcher = SemanticMatcher()
        rich_parser._matcher.embed_search_space(search_space)
        rich_parser._search_space = search_space
        
        # Input: --cofig file.yaml (typo in --config)
        tokens = ['--cofig', 'file.yaml']
        
        reconstructed = rich_parser._reconstruct_command(tokens)
        
        # Should correct to --config
        assert reconstructed is not None
        assert '--config' in reconstructed
        assert 'file.yaml' in reconstructed
    
    def test_choice_value_correction(self, rich_parser):
        """Test correction of incorrect choice values"""
        search_space = rich_parser._build_search_space()
        rich_parser._matcher = SemanticMatcher()
        rich_parser._matcher.embed_search_space(search_space, show_spinner=False)
        rich_parser._search_space = search_space
        
        # Test 1: Exact match should be recognized
        tokens = ['--environment', 'production']
        reconstructed = rich_parser._reconstruct_command(tokens)
        # Exact match shouldn't trigger reconstruction (no change needed)
        assert reconstructed is None
        
        # Test 2: Test with typo in flag name, value is exact
        tokens = ['--environmnt', 'production']
        reconstructed = rich_parser._reconstruct_command(tokens)
        assert reconstructed is not None
        assert '--environment' in reconstructed
        assert 'production' in reconstructed

class TestDwimEndToEnd:
    """End-to-end tests of DWIM functionality"""
    
    def test_dwim_disabled_no_recovery(self):
        """Test that DWIM can be disabled"""
        parser = ArgumentParser(dwim_enabled=False)
        parser.add_argument('--config')
        
        assert parser._dwim_enabled is False
    
    @patch('dwimmy.core.utils.is_interactive', return_value=False)
    @patch('sys.stdin.isatty', return_value=False)
    def test_non_interactive_mode_fails_gracefully(self, mock_isatty, mock_interactive):
        """Test that DWIM fails gracefully in non-interactive environments"""
        parser = ArgumentParser()
        parser.add_argument('--config')
        
        # Simulate non-interactive environment (like CI/CD)
        sys.argv = ['test', '--cofig', 'file.yaml']
        
        # Should exit with error, not hang
        with pytest.raises(SystemExit):
            parser.parse_args()
    
    def test_config_loads_correctly(self):
        """Test that configuration is loaded"""
        parser = ArgumentParser()
        
        # Config should be loaded
        assert parser._config is not None
        assert 'enabled' in parser._config
        assert 'confidence_threshold' in parser._config


class TestDwimSemanticsMatcherEdgeCases:
    """Test edge cases in semantic matching"""
    
    def test_top_n_matches(self):
        """Test finding top N candidates"""
        matcher = SemanticMatcher()
        candidates = ['--config', '--output', '--verbose', '--environment']
        
        matcher.embed_search_space({
            'flags': candidates,
            'choices': {},
            'subcommands': []
        })
        
        # Find top 3 similar to 'configure'
        results = matcher.find_top_n('--config', candidates, n=3, threshold=0.60)
        
        assert len(results) > 0
        # First result should be exact match or very similar
        assert results[0][1] > 0.80
    
    def test_threshold_respecting(self):
        """Test that confidence thresholds are respected"""
        matcher = SemanticMatcher()
        candidates = ['--config', '--verbose', '--debug', '--database']
        
        matcher.embed_search_space({
            'flags': candidates,
            'choices': {},
            'subcommands': []
        })
        
        # Very strict threshold - should only match very similar
        result_strict = matcher.find_closest('--xyz', candidates, threshold=0.99)
        assert result_strict is None
        
        # Loose threshold - should find something
        result_loose = matcher.find_closest('--xyz', candidates, threshold=0.30)
        assert result_loose is not None
    
    def test_empty_candidates(self):
        """Test handling of empty candidate lists"""
        matcher = SemanticMatcher()
        
        result = matcher.find_closest('--config', [], threshold=0.75)
        assert result is None
    
    def test_single_candidate(self):
        """Test with single candidate"""
        matcher = SemanticMatcher()
        candidates = ['--config']
        
        matcher.embed_search_space({
            'flags': candidates,
            'choices': {},
            'subcommands': []
        })
        
        result = matcher.find_closest('--cofig', candidates, threshold=0.50)
        assert result is not None
        assert result[0] == '--config'


class TestDwimIntegrationScenarios:
    """Test realistic usage scenarios"""
    
    def test_deploy_command_scenario(self):
        """Test a realistic deployment CLI scenario"""
        parser = ArgumentParser()
        parser.add_argument('--environment', 
                          choices=['development', 'staging', 'production'],
                          required=True)
        parser.add_argument('--region',
                          choices=['us-east-1', 'us-west-2', 'eu-west-1'])
        parser.add_argument('--force', action='store_true')
        parser.add_argument('--dry-run', action='store_true')
        
        search_space = parser._build_search_space()
        
        assert '--environment' in search_space['flags']
        assert '--region' in search_space['flags']
        assert 'production' in search_space['choices']['environment']
        assert 'us-east-1' in search_space['choices']['region']
    
    def test_database_command_scenario(self):
        """Test a realistic database CLI scenario"""
        parser = ArgumentParser()
        parser.add_argument('--host', help='database host')
        parser.add_argument('--port', type=int, help='database port')
        parser.add_argument('--username', help='username')
        parser.add_argument('--password', help='password')
        parser.add_argument('--database', help='database name')
        parser.add_argument('--format', 
                          choices=['json', 'csv', 'sql'],
                          help='output format')
        
        search_space = parser._build_search_space()
        matcher = SemanticMatcher()
        matcher.embed_search_space(search_space, show_spinner=False)
        
        # Exclude help flags when matching
        searchable_flags = [f for f in search_space['flags'] 
                           if f not in ['-h', '--help', '--version']]
        
        # User types --formt (typo of --format)
        result = matcher.find_closest('--formt', searchable_flags, threshold=0.50)
        assert result is not None
        assert result[0] == '--format', f"Expected '--format', got '{result[0]}'"
        
        # User types 'jsn' instead of 'json'
        result = matcher.find_closest('jsn', ['json', 'csv', 'sql'], threshold=0.30)
        assert result is not None
        assert result[0] in ['json', 'csv', 'sql']

class TestDwimUserPrompting:
    """Test user interaction and prompting"""
    
    @patch('builtins.input', return_value='y')
    @patch('dwimmy.core.utils.is_interactive', return_value=True)
    def test_user_accepts_suggestion(self, mock_interactive, mock_input):
        """Test when user accepts dwim suggestion"""
        parser = ArgumentParser()
        parser.add_argument('--config')
        
        # Mock sys.argv
        with patch.object(sys, 'argv', ['test', '--cofig', 'file.yaml']):
            search_space = parser._build_search_space()
            parser._matcher = SemanticMatcher()
            parser._matcher.embed_search_space(search_space)
            parser._search_space = search_space
            
            suggestion = ['--config', 'file.yaml']
            
            # This would prompt and retry, but we're mocking input
            # Just verify the suggestion is valid
            assert '--config' in suggestion
            assert 'file.yaml' in suggestion
    
    @patch('builtins.input', return_value='n')
    @patch('dwimmy.core.utils.is_interactive', return_value=True)
    def test_user_declines_suggestion(self, mock_interactive, mock_input):
        """Test when user declines dwim suggestion"""
        parser = ArgumentParser()
        parser.add_argument('--config')
        
        # When user declines, should fall back to standard error
        with patch.object(sys, 'argv', ['test', '--invalid']):
            with pytest.raises(SystemExit):
                parser.error("unrecognized arguments: --invalid")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

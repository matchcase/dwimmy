"""Tests for dwimmy argparse integration"""

import pytest
import sys
from dwimmy.argparse import ArgumentParser


def test_basic_parsing():
    """Test that normal parsing still works"""
    parser = ArgumentParser()
    parser.add_argument('--config')
    
    # Simulate command line
    sys.argv = ['test', '--config', 'test.yaml']
    args = parser.parse_args()
    
    assert args.config == 'test.yaml'


def test_flag_matching():
    """Test semantic matching of misspelled flags"""
    parser = ArgumentParser()
    parser.add_argument('--config', help='configuration file')
    parser.add_argument('--verbose', action='store_true')
    
    # Build search space
    search_space = parser._build_search_space()
    
    assert '--config' in search_space['flags']
    assert '--verbose' in search_space['flags']


def test_choice_matching():
    """Test matching of enum choices"""
    parser = ArgumentParser()
    parser.add_argument(
        '--environment',
        choices=['development', 'staging', 'production']
    )
    
    search_space = parser._build_search_space()
    
    assert 'environment' in search_space['choices']
    assert 'production' in search_space['choices']['environment']


def test_dwim_disabled():
    """Test that DWIM can be disabled"""
    parser = ArgumentParser(dwim_enabled=False)
    parser.add_argument('--config')
    
    assert parser._dwim_enabled is False


if __name__ == '__main__':
    pytest.main([__file__])

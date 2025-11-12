import pytest
import sys
from pathlib import Path

# Add the project root to the python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dwimmy.cli import _extract_cli_components

# Mock click and typer if not installed
try:
    import click
except ImportError:
    click = None

try:
    import typer
except ImportError:
    typer = None


@pytest.fixture
def temp_cli_project(tmp_path: Path) -> Path:
    """Create a temporary CLI project with a cli.py file."""
    project_dir = tmp_path / "temp_project"
    project_dir.mkdir()
    sys.path.insert(0, str(project_dir))
    yield project_dir
    sys.path.pop(0)


@pytest.mark.skipif(click is None, reason="click is not installed")
def test_extract_from_click(temp_cli_project: Path):
    """Test that CLI components can be extracted from a Click app."""
    cli_content = """
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name', help='The person to greet.')
def hello(count, name):
    pass

@cli.command()
@click.option('--choice', type=click.Choice(['a', 'b', 'c']))
def test(choice):
    pass
"""
    (temp_cli_project / "temp_cli.py").write_text(cli_content)

    components = _extract_cli_components(['temp_cli'])
    
    expected = {
        'hello', 
        '--count', 
        '--name', 
        'test', 
        '--choice', 
        'a', 'b', 'c'
    }
    assert set(components) == expected


@pytest.mark.skipif(typer is None, reason="typer is not installed")
def test_extract_from_typer(temp_cli_project: Path):
    """Test that CLI components can be extracted from a Typer app."""
    cli_content = """
import typer

app = typer.Typer()

@app.command()
def create(user: str):
    pass

@app.command()
def delete(user: str, force: bool = typer.Option(False, "--force")):
    pass
"""
    (temp_cli_project / "temp_cli_typer.py").write_text(cli_content)

    components = _extract_cli_components(['temp_cli_typer'])
    
    expected = {
        'create',
        'delete',
        '--force',
        'user'  # Typer automatically creates an argument for the 'user' parameter
    }
    # Typer also adds a --install-completion and --show-completion option
    assert set(components) - {'--install-completion', '--show-completion'} == expected

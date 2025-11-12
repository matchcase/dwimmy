"""Command-line interface for dwimmy - for package developers"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
import pickle
from typing import List, Optional, TYPE_CHECKING

from dwimmy.core.matcher import SemanticMatcher
if TYPE_CHECKING:
    import typer


from dwimmy.core.utils import SimpleSpinner, is_interactive

def generate_embeddings(
    parser_modules: Optional[List[str]] = None,
    output_file: str = '.dwimmy-embeddings',
    pyproject_path: str = 'pyproject.toml'
):
    """Generate embeddings and ONNX model for a CLI package"""
    print("🔍 Scanning for CLI definitions...", file=sys.stderr)
    
    # Auto-discover parser modules from pyproject.toml if not specified
    if not parser_modules:
        parser_modules = _discover_modules(pyproject_path)
    
    if not parser_modules:
        print("❌ No CLI modules found. Specify with --modules or define in pyproject.toml", 
              file=sys.stderr)
        return False
    
    print(f"Found {len(parser_modules)} module(s): {', '.join(parser_modules)}", 
          file=sys.stderr)
    
    try:
        import click
    except ImportError:
        click = None

    try:
        import typer
    except ImportError:
        typer = None

    # Extract all CLI components
    all_components = _extract_cli_components(parser_modules, click=click, typer=typer)
    
    if not all_components:
        print("❌ No ArgumentParser/Click/Typer definitions found", file=sys.stderr)
        return False
    
    print(f"Found {len(all_components)} CLI components", file=sys.stderr)
    
    # Generate embeddings
    spinner = SimpleSpinner(
        text=f'Generating embeddings for {len(all_components)} components',
        spinner='dots',
        stream=sys.stderr
    ) if is_interactive() else None
    
    if spinner:
        spinner.start()
    
    try:
        matcher = SemanticMatcher()
        matcher._ensure_model_loaded()  # Make sure model is loaded
        embeddings = matcher._model.encode(all_components, batch_size=64, show_progress_bar=False)
        
        if spinner:
            spinner.succeed(f'Generated embeddings for {len(all_components)} components')
        else:
            print(f"✓ Generated embeddings for {len(all_components)} components", 
                  file=sys.stderr)
    except Exception as e:
        if spinner:
            spinner.fail(f'Failed to generate embeddings: {e}')
        else:
            print(f"❌ Failed: {e}", file=sys.stderr)
        return False
    
    # Save embeddings
    cache_data = {item: emb for item, emb in zip(all_components, embeddings)}
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"✓ Saved embeddings to {output_path}", file=sys.stderr)

    # Export model to ONNX
    _export_model_to_onnx(matcher, pyproject_path)

    # Update pyproject.toml
    _update_pyproject(pyproject_path, embeddings_file=output_file)
    
    print("\n✨ Setup complete! Embeddings and ONNX model will be included in your package.", 
          file=sys.stderr)
    
    return True


def _export_model_to_onnx(matcher: 'SemanticMatcher', pyproject_path: str):
    """Export the sentence transformer model to ONNX format"""
    from optimum.exporters.onnx import main_export

    model_name = matcher.model_name
    output_dir = Path.cwd() / '.dwimmy'
    output_dir.mkdir(exist_ok=True)

    print(f"\n🚀 Exporting {model_name} to ONNX format...", file=sys.stderr)

    try:
        main_export(
            model_name_or_path=model_name,
            output=output_dir,
            task="feature-extraction",
            opset=14,  # Use a reasonably modern opset
            do_validation=True,
        )

        print(f"✓ Saved ONNX model to {output_dir}", file=sys.stderr)

        # Update pyproject.toml to include the model directory
        _update_pyproject(pyproject_path, model_dir='.dwimmy')

    except Exception as e:
        print(f"❌ Failed to export ONNX model: {e}", file=sys.stderr)
        print("  Please ensure you have `optimum[onnxruntime]` installed:", file=sys.stderr)
        print("  pip install dwimmy[dev]", file=sys.stderr)




def _discover_modules(pyproject_path: str) -> List[str]:
    """Auto-discover CLI modules from pyproject.toml"""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return []
    
    try:
        with open(pyproject_path, 'rb') as f:
            pyproject = tomllib.load(f)
    except FileNotFoundError:
        return []
    
    # Look for dwimmy config in pyproject.toml
    dwimmy_config = pyproject.get('tool', {}).get('dwimmy', {})
    
    if 'cli-modules' in dwimmy_config:
        return dwimmy_config['cli-modules']
    
    # Fallback: look for common patterns
    project_name = pyproject.get('project', {}).get('name', 'app')
    common_modules = [
        f'{project_name}.cli',
        f'{project_name}/cli.py',
        'cli.py',
    ]
    
    discovered = []
    for module in common_modules:
        if Path(module).exists() or _can_import(module):
            discovered.append(module)
    
    return discovered


def _can_import(module_name: str) -> bool:
    """Check if a module can be imported"""
    try:
        __import__(module_name.replace('.py', '').replace('/', '.'))
        return True
    except ImportError:
        return False


def _extract_cli_components(
    modules: List[str],
    click: Optional[object] = None,
    typer: Optional[object] = None,
) -> List[str]:
    """Extract all CLI component names from modules"""
    import argparse
    import importlib
    
    if click is None:
        try:
            import click
        except ImportError:
            click = None

    if typer is None:
        try:
            import typer
        except ImportError:
            typer = None
    
    components = set()
    
    for module_name in modules:
        # Handle both module.py and module formats
        module_name = module_name.replace('.py', '').replace('/', '.')
        
        try:
            # Import the module (reload if already imported)
            module = __import__(module_name, fromlist=[None])
            # Reload to get fresh version
            importlib.reload(module)
            
            # Find all ArgumentParser, Click, or Typer instances
            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                    
                    # argparse
                    if isinstance(attr, argparse.ArgumentParser):
                        _extract_from_argparse(attr, components)
                    
                    # click
                    if click and isinstance(attr, (click.Group, click.Command)):
                        _extract_from_click(attr, components, click)

                    # typer
                    if typer and isinstance(attr, typer.Typer):
                        _extract_from_typer(attr, components, typer, click)

                except (AttributeError, TypeError):
                    # Skip attributes that can't be inspected
                    pass
        except Exception as e:
            print(f"⚠️  Failed to extract from {module_name}: {e}", file=sys.stderr)
    
    # Filter out help flags
    components = {c for c in components if c not in ['-h', '--help', '--version']}
    
    return list(components)


def _extract_from_argparse(parser: argparse.ArgumentParser, components: set):
    """Extract components from an argparse.ArgumentParser"""
    for action in parser._actions:
        if action.option_strings:
            components.update(action.option_strings)
        
        if action.choices:
            if isinstance(action, argparse._SubParsersAction):
                components.update(action.choices.keys())
            else:
                components.update(str(c) for c in action.choices)


def _extract_from_click(cli_obj, components: set, click: object):
    """Recursively extract components from a click.Group or click.Command"""
    if hasattr(cli_obj, 'commands') and isinstance(cli_obj.commands, dict):
        for name, cmd in cli_obj.commands.items():
            components.add(name)
            _extract_from_click(cmd, components, click)

    if hasattr(cli_obj, 'params'):
        for param in cli_obj.params:
            components.update(param.opts)
            if isinstance(param.type, click.Choice):
                components.update(param.type.choices)


def _extract_from_typer(app: 'typer.Typer', components: set, typer: object, click: object):
    """Extract components from a typer.Typer app"""
    try:
        # Convert Typer app to a Click object
        click_obj = typer.main.get_command(app)
        # Now, reuse the click extraction logic
        _extract_from_click(click_obj, components, click)
    except Exception as e:
        print(f"⚠️  Failed to extract from typer app: {e}", file=sys.stderr)



def _update_pyproject(
    pyproject_path: str, 
    embeddings_file: Optional[str] = None,
    model_dir: Optional[str] = None
):
    """
    Update pyproject.toml using tomlkit to preserve formatting and comments.
    """
    try:
        import tomlkit
    except ImportError:
        print("⚠️  tomlkit is not installed. Please run 'pip install dwimmy[dev]'.", file=sys.stderr)
        return

    pyproject_file = Path(pyproject_path)
    if not pyproject_file.exists():
        print(f"⚠️  {pyproject_path} not found, skipping update", file=sys.stderr)
        return

    try:
        content = pyproject_file.read_text()
        doc = tomlkit.parse(content)
    except Exception as e:
        print(f"⚠️  Failed to parse {pyproject_path}: {e}", file=sys.stderr)
        return

    # Ensure [tool.dwimmy] exists
    if 'tool' not in doc:
        doc.add('tool', tomlkit.table())
    tool_table = doc['tool']
    if 'dwimmy' not in tool_table:
        tool_table.add('dwimmy', tomlkit.table())
    dwimmy_table = tool_table['dwimmy']

    if embeddings_file:
        dwimmy_table['embeddings-file'] = embeddings_file
    if model_dir:
        dwimmy_table['model-dir'] = model_dir

    # Add to package-data for setuptools
    if 'tool' in doc and 'setuptools' in doc['tool']:
        project_name = doc.get('project', {}).get('name')
        if project_name:
            pkg_name = project_name.replace('-', '_')
            
            if 'package-data' not in doc['tool']['setuptools']:
                doc['tool']['setuptools'].add('package-data', tomlkit.table())
            
            if pkg_name not in doc['tool']['setuptools']['package-data']:
                 doc['tool']['setuptools']['package-data'].add(pkg_name, tomlkit.array())

            package_data = doc['tool']['setuptools']['package-data'][pkg_name]
            
            # Add embeddings file if it's not already there
            if embeddings_file and embeddings_file not in package_data:
                package_data.append(embeddings_file)
            
            # Add model directory glob if it's not already there
            if model_dir:
                model_glob = f"{Path(model_dir).name}/**"
                if model_glob not in package_data:
                    package_data.append(model_glob)

    try:
        pyproject_file.write_text(tomlkit.dumps(doc))
        print(f"✓ Updated {pyproject_path} with dwimmy configuration", file=sys.stderr)
    except Exception as e:
        print(f"⚠️  Failed to write to {pyproject_path}: {e}", file=sys.stderr)



def main():
    """Main entry point for dwimmy CLI"""
    parser = argparse.ArgumentParser(
        description='dwimmy - Semantic CLI initialization tool',
        prog='dwimmy'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # generate command
    gen_parser = subparsers.add_parser(
        'generate',
        help='Generate embeddings and ONNX model for your CLI package'
    )
    gen_parser.add_argument(
        '--modules',
        nargs='+',
        help='Module paths to scan for CLI definitions (e.g., myapp.cli myapp.commands)'
    )
    gen_parser.add_argument(
        '--output',
        default='.dwimmy-embeddings',
        help='Output file for embeddings (default: .dwimmy-embeddings)'
    )
    gen_parser.add_argument(
        '--pyproject',
        default='pyproject.toml',
        help='Path to pyproject.toml (default: pyproject.toml)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'generate':
        success = generate_embeddings(
            parser_modules=args.modules,
            output_file=args.output,
            pyproject_path=args.pyproject
        )
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

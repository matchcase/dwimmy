"""Command-line interface for dwimmy - for package developers"""

import sys
import argparse
from pathlib import Path
import pickle
import json
from typing import List, Optional


def generate_embeddings(
    parser_modules: Optional[List[str]] = None,
    output_file: str = '.dwimmy-embeddings',
    pyproject_path: str = 'pyproject.toml'
):
    """Generate embeddings and ONNX model for a CLI package"""
    from dwimmy.core.matcher import SemanticMatcher
    from dwimmy.core.utils import SimpleSpinner, is_interactive
    
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
    
    # Extract all CLI components
    all_components = _extract_cli_components(parser_modules)
    
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


def _extract_cli_components(modules: List[str]) -> List[str]:
    """Extract all CLI component names from modules"""
    from dwimmy.argparse import ArgumentParser as DwimArgumentParser
    import argparse
    import importlib
    
    components = set()
    
    for module_name in modules:
        # Handle both module.py and module formats
        module_name = module_name.replace('.py', '').replace('/', '.')
        
        try:
            # Import the module (reload if already imported)
            module = __import__(module_name, fromlist=[None])
            # Reload to get fresh version
            importlib.reload(module)
            
            # Find all ArgumentParser instances
            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                    
                    # Check if it's an ArgumentParser (includes DwimArgumentParser)
                    if isinstance(attr, argparse.ArgumentParser):
                        parser = attr
                        
                        # Extract flags
                        for action in parser._actions:
                            if action.option_strings:
                                components.update(action.option_strings)
                            
                            # Extract choices
                            if action.choices:
                                components.update(str(c) for c in action.choices)
                        
                        # Extract subcommands
                        for action in parser._actions:
                            if isinstance(action, argparse._SubParsersAction):
                                components.update(action.choices.keys())
                except (AttributeError, TypeError):
                    # Skip attributes that can't be inspected
                    pass
        except Exception as e:
            print(f"⚠️  Failed to extract from {module_name}: {e}", file=sys.stderr)
    
    # Filter out help flags
    components = {c for c in components if c not in ['-h', '--help', '--version']}
    
    return list(components)


def _update_pyproject(
    pyproject_path: str, 
    embeddings_file: Optional[str] = None,
    model_dir: Optional[str] = None
):
    """Update pyproject.toml to include dwimmy files and package data."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            print("⚠️  toml library not available, skipping pyproject.toml update", file=sys.stderr)
            return

    try:
        with open(pyproject_path, 'rb') as f:
            content = f.read()
        pyproject = tomllib.loads(content.decode())
    except FileNotFoundError:
        print(f"⚠️  {pyproject_path} not found, skipping update", file=sys.stderr)
        return
    except Exception as e:
        print(f"⚠️  Failed to parse {pyproject_path}: {e}", file=sys.stderr)
        return

    # Add dwimmy config if not present
    if 'tool' not in pyproject:
        pyproject['tool'] = {}
    if 'dwimmy' not in pyproject['tool']:
        pyproject['tool']['dwimmy'] = {}

    if embeddings_file:
        pyproject['tool']['dwimmy']['embeddings-file'] = embeddings_file
    if model_dir:
        pyproject['tool']['dwimmy']['model-dir'] = model_dir

    # Add to package-data (for setuptools)
    if 'project' in pyproject and 'tool' in pyproject and 'setuptools' in pyproject['tool']:
        project_name = pyproject.get('project', {}).get('name')
        if project_name:
            pkg_name = project_name.replace('-', '_')
            
            if 'package-data' not in pyproject['tool']['setuptools']:
                pyproject['tool']['setuptools']['package-data'] = {}
            if pkg_name not in pyproject['tool']['setuptools']['package-data']:
                pyproject['tool']['setuptools']['package-data'][pkg_name] = []

            package_data = pyproject['tool']['setuptools']['package-data'][pkg_name]
            
            # Add embeddings file if it's not already there
            if embeddings_file and embeddings_file not in package_data:
                package_data.append(embeddings_file)
            
            # Add model directory glob if it's not already there
            if model_dir:
                model_glob = f"{Path(model_dir).name}/**"
                if model_glob not in package_data:
                    package_data.append(model_glob)

    _write_toml_safe(pyproject_path, pyproject)
    print(f"✓ Updated {pyproject_path} with dwimmy configuration", file=sys.stderr)


def _write_toml_safe(pyproject_path: str, pyproject: dict):
    """Write TOML file preserving as much formatting as possible."""
    try:
        import toml
        with open(pyproject_path, 'w') as f:
            toml.dump(pyproject, f)
    except ImportError:
        print("⚠️  `toml` library not found, cannot write to pyproject.toml.", file=sys.stderr)
        print("   Please add `toml` to your dev dependencies.", file=sys.stderr)



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

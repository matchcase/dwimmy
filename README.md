<div align="center">
  <img src="dwimmy.svg" alt="Dwimmy Banner" width="600"/>
</div>

<h1 align="center">Dwimmy</h1>

<div align="center">
  <strong>Semantic "Do What I Mean" for your CLI</strong>
  <br />
  <p>Stop correcting typos. Start understanding intent.</p>
</div>

<div align="center">

<!--
[![PyPI Version](https://img.shields.io/pypi/v/dwimmy.svg)](https://pypi.org/project/dwimmy/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/matchcase/dwimmy/ci.yml?branch=main)](https://github.com/matchcase/dwimmy/actions)
-->
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://www.python.org)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

---

**Dwimmy** is a Python library that gives your command-line interface (CLI) a "Do What I Mean" (DWIM) capability. Instead of just failing on typos or slightly incorrect phrasing, Dwimmy uses lightweight, pre-computed sentence embeddings to understand the user's intent and suggest the correct command, flag, or argument.

It's designed in such a way that only the developer needs to download a complete deep learning library (sentence-transformers) and the user can enjoy a lightweight experience using ONNX runtime.

## Key Features

- **Natural Language Correction**: Understands user intent beyond exact string matching.
- **Extremely Lightweight Runtime**: End-users only need `numpy` and `onnxruntime`. No heavy libraries like PyTorch or Sentence-Transformers are needed in production.
- **Broad Framework Support**: Works with `argparse`, `click`, and `typer`.
- **Developer-Friendly**: A simple `dwimmy generate` command scans your code, generates embeddings, and packages a lightweight ONNX model for you.
- **Configurable**: The underlying sentence-transformer model can be configured in your `pyproject.toml`.

## How It Works

Dwimmy operates in two stages:

1.  **Development Time (`dwimmy generate`)**:
    - The CLI developer runs `dwimmy generate`.
    - This command scans the specified CLI modules (`argparse`, `click`, or `typer` apps).
    - It extracts all command names, flags, and choices.
    - It uses a powerful `sentence-transformers` model to create semantic vector embeddings for all these components.
    - It then exports the transformer model to the lightweight ONNX format and saves it along with the embeddings. These artifacts are included in your Python package.

2.  **Runtime (End-User Environment)**:
    - When an end-user of your CLI provides an unknown argument, your CLI can call Dwimmy's runtime functions.
    - Dwimmy loads the pre-computed embeddings and the fast ONNX model.
    - It generates an embedding for the user's input and uses cosine similarity to find the most likely intended command or flag from the pre-computed list.
    - It can then suggest the correction to the user (e.g., "Unknown command 'comit'. Did you mean 'commit'?").

## Installation

For end-users of a Dwimmy-enabled CLI, no special installation is required beyond the CLI tool itself, as long as the developer has listed `dwimmy` as a dependency.

For CLI developers who want to integrate Dwimmy:

```bash
# Install with dev dependencies to get the 'generate' command
pip install "dwimmy[dev]"
```

## Usage for CLI Developers

Integrating Dwimmy into your project is a three-step process.

### 1. Add Dwimmy as a Dependency

In your `pyproject.toml`, add `dwimmy` to your list of dependencies.

```toml
[project]
# ...
dependencies = [
    "dwimmy>=0.1.0",
    # other dependencies...
]
```

### 2. Configure your Project

In your `pyproject.toml`, add a `[tool.dwimmy]` section to tell Dwimmy where to find your CLI code.

```toml
[tool.dwimmy]
# A list of modules where your argparse, click, or typer apps are defined.
cli-modules = ["my_cli.main"]

# (Optional) Specify a custom sentence-transformer model.
# Defaults to 'all-MiniLM-L6-v2'.
model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
```

### 3. Generate Embeddings

Run the `generate` command from your project's root directory (where `pyproject.toml` is located).

```bash
dwimmy generate
```

This will:
1.  Read your `[tool.dwimmy]` configuration.
2.  Scan the modules in `cli-modules`.
3.  Generate embeddings and an ONNX model in a `.dwimmy/` directory.
4.  Create a `.dwimmy-embeddings` file.
5.  Update your `pyproject.toml` to include these new files in your package data.

### 4. Use the Runtime

You can now use Dwimmy's runtime functions to find suggestions. For example, in your `argparse` setup:

```python
# In your CLI's main entry point
from dwimmy.argparse import ArgumentParser

# Use our custom ArgumentParser
parser = ArgumentParser(description="My CLI")
# ... add your arguments ...

# The custom parser automatically handles unknown arguments
# and provides suggestions.
args = parser.parse_args()
```

For more advanced usage with `click` or `typer`, you can use the runtime functions directly:

```python
from dwimmy.core.matcher import SemanticMatcher

def find_suggestion(user_input: str, available_commands: list[str]):
    matcher = SemanticMatcher()
    suggestion = matcher.find_closest_runtime(user_input, candidates=available_commands)
    if suggestion:
        print(f"Command not found. Did you mean '{suggestion[0]}'?")

```

## License

This project is licensed under the terms of the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for the full license text.

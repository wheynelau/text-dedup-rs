#!/bin/bash
# Test script for running Rust tests that call Python functions
# This requires a virtual environment with text_dedup installed

set -e

# Activate virtual environment
source .venv/bin/activate

# Set up Python path to include current directory and site packages
export PYTHONPATH=$(pwd):$(python3 -c "import site; print(':'.join(site.getsitepackages()))")

# Set PyO3 Python interpreter
export PYO3_PYTHON=$(which python3)

# Set Rust flags for linking Python on macOS arm64
export RUSTFLAGS="-L /opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib -l python3.13"

# Run the Python comparison tests
cargo test --lib python_comparison "$@"

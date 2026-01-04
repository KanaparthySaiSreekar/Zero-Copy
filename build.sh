#!/bin/bash
set -e

echo "Building Zero-Copy Vector Search Engine..."

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Build in release mode
echo "Building Rust extension..."
maturin develop --release

echo "Build complete! You can now use 'import zero_copy_search' in Python."

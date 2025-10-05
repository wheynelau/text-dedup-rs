# Optimising Text Deduplication in Rust

This is a Rust port of the [text-dedup](https://github.com/ChenghaoMou/text-dedup) project.

**Note:** This repository uses [dedup-rs](https://github.com/wheynelau/dedup-rs) as a git submodule for the Rust CLI binary. The Python bindings are published separately to PyPI from the dedup-rs repository. Use both repositories in conjunction:
- [dedup-rs](https://github.com/wheynelau/dedup-rs) - Rust implementation
- [text-dedup-rs](https://github.com/wheynelau/text-dedup-rs) - Python package (this repo)
- [Optimising Text Deduplication in Rust](#optimising-text-deduplication-in-rust)
  - [Description](#description)
  - [How to run](#how-to-run)
  - [Changes made to original code](#changes-made-to-original-code)
  - [Results](#results)
  - [Issues](#issues)
  - [TODO](#todo)


## Description
The original algorithrm is from [here](https://github.com/ChenghaoMou/text-dedup) and I just ported it to Rust.  
It uses a minhash LSH algorithm to find similar documents.  

## How to run

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/wheynelau/text-dedup-rs
# OR if already cloned:
git submodule update --init

# Install Python package (Python bindings from PyPI)
pip install .

# Build Rust CLI binary (optional, only if using minhash_pure_rs.py)
cd rust && cargo build --release -p dedup-bin

# Run tests
python tests/benchmark_core.py
```

## Changes made to original code

- Removed pyspark from requirements
- Removed simhash and pyspark runs from benchmark core
- Added a minhashrust and pure rust to benchmark core and news.
- Added a new dataset for speed and memory testing

## Results

Benchmark results are in [BENCHMARKS](docs/BENCHMARKS.md)

## Issues

See [ISSUES](docs/ISSUES.md)

## TODO
See [FUTURE](docs/FUTURE.md)

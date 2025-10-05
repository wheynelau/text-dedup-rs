# Optimising Text Deduplication in Rust

This is a Rust port of the [text-dedup](https://github.com/ChenghaoMou/text-dedup) project.
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
# Install Rust if needed
curl https://sh.rustup.rs -sSf | sh

# Build Python extension
pip install .

# Build standalone binary
cargo build --release -p dedup-bin

# Run tests
python tests/benchmark_core.py
```

**Note:** This project uses a Cargo workspace with three crates:
- `dedup-core`: Pure Rust library (shared code)
- `dedup-py`: Python extension (requires `maturin`)
- `dedup-bin`: Standalone binary (no Python dependency)

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

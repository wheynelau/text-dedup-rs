# Optimising Text Deduplication in Rust

This is a Rust port of the [text-dedup](https://github.com/ChenghaoMou/text-dedup) project.
- [Optimising Text Deduplication in Rust](#optimising-text-deduplication-in-rust)
  - [Description](#description)
  - [Learning points](#learning-points)
  - [How to run](#how-to-run)
  - [Changes made to original code](#changes-made-to-original-code)
  - [Results](#results)
  - [Issues](#issues)
  - [TODO](#todo)


## Description
The original algorithrm is from [here](https://github.com/ChenghaoMou/text-dedup) and I just ported it to Rust.  
It uses a minhash LSH algorithm to find similar documents.  

## Learning points

I intend to write a gist for the additional issues for learning purposes.

- flamegraph: Learnt quite a bit about flamegraph when my initial runs were slower than python. Oof
- Concepts I never knew existed in python: Integer overflow, bytes encoding and decoding, memory reallocation, etc.
- Writing tests for rust code

## How to run

```bash
# Install Rust if needed
curl https://sh.rustup.rs -sSf | sh

# Build Python extension
maturin develop --release

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

1. After setting up some deterministic flags, I found that there were still some variations in the results.  

For testing purposes, the parellelized version is kept. The original code is at this [point](https://github.com/wheynelau/text-dedup-rs/blob/b121d1431f657ea71034b07dc39ae3428f363dbd/src/dedup.rs)

## TODO
- [x] Write setup.py -> setup pyproject.toml for `pip install .`
- [ ] Remove hard codes
- [x] ~~End goal: Make it work with pyspark~~ Check out the [experimental-pyspark](https://github.com/wheynelau/text-dedup-rs/tree/experimental-pyspark) branch
- [x] Check for potential improvements
- [ ] Write idiomatic rust code
- [ ] Allow generics for u32,u64,u128
- [x] Move to numpy crate
- [ ] Implement test for u64
- [x] Try to get a closer match on the benchmark scores
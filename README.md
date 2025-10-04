# Optimising Text Deduplication in Rust

> "The First Rule of Program Optimization: Don't do it. The Second Rule of Program Optimization (for experts only!): Don't do it yet." — Michael A. Jackson
> 
> "If it ain't broke, don't fix it." — Bert Lance

This is a Rust port of the [text-dedup](https://github.com/ChenghaoMou/text-dedup) project.
  - [Description](#description)
  - [Learning points](#learning-points)
  - [How to run](#how-to-run)
    - [Docker](#docker)
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

I have not tested on another fresh environment so expect "It works on my machine" issues.  
Additionally, I have not setup wheels so only building from source is possible.

```bash
# from the original repo
# this assumes you have python, environment management is up to you
curl https://sh.rustup.rs -sSf | sh # installing rust
pip install .
python tests/benchmark_core.py
```
### Docker

```bash
docker build -t text-dedup .
docker run text-dedup "python tests/benchmark_core.py"
docker run text-dedup "python tests/benchmark_news.py"
```


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
- [ ] Check for potential improvements
- [ ] Write idiomatic rust code
- [ ] Allow generics for u32,u64,u128
- [ ] Move to numpy crate
- [ ] Implement test for u64
- [ ] Try to get a closer match on the benchmark scores
- 
# MAKING DEDUP GO BRRRRRR 🚀🚀🚀 (... somewhat)

> "The First Rule of Program Optimization: Don't do it. The Second Rule of Program Optimization (for experts only!): Don't do it yet." — Michael A. Jackson
> 
> "If it ain't broke, don't fix it." — Bert Lance

- [MAKING DEDUP GO BRRRRRR 🚀🚀🚀 (... somewhat)](#making-dedup-go-brrrrrr---somewhat)
  - [Description](#description)
  - [Learning points](#learning-points)
  - [How to run](#how-to-run)
  - [Changes made to original code](#changes-made-to-original-code)
  - [Results](#results)
  - [TODO:](#todo)


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

```bash
# from the original repo
# this assumes you have python, environment management is up to you
pip install -e . # editable mode
cd dedup-rs
maturin build --release # wondering if this is correct
pip install .
cd ..
python tests/benchmark_core.py
```

## Changes made to original code

- Removed pyspark from requirements
- Removed simhash and pyspark runs from benchmark core
- Added a minhashrust run to benchmark core

## Results

| Algorithm   |   Precision (Duplicates) |   Recall (Duplicates) |   Precision (Non Duplicates) |   Recall (Non Duplicates) |   Macro F1 score |   Accuracy | Time   |
|:------------|-------------------------:|----------------------:|-----------------------------:|--------------------------:|-----------------:|-----------:|:-------|
| MinHash     |                   0.9594 |                0.9445 |                       0.9474 |                    0.9616 |           0.9534 |     0.924  | 22.82s |
| MinHashRust |                   0.9572 |                0.9426 |                       0.946  |                    0.9598 |           0.9516 |     0.9284 | 13.38s |
| Exact Title |                   0.8302 |                0.5521 |                       0.7098 |                    0.9065 |           0.77   |     0.7456 | -      |

rust:  
 INFO     Fused embedding, sharding       : 5.08s timer.py:65  
 INFO     Clustering                      : 0.40s timer.py:65  
 INFO     Filtering                       : 2.94s timer.py:65  
 INFO     Saving                          : 1.37s timer.py:65  
 INFO     Cleaning                        : 0.02s timer.py:65  
 INFO     Total                           : 12.15s timer.py:65  
 INFO     Before                          : 100000 minhash_rust.py:123  
 INFO     After                           : 72495   
python:  
 INFO     MinHashing                      : 6.60s timer.py:65  
 INFO     Sharding                        : 8.05s timer.py:65  
 INFO     Clustering                      : 1.20s timer.py:65  
 INFO     Filtering                       : 2.64s timer.py:65  
 INFO     Saving                          : 0.75s timer.py:65  
 INFO     Cleaning                        : 0.03s timer.py:65  
 INFO     Total                           : 21.41s timer.py:65  
 INFO     Before                          : 100000 minhash.py:306  
 INFO     After                           : 72208    
## TODO:
- [ ] Write setup.py
- [ ] Remove hard codes
- [ ] End goal: Make it work with pyspark
- [ ] Check for potential improvements
- [ ] Write idiomatic rust code

<center><img src="https://camo.githubusercontent.com/adc94e53c011a5ee3606dad1223c776c169d32e26055a6d9a01ef28fb0a55964/68747470733a2f2f70617065722d6174746163686d656e74732e64726f70626f782e636f6d2f735f353445314239364546464546443239343536323930324443354239393731443335434436423635304243383744313230303341333041343635313737363230315f313538363531353635343537335f737469636b65722e77656270"/ style="background-color:white;"></center>

# MAKING DEDUP GO BRRRRRR 🚀🚀🚀 (... somewhat)

> "The First Rule of Program Optimization: Don't do it. The Second Rule of Program Optimization (for experts only!): Don't do it yet." — Michael A. Jackson
> 
> "If it ain't broke, don't fix it." — Bert Lance

- [MAKING DEDUP GO BRRRRRR 🚀🚀🚀 (... somewhat)](#making-dedup-go-brrrrrr---somewhat)
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
curl https://sh.rustup.rs -sSf | sh 
pip install . # editable mode
cd dedup-rs
pip install .
cd ..
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

1. One caveat I noticed was that rust didn't produce tuples with whitespaces, for example  ("a", "b" "c") instead of  
 (" ", "a", "b") which is what python produced. It may affect the results but I'm not sure.

2. When testing on a larger dataset, the rust code removed significantly lesser data () than the python code.
As the data was not labelled, it could be not be verified if the duplicates were removed correctly.

3. Using par_iter for some portions of the rust code lead to small differences in the results. 

For benchmark core:
```
# With par_iter on clustering and hashing
| Algorithm     |   Precision (Duplicates) |   Recall (Duplicates) |   Precision (Non Duplicates) |   Recall (Non Duplicates) |   Macro F1 score |   Accuracy | Time   |
| MinHashRust   |                   0.9565 |                0.9432 |                       0.9466 |                    0.9591 |           0.9513 |     0.9292 | 15.04s |
| MinHashPureRS |                   0.9555 |                0.9431 |                       0.9466 |                    0.9582 |           0.9508 |     0.9293 | 6.86s  |

# Without par_iter on clustering and hashing
| MinHashRust   |                   0.9565 |                0.9432 |                       0.9466 |                    0.9591 |           0.9513 |     0.9292 | 15.04s |
| MinHashPureRS |                   0.9565 |                0.9432 |                       0.9466 |                    0.9592 |           0.9513 |     0.9292 | 9.14s  |

# On a larger dataset, the difference was more pronounced in the final len from 930460
MinHashPureRS: 844493
MinHashRust: 860760
```

For testing purposes, the parellelized version is kept. The original code is at this [point](https://github.com/wheynelau/text-dedup-rs/blob/b121d1431f657ea71034b07dc39ae3428f363dbd/src/dedup.rs)

## TODO
- [x] Write setup.py -> setup pyproject.toml for `pip install .`
- [ ] Remove hard codes
- [x] ~~End goal: Make it work with pyspark~~ Check out the [experimental-pyspark](https://github.com/wheynelau/text-dedup-rs/tree/experimental-pyspark) branch
- [ ] Check for potential improvements
- [ ] Write idiomatic rust code

# Benchmarks

- [Benchmarks](#benchmarks)
  - [Benchmark core](#benchmark-core)
  - [Benchmark news](#benchmark-news)
  - [Benchmark RP](#benchmark-rp)

## Benchmark core

Found that after fixing the u32 to u64, got better results
And setting deterministic flags, got better results as well.

| Algorithm     |   Precision (Duplicates) |   Recall (Duplicates) |   Precision (Non Duplicates) |   Recall (Non Duplicates) |   Macro F1 score |   Accuracy | Time   |
|:--------------|-------------------------:|----------------------:|-----------------------------:|--------------------------:|-----------------:|-----------:|:-------|
| MinHash       |                   0.8537 |                0.9488 |                       0.946  |                    0.8463 |           0.8961 |     0.8197 | 11.39s |
| MinHashRust   |                   0.8537 |                0.9489 |                       0.946  |                    0.8464 |           0.8961 |     0.8197 | 5.67s  |
| MinHashPureRS |                   0.8537 |                0.9489 |                       0.946  |                    0.8463 |           0.8961 |     0.8197 | 3.22s  |
| Exact Title   |                   0.8302 |                0.5521 |                       0.7098 |                    0.9065 |           0.77   |     0.7456 | -      |

Python 10.82s  
Hybrid 4.74s  
Rust 3.21s  

## Benchmark news

MinHash ARI: 0.7228948156767026
MinRust ARI: 0.7244780630042161
MinHash Pure RS ARI: 0.7244780630042161

## Benchmark RP

This dataset has no validation, only used for throughput testing, noticed that values are quite different when the dataset  
is bigger.

Pure rust

INFO     Total                           : 88.87s
INFO     Before                          : 930514
INFO     After                           : 439944 

Rust from datasets  

INFO     Loading                         : 16.46s
INFO     Fused embedding, sharding       : 77.86s
INFO     Clustering                      : 0.39s
INFO     Filtering                       : 4.47s
INFO     Saving                          : 1.45s
INFO     Cleaning                        : 0.15s
INFO     Total                           : 100.78s
INFO     Before                          : 930460
INFO     After                           : 439943

Original python  

INFO     Loading                         : 16.71s
INFO     MinHashing                      : 117.59s
INFO     Sharding                        : 75.02s
INFO     Clustering                      : 3.94s
INFO     Filtering                       : 4.76s
INFO     Saving                          : 1.35s
INFO     Cleaning                        : 0.17s
INFO     Total                           : 219.53s
INFO     Before                          : 930460
INFO     After                           : 440263


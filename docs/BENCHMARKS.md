# Benchmarks

- [Benchmarks](#benchmarks)
  - [Benchmark core](#benchmark-core)
  - [Benchmark news](#benchmark-news)
  - [Benchmark RP](#benchmark-rp)

## Benchmark core

Found that after fixing the u32 to u64, got better results

| Algorithm     |   Precision (Duplicates) |   Recall (Duplicates) |   Precision (Non Duplicates) |   Recall (Non Duplicates) |   Macro F1 score |   Accuracy | Time   |
|:--------------|-------------------------:|----------------------:|-----------------------------:|--------------------------:|-----------------:|-----------:|:-------|
| MinHash       |                   0.9594 |                0.9445 |                       0.9474 |                    0.9616 |           0.9532 |     0.924  | 11.99s |
| MinHashRust   |                   0.9585 |                0.9449 |                       0.9478 |                    0.9606 |           0.9529 |     0.9245 | 5.99s  |
| MinHashPureRS |                   0.9592 |                0.9449 |                       0.9478 |                    0.9613 |           0.9533 |     0.9245 | 2.98s  |
| Exact Title   |                   0.8302 |                0.5521 |                       0.7098 |                    0.9065 |           0.77   |     0.7456 | -      |

## Benchmark news

MinHash ARI: 0.7228948156767026
MinRust ARI: 0.7533973513900195
MinHash Pure RS ARI: 0.7533973513900195

## Benchmark RP

This dataset has no validation, only used for throughput testing

Rust from datasets

INFO     Loading                         : 16.60s
INFO     Fused embedding, sharding       : 83.78s
INFO     Clustering                      : 0.39s
INFO     Filtering                       : 4.88s
INFO     Saving                          : 2.05s
INFO     Cleaning                        : 0.13s
INFO     Total                           : 107.84s
INFO     Before                          : 930460
INFO     After                           : 659598

Original python

INFO     Loading                         : 17.14s
INFO     MinHashing                      : 129.99s
INFO     Sharding                        : 80.11s 
INFO     Clustering                      : 3.83s  
INFO     Filtering                       : 4.60s  
INFO     Saving                          : 1.87s  
INFO     Cleaning                        : 0.17s   
INFO     Total                           : 237.72s 
INFO     Before                          : 930460  
INFO     After                           : 631281 


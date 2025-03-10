# Benchmarks

- [Benchmarks](#benchmarks)
  - [Benchmark core](#benchmark-core)
  - [Benchmark news](#benchmark-news)
  - [Benchmark RP](#benchmark-rp)

## Benchmark core

Found that after fixing the u32 to u64, got better results

| Algorithm     |   Precision (Duplicates) |   Recall (Duplicates) |   Precision (Non Duplicates) |   Recall (Non Duplicates) |   Macro F1 score |   Accuracy | Time   |
|:--------------|-------------------------:|----------------------:|-----------------------------:|--------------------------:|-----------------:|-----------:|:-------|
| MinHash       |                   0.9594 |                0.9445 |                       0.9474 |                    0.9616 |           0.9532 |     0.924  | 11.14s |
| MinHashRust   |                   0.9579 |                0.944  |                       0.947  |                    0.9602 |           0.9522 |     0.9235 | 5.36s  |
| MinHashPureRS |                   0.9577 |                0.944  |                       0.947  |                    0.96   |           0.9521 |     0.9235 | 2.33s  |
| Exact Title   |                   0.8302 |                0.5521 |                       0.7098 |                    0.9065 |           0.77   |     0.7456 | -      |

## Benchmark news

MinHash ARI: 0.7228948156767026
MinRust ARI: 0.7244780630042161
MinHash Pure RS ARI: 0.7244780630042161

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


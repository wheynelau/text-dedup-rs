# Benchmarks

- [Benchmarks](#benchmarks)
  - [Benchmark core](#benchmark-core)
  - [Benchmark news](#benchmark-news)
  - [Speed and memory testing](#speed-and-memory-testing)

## Benchmark core

System specs:
- OS: "Debian GNU/Linux 12 (bookworm)"
- CPU: Intel i5 11400

| Algorithm     |   Precision (Duplicates) |   Recall (Duplicates) |   Precision (Non Duplicates) |   Recall (Non Duplicates) |   Macro F1 score |   Accuracy | Time   |
|:--------------|-------------------------:|----------------------:|-----------------------------:|--------------------------:|-----------------:|-----------:|:-------|
| MinHash       |                   0.9594 |                0.9445 |                       0.9474 |                    0.9616 |           0.9534 |     0.924  | 22.08s |
| MinHashRust   |                   0.9571 |                0.9426 |                       0.946  |                    0.9597 |           0.9516 |     0.9284 | 13.88s |
| MinHashPureRS |                   0.9572 |                0.9426 |                       0.946  |                    0.9598 |           0.9516 |     0.9284 | 8.43s  |
| Exact Title   |                   0.8302 |                0.5521 |                       0.7098 |                    0.9065 |           0.77   |     0.7456 | -      |

MinHash:  
INFO     Loading                         : 2.94s  
INFO     MinHashing                      : 6.42s  
INFO     Sharding                        : 7.45s  
INFO     Clustering                      : 1.25s  
INFO     Filtering                       : 2.49s  
INFO     Saving                          : 0.73s  
INFO     Cleaning                        : 0.09s   
INFO     Total                           : 21.38s  
INFO     Before                          : 100000  
INFO     After                           : 72208   

MinHashRust:  
INFO     Loading                         : 2.14s  
INFO     Fused embedding, sharding       : 4.76s  
INFO     Clustering                      : 0.36s  
INFO     Filtering                       : 3.40s  
INFO     Saving                          : 1.31s  
INFO     Cleaning                        : 0.04s  
INFO     Total                           : 12.01s  
INFO     Before                          : 100000  
INFO     After                           : 72495   

MinHashPureRS:  
INFO     Loading                         : 2.55s  
INFO     Embed                           : 5.88s  
INFO     Total                           : 8.43s  
INFO     Before                          : 100000  
INFO     After                           : 72495  


## Benchmark news

System specs:
Same as above

MinHash ARI: 0.7228948156767026  
MinRust ARI: 0.7696839460383871  
MinHash Pure RS ARI: 0.7696839460383871  

MinHash:
INFO     Loading                         : 0.39s  
INFO     MinHashing                      : 2.17s  
INFO     Sharding                        : 1.18s  
INFO     Clustering                      : 0.08s  
INFO     Filtering                       : 0.90s  
INFO     Saving                          : 0.20s  
INFO     Cleaning                        : 0.01s  
INFO     Total                           : 4.94s  
INFO     Before                          : 14211  
INFO     After                           : 10386  

MinhashRust:
INFO     Loading                         : 0.84s  
INFO     Fused embedding, sharding       : 0.89s  
INFO     Clustering                      : 0.01s  
INFO     Filtering                       : 0.98s  
INFO     Saving                          : 0.24s  
INFO     Cleaning                        : 0.01s  
INFO     Total                           : 2.98s  
INFO     Before                          : 14211 
INFO     After                           : 10518  

MinhashPureRS:
INFO     Loading                         : 0.91s  
INFO     Embed                           : 0.98s  
INFO     Total                           : 1.89s  
INFO     Before                          : 14211  
INFO     After                           : 10518  

## Speed and memory testing

This benchmark uses this dataset "togethercomputer/RedPajama-Data-1T-Sample" which is a sample of the RedPajama dataset.

Speedup:  

Note that this dataset does not have any validation, so the results are not verified.

System specs:
- 128 vCPUs

MinHash:
INFO     Loading                         : 191.34s  
INFO     MinHashing                      : 327.85s  
INFO     Sharding                        : 381.44s  
INFO     Clustering                      : 12.60s   
INFO     Filtering                       : 205.53s  
INFO     Saving                          : 11.44s   
INFO     Cleaning                        : 1.91s    
INFO     Total                           : 1132.11s  
INFO     Before                          : 930460    
INFO     After                           : 631281  

MinHashRust:

INFO     Loading                         : 66.87s  
INFO     Fused embedding, sharding       : 349.13s  
INFO     Clustering                      : 3.43s    
INFO     Filtering                       : 180.63s  
INFO     Saving                          : 19.72s   
INFO     Cleaning                        : 1.19s    
INFO     Total                           : 620.97s 1.82x speedup
INFO     Before                          : 930460  
INFO     After                           : 860760  

MinHashPureRS:
Data received from Rust: {'len': 844493}  
INFO     Total                           : 71.20s  15.88x speedup
INFO     Before                          : 930514  
INFO     After                           : 844493  



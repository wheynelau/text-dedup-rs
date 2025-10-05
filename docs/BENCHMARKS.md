# Benchmarks

- [Benchmarks](#benchmarks)
  - [Benchmark core](#benchmark-core)
  - [Benchmark news](#benchmark-news)
  - [Benchmark RP](#benchmark-rp)
    - [Notes](#notes)

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

This was done on 32 cores. 

This dataset has no validation, only used for throughput testing, noticed that values are quite different when the dataset  
is bigger.

Pure rust

INFO     Total                           : 99.18s  
INFO     Before                          : 930514  
INFO     After                           : 440235  

Rust from datasets  

INFO     Loading                         : 24.28s  
INFO     Fused embedding, sharding       : 82.48s  
INFO     Clustering                      : 4.14s  
INFO     Filtering                       : 3.75s  
INFO     Saving                          : 3.73s  
INFO     Cleaning                        : 0.03s  
INFO     Total                           : 118.40s  
INFO     Before                          : 930460  
INFO     After                           : 440234  

Original python  

INFO     Loading                         : 27.89s  
INFO     MinHashing                      : 158.43s  
INFO     Sharding                        : 84.81s  
INFO     Clustering                      : 5.78s  
INFO     Filtering                       : 49.43s  
INFO     Saving                          : 3.58s  
INFO     Cleaning                        : 0.04s   
INFO     Total                           : 329.96s  
INFO     Before                          : 930460  
INFO     After                           : 440263  

### Notes

1. The difference in Before is due to the python code checking the length of the dataset after filtering out the min_length, as shown in the code:
```python
        with timer("Loading"):
            ds, id2id = load_hf_dataset(io_args=io_args, meta_args=meta_args)
            ds = ds.filter(
                lambda x: len(NON_ALPHA.split(x[meta_args.column].lower())) >= minhash_args.min_length,
                num_proc=io_args.num_proc,
            )

        LEN_DATASET = len(ds)
```

2. The differences in the values are still present, but not as significant, further testing needed.

Perhaps testing with another framework would help.


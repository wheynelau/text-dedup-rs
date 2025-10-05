#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import multiprocessing as mp
import os
import random
import re
from collections import defaultdict
from typing import Any

import click
import datasets
import numpy as np

from text_dedup import logger
from text_dedup.dedup_rs import EmbedFunc
from text_dedup.utils import INDEX_COLUMN
from text_dedup.utils import DisableReferenceCount
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils import MinHashArgs
from text_dedup.utils import Timer
from text_dedup.utils import UnionFind
from text_dedup.utils import load_hf_dataset
from text_dedup.utils import optimal_param

SEED = 42
RNG = np.random.RandomState()
NON_ALPHA = re.compile(r"\W", re.UNICODE)
datasets.logging.set_verbosity_error()
# for is originally used to reduce memory usage in MacOS but also ensures that the Union Find data structure
# is not copied to child processes as long as it is not modified.
mp.set_start_method("fork", force=True)
uf = UnionFind()
SIGNATURE_COLUMN = "__signatures__"


@click.command
@IOArgs.option_group
@MetaArgs.option_group
@MinHashArgs.option_group
def main(
    io_args: IOArgs,
    meta_args: MetaArgs,
    minhash_args: MinHashArgs,
):
    global uf
    uf.reset()

    HASH_BITS: int = minhash_args.hash_bits

    # 64 bit config is backwards compatibility mode.
    # it uses 64 bit types but almost entirely 32bit data, except for one mersenne prime 2^61
    # why legacy implementations used mersenne primes for modulo:
    # https://en.wikipedia.org/wiki/Universal_hashing#Hashing_strings
    HASH_CONFIG: dict[int, tuple[type, Any, Any]] = {
        64: (np.uint64, np.uint32((1 << 32) - 1), np.uint64((1 << 61) - 1)),
        # 32, 16 bit config does not use a mersenne prime.
        # The original reason for using mersenne prime was speed.
        # Testing reveals, there is no benefit to using a 2^61 mersenne prime for division
        32: (np.uint32, np.uint32((1 << 32) - 1), np.uint32((1 << 32) - 5)),
        16: (np.uint16, np.uint16((1 << 16) - 1), np.uint16((1 << 16) - 15)),
    }

    # defaults to backwards compatible HASH_BITS = 64, which is np.uint64 dtypes with 32bit hashes
    DTYPE, MAX_HASH, MODULO_PRIME = HASH_CONFIG.get(HASH_BITS, HASH_CONFIG[64])

    timer = Timer()

    if minhash_args.b is not None and minhash_args.r is not None:
        B, R = minhash_args.b, minhash_args.r
    else:
        # Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
        # of probabilities of false positive and false negative, taken from datasketch.
        # You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.
        # The following assumes a "perfect hash". using 16 bit hashes might challenge this assumption
        # lower precision dtype will cause more collisions, so higher false_positives and less false negatives.
        # Both effects move the result towards more documents being considered duplicates.
        B, R = optimal_param(
            minhash_args.threshold,
            minhash_args.num_perm,
            false_positive_weight=0.5,
            false_negative_weight=0.5,
        )

    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]

    # for minhash, we need to make a lot of hashes(=num_perms).
    # In many previous implementations, this is achieved through a method described in
    # `Universal classes of hash functions` https://doi.org/10.1016/0022-0000(79)90044-8
    # There we start with a know good hash x (=hash_func) and permutate it as the following:
    # `new_hash = (a * x + b) mod prime mod max_hash` we need one a (!=0), b pair per new hash
    # the following produces these a, b pairs
    if os.getenv("DETERMINISTIC", "0") == "1":
        a = np.arange((1 << 32), (1 << 32) + minhash_args.num_perm, dtype=DTYPE)
        b = np.arange((1 << 32), (1 << 32) + minhash_args.num_perm, dtype=DTYPE)
        PERMUTATIONS = (a, b)
    else:
        PERMUTATIONS: tuple[np.ndarray, np.ndarray] = (
            RNG.randint(
                1, MODULO_PRIME, size=(minhash_args.num_perm,), dtype=DTYPE
            ),  # a is a multiplier so should not be 0
            RNG.randint(0, MODULO_PRIME, size=(minhash_args.num_perm,), dtype=DTYPE),  # b
        )

    Emb = EmbedFunc.from_permutations(
        n_grams=minhash_args.ngram,
        min_len=minhash_args.min_length,
        hashranges=HASH_RANGES,
        permutations=PERMUTATIONS,
    )

    timer = Timer()

    with timer("Total"):
        with timer("Loading"):
            ds, id2id = load_hf_dataset(io_args=io_args, meta_args=meta_args)
            ds = ds.filter(
                lambda x: len(NON_ALPHA.split(x[meta_args.column].lower())) >= minhash_args.min_length,
                num_proc=io_args.num_proc,
            )

        LEN_DATASET = len(ds)

        with timer("Fused embedding, sharding"):

            def batch_embed_shard(records, idx):
                Emb.batch_embed_shard(records, np.array(idx, dtype=np.uint32))

            ds.map(
                function=batch_embed_shard,
                input_columns=[meta_args.column, INDEX_COLUMN],
                remove_columns=[col for col in ds.column_names if col != INDEX_COLUMN],
                batched=True,
                batch_size=len(ds),  # use the full dataset as batch size
                with_indices=False,
                desc="Fingerprinting with rust...",
            )

        with timer("Clustering"):
            uf = Emb.cluster()

        with timer("Filtering"), DisableReferenceCount():
            # use rust to filter
            indices_array = np.array(ds[INDEX_COLUMN], dtype=np.uint32)
            keep_positions = Emb.filter_duplicates(uf, indices_array)

            # use .select() which is much faster than .map() + .filter()
            final_data = ds.select(keep_positions)
            logger.info(f"Number of edges: {uf.edges}")

        with timer("Saving"):
            final_data = final_data.remove_columns([INDEX_COLUMN])
            final_data.save_to_disk(io_args.output)
            if io_args.debug:
                uf.dump(os.path.join(io_args.output, "uf.json"))

        with timer("Cleaning"):
            if io_args.clean_cache:
                ds.cleanup_cache_files()
                final_data.cleanup_cache_files()

    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"{'Before':<{PAD}}: {LEN_DATASET}")
    logger.info(f"{'After':<{PAD}}: {len(final_data)}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()

#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import multiprocessing as mp
import os
import random
import re
import time

import click
import datasets
import numpy as np
from tqdm import tqdm

from dedup_rs import EmbedFunc
from text_dedup import logger
from text_dedup.utils import (CLUSTER_COLUMN, INDEX_COLUMN,
                              DisableReferenceCount, IOArgs, MetaArgs,
                              MinHashArgs, Timer, UnionFind, load_hf_dataset,
                              optimal_param)

SEED = 42
RNG = np.random.RandomState(SEED)
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

    timer = Timer()

    with timer("Total"):
        with timer("Loading"):
            ds, id2id = load_hf_dataset(io_args=io_args, meta_args=meta_args)
            ds = ds.filter(
                lambda x: len(NON_ALPHA.split(x[meta_args.column].lower()))
                >= minhash_args.min_length,
                num_proc=io_args.num_proc,
            )
        Emb = EmbedFunc.from_b_r(
            B, R, minhash_args.num_perm, SIGNATURE_COLUMN, INDEX_COLUMN
        )

        LEN_DATASET = len(ds)

        with timer("Fused embedding, sharding"):
            ds.map(
                Emb.batch_embed_shard,
                input_columns=[meta_args.column, INDEX_COLUMN],
                remove_columns=[col for col in ds.column_names if col != INDEX_COLUMN],
                batched=True,
                batch_size=10000,
                with_indices=False,
                desc="Fingerprinting with rust...",
            )

        with timer("Clustering"):
            uf = Emb.cluster()

        with timer("Filtering"), DisableReferenceCount():
            ds = ds.map(
                function=lambda record: {CLUSTER_COLUMN: uf.find(record[INDEX_COLUMN])},
                with_indices=False,
                num_proc=io_args.num_proc,
                new_fingerprint=str(random.getrandbits(128)),
                desc="Finding clusters...",
            )
            # This is where the deduplication happens
            # Since there is no easy groupby in datasets
            # I will use this simple filter for now
            final_data = ds.filter(
                function=lambda record: record[CLUSTER_COLUMN] == record[INDEX_COLUMN],
                with_indices=False,
                num_proc=io_args.num_proc,
                desc="Filtering clusters...",
            )

        with timer("Saving"):
            final_data = final_data.remove_columns([CLUSTER_COLUMN, INDEX_COLUMN])
            final_data.save_to_disk(io_args.output)
            if io_args.debug:
                UserWarning(
                    "Saving UnionFind not implemented yet. Sleeping for fairness"
                )
                time.sleep(2)
                uf.dump(os.path.join(io_args.output, "uf.pkl"), id2id=id2id)

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

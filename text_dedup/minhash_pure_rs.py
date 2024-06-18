#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations
import subprocess
import multiprocessing as mp
import os
import random
import re
import json
import click
import datasets
import numpy as np

from text_dedup import logger
from text_dedup.dedup_rs import UnionFind as UnionFindRS
from text_dedup.utils import (CLUSTER_COLUMN, INDEX_COLUMN,
                              DisableReferenceCount, IOArgs, MetaArgs,
                              MinHashArgs, Timer, UnionFind, load_hf_dataset,
                              optimal_param)

mp.set_start_method("fork", force=True)

NON_ALPHA = re.compile(r"\W", re.UNICODE)

@click.command
@IOArgs.option_group
@MetaArgs.option_group
@MinHashArgs.option_group
def main(
    io_args: IOArgs,
    meta_args: MetaArgs,
    minhash_args: MinHashArgs,
):

    if minhash_args.b is not None and minhash_args.r is not None:
        B, R = minhash_args.b, minhash_args.r

    timer = Timer()
    with timer("Total"):
        with timer("Loading"):
            ds, id2id = load_hf_dataset(io_args=io_args, meta_args=meta_args)
            ds = ds.filter(
                lambda x: len(NON_ALPHA.split(x[meta_args.column].lower()))
                >= minhash_args.min_length,
                num_proc=io_args.num_proc,
            )
        LEN_DATASET = len(ds)
        with timer("Embed"):
            command = ("./target/release/dedup "
                    "--b {} --r {} --num-perm {} --uf-output {}".format(
                minhash_args.b, minhash_args.r, minhash_args.num_perm, os.path.join(io_args.output, "uf.json")
            )
            )
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                print("Data received from Rust:", data)

    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"{'Before':<{PAD}}: {LEN_DATASET}")
    logger.info(f"{'After':<{PAD}}: {data['len']}")
if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()
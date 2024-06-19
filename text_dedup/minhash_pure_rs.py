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
from text_dedup.utils import (
    CLUSTER_COLUMN,
    INDEX_COLUMN,
    DisableReferenceCount,
    IOArgs,
    MetaArgs,
    MinHashArgs,
    Timer,
    UnionFind,
    load_hf_dataset,
    optimal_param,
)

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
    parquet_path: str,
):

    timer = Timer()
    binary_path = "target/release/dedup"
    if not os.path.exists(binary_path):
        command = "cargo build --release"
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            print(result.stderr)
            return
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
            # check if exists

            command = (
                f"{binary_path} "
                "--b {} --r {} --num-perm {} --parquet-path {} --main-col {} --idx-col {} --uf-output {}".format(
                    minhash_args.b,
                    minhash_args.r,
                    minhash_args.num_perm,
                    parquet_path,
                    meta_args.column,
                    meta_args.idx_column if meta_args.idx_column else "id",
                    os.path.join(io_args.output, "uf.json"),
                )
            )
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            result = result.stdout.split("\n")[-1]
            data = json.loads(result)
            print("Data received from Rust:", data)

    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"{'Before':<{PAD}}: {LEN_DATASET}")
    logger.info(f"{'After':<{PAD}}: {data['len']}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()

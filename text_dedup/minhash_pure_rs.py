#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations
import subprocess
import multiprocessing as mp
import os
import re
import json
import click

from text_dedup import logger
from text_dedup.utils import (
    IOArgs,
    MetaArgs,
    MinHashArgs,
    Timer,
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
        # check if exists

            command = [
                binary_path,
                "--b", str(minhash_args.b),
                "--r", str(minhash_args.r),
                "--num-perm", str(minhash_args.num_perm),
                "--n-grams", str(minhash_args.ngram),
                "--parquet-path", parquet_path,
                "--main-col", meta_args.column,
                "--streaming",
                "--idx-col", meta_args.idx_column if meta_args.idx_column else "id",
                "--uf-output", os.path.join(io_args.output, "uf.json")
                ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(result.stderr)
            
            print(result.stdout)
            
            data_from_rust = result.stdout.split("\n")[-1]
            data = json.loads(data_from_rust)

    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"{'Before':<{PAD}}: {data['before']}")
    logger.info(f"{'After':<{PAD}}: {data['after']}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()

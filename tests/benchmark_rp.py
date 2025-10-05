import os

import click
import datasets

from text_dedup import logger
from text_dedup.minhash import main as minhash_main
from text_dedup.minhash_pure_rs import main as minhash_pure_rs_main
from text_dedup.minhash_rust import main as minhash_rust_main
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils import MinHashArgs
from text_dedup.utils import Timer

# os.cpu_count does not respect slurm job scheduler
NUM_PROC = min(os.cpu_count(), len(os.sched_getaffinity(0)))

DATASET = "togethercomputer/RedPajama-Data-1T-Sample"
PARQUET_PATH = "temp_files/temp_inp_parquet/data.parquet"
if __name__ == "__main__":
    t = Timer()

    if not os.environ.get("HF_DATASETS_CACHE"):
        logger.info("`HF_DATASETS_CACHE` not set, default location is `$HOME/.cache/huggingface/datasets`")

    table = []
    ds = datasets.load_dataset(
        DATASET,
        split="train",
        num_proc=NUM_PROC,
    ).map(
        lambda _, i: {"id": i},
        num_proc=NUM_PROC,
        with_indices=True,
    )
    logger.info("Saving dataset to disk")
    ds.save_to_disk("temp_files/temp_inp_ds")

    logger.info("Saving dataset to parquet")
    os.makedirs(os.path.dirname(PARQUET_PATH), exist_ok=True)
    (ds.to_pandas().to_parquet(PARQUET_PATH))
    logger.warning("This benchmark has no validation, and is purely for memory and speed benchmarking.")

    io_args = IOArgs(
        path="./temp_files/temp_inp_ds",
        local=True,
        num_proc=NUM_PROC,
        cache_dir=".cache",
        output="./temp_files/temp_output_minhash",
        debug=True,
        clean_cache=True,
    )
    meta_args = MetaArgs(column="text", batch_size=100000)
    minhash_args = MinHashArgs(num_perm=200, ngram=2, threshold=0.5, b=50, r=4)
    with t("MinHash Pure RS"):
        ctx = click.Context(minhash_pure_rs_main)
        io_args.output = minhash_output_rs = "./temp_files/temp_output_minhash_rs"
        ctx.invoke(
            minhash_pure_rs_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
            parquet_path=PARQUET_PATH,
        )

    with t("MinRust"):
        ctx = click.Context(minhash_rust_main)
        io_args.output = minhash_output_rust = "./temp_files/temp_output_minhash_rust"
        ctx.invoke(
            minhash_rust_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
        )

    with t("MinHash"):
        ctx = click.Context(minhash_main)
        io_args.output = minhash_output = "./temp_files/temp_output_minhash"
        ctx.invoke(
            minhash_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
        )

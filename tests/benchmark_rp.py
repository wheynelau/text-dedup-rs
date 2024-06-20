import os

import click
import datasets
import tracemalloc
from text_dedup import logger
from text_dedup.minhash import main as minhash_main
from text_dedup.minhash_rust import main as minhash_rust_main
from text_dedup.minhash_pure_rs import main as minhash_pure_rs_main
from text_dedup.utils import (IOArgs, MetaArgs, MinHashArgs,
                              Timer)

import warnings

NUM_PROC = os.cpu_count()
DATASET = "togethercomputer/RedPajama-Data-1T-Sample"
PARQUET_PATH = "temp_files/temp_inp_paruqet/data.parquet"
if __name__ == "__main__":
    t = Timer()

    if not os.environ.get('HF_DATASETS_CACHE'):
        logger.info("`HF_DATASETS_CACHE` not set, default location is `$HOME/.cache/huggingface/datasets`")

    table = []
    ds = ( 
            datasets.load_dataset(
            DATASET,
            split="train",
            num_proc=NUM_PROC,
        )
        .map(
            lambda _,i: {
                "id": i
            },
            num_proc=NUM_PROC,
            with_indices=True,
            )
        )   
    logger.info("Saving dataset to disk")
    ds.save_to_disk("temp_files/temp_inp_ds")

    logger.info("Saving dataset to parquet")
    os.makedirs("temp_files/temp_inp_paruqet", exist_ok=True)
    (
        ds
        .to_pandas()
        .to_parquet(PARQUET_PATH)
    )
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
            parquet_path = PARQUET_PATH
        )

    tracemalloc.start()
    with t("MinRust"):
        ctx = click.Context(minhash_rust_main)
        io_args.output = minhash_output_rust = "./temp_files/temp_output_minhash_rust"
        ctx.invoke(
            minhash_rust_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
        )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Current memory usage: {current / 1024**2:.2f} MB")
    print(f"Peak memory usage: {peak / 1024**2:.2f} MB")

    tracemalloc.start()
    with t("MinHash"):
        ctx = click.Context(minhash_main)
        io_args.output = minhash_output = "./temp_files/temp_output_minhash"
        ctx.invoke(
            minhash_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
        )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Current memory usage: {current / 1024**2:.2f} MB")
    print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
    # try:
    #     uf2results(f"{minhash_output}/uf.pkl", "MinHash", t.elapsed_times.get("MinHash"))
    #     uf2results(
    #         f"{minhash_output_rust}/uf.json", "MinHashRust", t.elapsed_times.get("MinRust")
    #     )
    #     uf2results(
    #         f"{minhash_output_rs}/uf.json", "MinHashPureRS", t.elapsed_times.get("MinHash Pure RS")
    #     )
    # except FileNotFoundError:
    #     print(f"Unable to find uf.pkl in {minhash_output} or {minhash_output_rust}")

    # exact_title_results(ds=ds, name="Exact Title")

    # print(
    #     pd.DataFrame(
    #         table,
    #         columns=[
    #             "Algorithm",
    #             "Precision (Duplicates)",
    #             "Recall (Duplicates)",
    #             "Precision (Non Duplicates)",
    #             "Recall (Non Duplicates)",
    #             "Macro F1 score",
    #             "Accuracy",
    #             "Time",
    #         ],
    #     ).to_markdown(index=False)
    # )

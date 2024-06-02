import os
import pickle  # nosec
import subprocess  # nosec

import click
import datasets
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from text_dedup.minhash import main as minhash_main
from text_dedup.minhash_rust import main as minhash_rust_main
from text_dedup.utils import (IOArgs, MetaArgs, MinHashArgs, SimHashArgs,
                              UniSimArgs)
from text_dedup.utils.preprocess import news_copy_preprocessing
from text_dedup.utils.timer import Timer
from text_dedup.utils.union_find import UnionFind

from text_dedup.dedup_rs import UnionFind as UnionFindRS

NUM_PROC = os.cpu_count()
TMP_DIR = "temp_files"


def prepare_data(dataset, output_path_ds, output_path_spark):
    dataset = dataset.map(
        lambda x: {"text": news_copy_preprocessing(x["article"])},
        num_proc=NUM_PROC,
    )
    dataset.save_to_disk(output_path_ds)
    os.makedirs(output_path_spark, exist_ok=True)
    dataset.to_pandas().to_parquet(output_path_spark + "/data.parquet")

    return dataset["cluster"]


def uf2results(labels, path):
    try:
        with open(path, "rb") as f:
            uf = pickle.load(f)  # nosec
    except: # noqa
        uf = UnionFindRS.load(path)

    predictions = [uf.find(i) for i in range(len(labels))]
    return adjusted_rand_score(labels, predictions)


def spark_assignment_to_uf(path: str):
    df = pd.read_parquet(path)
    uf = UnionFind()
    for _, row in df.iterrows():
        uf.union(row["id"], row["component"])

    with open(f"{spark_output}/uf.pkl", "wb") as f:
        pickle.dump(uf, f)
    return uf


if __name__ == "__main__":
    t = Timer()

    output_path_ds = f"{TMP_DIR}/news_input_ds"
    output_path_spark = f"{TMP_DIR}/news_input_spark"

    test_data = datasets.load_dataset("chenghao/NEWS-COPY-eval", split="test")
    labels = prepare_data(test_data, output_path_ds, output_path_spark)

    io_args = IOArgs(
        path=output_path_ds,
        local=True,
        num_proc=NUM_PROC,
        cache_dir=".cache",
        output=f"{TMP_DIR}/news_output_minhash",
        debug=True,
        clean_cache=True,
    )
    meta_args = MetaArgs(column="text", batch_size=10000, idx_column="idx")

    # TODO: hyperparameter tuning
    with t("MinHash"):
        ctx = click.Context(minhash_main)
        minhash_args = MinHashArgs(num_perm=200, ngram=2, threshold=0.45, b=50, r=4)
        io_args.output = minhash_output = f"{TMP_DIR}/news_output_minhash"
        ctx.invoke(
            minhash_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
        )

    with t("MinRust"):
        ctx = click.Context(minhash_rust_main)
        minhash_args = MinHashArgs(num_perm=200, ngram=2, threshold=0.45, b=50, r=4)
        io_args.output = minhash_output_rust = f"{TMP_DIR}/temp_output_minhash_rust"
        ctx.invoke(
            minhash_rust_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
        )

    with t("MinHash Spark Rust"):
        spark_output = f"{TMP_DIR}/temp_output_spark"
        spark_args = f"""
        spark-submit --executor-memory 86g
            --driver-memory 4g
            --executor-cores 12
            --num-executors 12
            --packages graphframes:graphframes:0.8.2-spark3.2-s_2.12
            --conf spark.executor.extraJavaOptions=-Dlog4j.configuration=./log4j.properties
            --conf spark.driver.extraJavaOptions=-Dlog4j.configuration=./log4j.properties
            text_dedup/minhash_spark_rust.py
            --input {output_path_spark}
            --output {spark_output}
            --column text
            --index idx
            --threshold 0.45
            --min_length 0
            --num_perm 256
            --ngram 2
            --debug
        """.split(
            "\n"
        )
        subprocess.run(
            [
                part.strip()
                for line in spark_args
                for part in line.strip().split(" ")
                if part.strip()
            ],
        )  # nosec
        # spark_assignment_to_uf(f"{spark_output}-assignment/assignment.parquet")

    with t("MinHash Spark"):
        spark_output = f"{TMP_DIR}/temp_output_spark"
        spark_args = f"""
        spark-submit --executor-memory 86g
            --driver-memory 4g
            --executor-cores 12
            --num-executors 12
            --packages graphframes:graphframes:0.8.2-spark3.2-s_2.12
            --conf spark.executor.extraJavaOptions=-Dlog4j.configuration=./log4j.properties
            --conf spark.driver.extraJavaOptions=-Dlog4j.configuration=./log4j.properties
            text_dedup/minhash_spark.py
            --input {output_path_spark}
            --output {spark_output}
            --column text
            --index idx
            --threshold 0.45
            --min_length 0
            --num_perm 256
            --ngram 2
            --debug
        """.split(
            "\n"
        )
        subprocess.run(
            [
                part.strip()
                for line in spark_args
                for part in line.strip().split(" ")
                if part.strip()
            ],
        )  # nosec
        # spark_assignment_to_uf(f"{spark_output}-assignment/assignment.parquet")

    # # TODO: hyperparameter tuning
    # with t("SimHash"):
    #     ctx = click.Context(simhash_main)
    #     simhash_args = SimHashArgs(bit_diff=12, num_bucket=13, ngram=5)
    #     io_args.output = simhash_output = "./temp_output_simhash"
    #     ctx.invoke(
    #         simhash_main,
    #         io_args=io_args,
    #         meta_args=meta_args,
    #         simhash_args=simhash_args,
    #     )

    # with t("UniSim"):
    #     ctx = click.Context(unisim_main)
    #     unisim_args = UniSimArgs(
    #         store_data=False,
    #         index_type="approx",
    #         similarity_threshold=0.89,
    #     )
    #     io_args.output = unisim_output = "./temp_output_unisim"
    #     meta_args.batch_size = 24
    #     ctx.invoke(
    #         unisim_main,
    #         io_args=io_args,
    #         meta_args=meta_args,
    #         unisim_args=unisim_args,
    #     )

    print(f"MinHash ARI: {uf2results(labels, f'{minhash_output}/uf.pkl')}")
    print(f"MinRust ARI: {uf2results(labels, f'{minhash_output_rust}/uf.json')}")
    print(f"MinHash (Spark) ARI: {uf2results(labels, f'{spark_output}/uf.pkl')}")

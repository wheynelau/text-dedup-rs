#!/usr/bin/env python
# @Date    : 2023-08-12 22:18:30
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

import argparse
import math
import re
import sys
import time
import warnings
from itertools import tee
from logging import Logger
from typing import List, Set, Tuple
from text_dedup.minhash_spark import ngrams_length_check, optimal_param
from text_dedup.dedup_rs import EmbedFunc, pyspark_hash, pyspark_edges

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import numpy as np
    import numpy.typing as npt
    import pyspark
    import xxhash
    from graphframes import GraphFrame  # type: ignore
    from pyspark import SparkConf
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.functions import udf
    from pyspark.sql.types import BooleanType
    from scipy.integrate import quad as integrate

SEED = 42
RNG = np.random.RandomState(SEED)
NON_ALPHA = re.compile(r"\W", re.UNICODE)
DTYPE = np.uint32
MAX_HASH = 4_294_967_295  # maximum 32-bit unsigned integer
MOD_PRIME = 4_294_967_291  # maximum 32-bit prime number

# region: IO
def partitioned_save(df: DataFrame, chunk_size: int, max_partitions: int, output: str):
    """
    Save a Spark DataFrame to a GCS directory in batches of `chunk_size` rows. PySpark natively does not support this
    functionality, so this workaround is necessary.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The Spark DataFrame to save.
    chunk_size : int
        The number of rows per batch.
    max_partitions : int
        The maximum number of partitions.
    output : str
        The GCS output directory.

    Raises
    ------
    RuntimeError
        If the save fails.
    """

    total_rows = df.count()
    partitions = max(256, min(math.ceil(total_rows / chunk_size), max_partitions))

    (
        df.repartition(partitions)
        .withColumn("__pid__", F.spark_partition_id())
        .write.partitionBy("__pid__")
        .parquet(output, mode="overwrite", compression="snappy")
    )


# endregion


if __name__ == "__main__":  # pragma: no cover
    # region: Argument Parsing
    parser = argparse.ArgumentParser(
        description="Intra-dataset near-deduplicating with PySpark"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input directory of parquet files",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Similarity threshold"
    )
    parser.add_argument("--ngram_size", type=int, default=5, help="N-gram size")
    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="Minimum token length of document to be considered. Short ones will be removed",
    )
    parser.add_argument(
        "--num_perm", type=int, default=250, help="Number of permutations"
    )
    parser.add_argument("--b", type=int, default=None, help="Number of bands")
    parser.add_argument("--r", type=int, default=None, help="Number of rows per band")
    parser.add_argument(
        "--column", "-c", type=str, default="content", help="Column to deduplicate on"
    )
    parser.add_argument("--index", type=str, default=None, help="Column to index on")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="GCS output directory of parquet files",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode by saving cluster results",
    )
    args = parser.parse_args()
    # endregion
    
    # region: Spark Configuration
    conf = (
        SparkConf()
        .set("spark.app.name", "MinHashLSH")
        .set("spark.sql.execution.arrow.pyspark.enabled", "true")
        .set("spark.storage.memoryFraction", "1")
        .set("spark.default.parallelism", "100")
        .set("spark.sql.autoBroadcastJoinThreshold", "20485760")
        .set("spark.sql.broadcastTimeout", "3600")
        .set("spark.sql.shuffle.partitions", "8192")
    )
    spark = SparkSession.Builder().config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    sc.setCheckpointDir(args.checkpoint_dir)
    log: Logger = spark.sparkContext._jvm.org.apache.log4j.LogManager.getLogger(__name__)  # type: ignore
    # endregion

    # region: Global Variables
    FINAL_SIZE: int = 0
    MAX_WRITE_CHUNK_SIZE: int = 200_000
    MAX_WRITE_PARTITIONS: int = 2048
    # endregion

    start_time: float = time.time()
    index_column = args.index or "__id__"

    B, R = args.b, args.r
    if B is None or R is None:
        B, R = optimal_param(args.threshold, args.num_perm)

    HASH_RANGES: List[Tuple[int, int]] = [(i * R, (i + 1) * R) for i in range(B)]
    PERMUTATIONS: Tuple[npt.NDArray[DTYPE], npt.NDArray[DTYPE]] = (
        RNG.randint(1, MOD_PRIME, size=(args.num_perm,), dtype=DTYPE),
        RNG.randint(0, MOD_PRIME, size=(args.num_perm,), dtype=DTYPE),
    )

    a,b = PERMUTATIONS

    a = a.tolist()
    b = b.tolist()
    # region: Data Loading
    # persist justification: this data will be needed when removing duplicates
    df: DataFrame = (
        spark.read.option("mergeSchema", "true")
        .parquet(args.input)
        .filter(
            udf(ngrams_length_check, BooleanType())(
                F.col(args.column), F.lit(args.ngram_size), F.lit(args.min_length)
            )
        )
        .withColumn("__id__", F.monotonically_increasing_id())
        .persist(pyspark.StorageLevel.DISK_ONLY)
    )
    # persist trigger
    DATA_SIZE: int = df.count()
    log.debug("-" * 120)
    log.debug(f"Loaded documents: {DATA_SIZE}")
    log.debug(f"{args.input=}")
    log.debug(f"{args.output=}")
    log.debug(f"{args.threshold=}")
    log.debug(f"{args.ngram_size=}")
    log.debug(f"{args.min_length=}")
    log.debug(f"{args.num_perm=}")
    log.debug(f"{args.column=}")
    for col, dtype in df.dtypes:
        log.debug(f"{col:<64}: {dtype}")
    log.debug("-" * 120)

    if DATA_SIZE == 0:
        log.debug("No data found.")
        exit(0)
    # endregion

    # region: MinHash
    edges: pyspark.RDD = (
        df.select(index_column, args.column)
        .rdd.flatMap(
            lambda x: pyspark_hash(
                x[1],  # args.column
                x[0],  # __id__
                a,
                b,
                HASH_RANGES
            )
        )  # (band_idx, band hash value, idx)
        .groupBy(
            lambda x: (x[0], x[1])
        )  # group by (band_idx, band hash value), potential bottleneck
        .flatMap(lambda x: pyspark_edges([ele[2] for ele in x[1]]))
        .distinct()
    ).persist(pyspark.StorageLevel.DISK_ONLY)
    # endregion

    # region: Connected Components

    if edges.isEmpty():
        partitioned_save(df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output)
        df.unpersist()
        edges.unpersist()

        log.debug("-" * 120)
        log.debug("No duplicates found.")
        log.debug(f"Data Output:    {args.output}")
        log.debug(f"Time:           {time.time() - start_time:.2f}s")
        log.debug("-" * 120)

        sys.exit(0)

    log.debug(f"MinHash Time:           {time.time() - start_time:.2f}s")
    log.debug("-" * 120)

    sys.exit(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edges_df: DataFrame = (
            spark.createDataFrame(edges, schema=["src", "dst"])
            .repartition(4096)
            .persist(pyspark.StorageLevel.DISK_ONLY)
        )
        log.debug(f"Edges DataFrame: {edges_df.count()}")
        vertices_df: DataFrame = (
            edges_df.select(F.col("src").alias("id"))
            .union(edges_df.select(F.col("dst").alias("id")))
            .distinct()
            .repartition(4096)
            .persist(pyspark.StorageLevel.DISK_ONLY)
        )
        log.debug(f"Vertices DataFrame: {vertices_df.count()}")
        assignment: DataFrame = (
            GraphFrame(vertices_df, edges_df)
            .connectedComponents()
            .persist(pyspark.StorageLevel.DISK_ONLY)
        )
        log.debug(f"Assignment DataFrame: {assignment.count()}")
        edges_df.unpersist()
        vertices_df.unpersist()
    # endregion

    if args.debug:
        # save assignment for debugging purposes
        assignment.write.parquet(
            f"{args.output}-assignment/assignment.parquet", mode="overwrite"
        )

    # region: Merge Results
    # justification: this is needed for final output
    df = df.join(
        assignment.select(
            F.col("id").alias(index_column), F.col("component").alias("__component__")
        ),
        on=index_column,
        how="left",
    ).persist(pyspark.StorageLevel.DISK_ONLY)
    assignment.unpersist()
    log.debug(f"Merging records: {df.count()}")
    # endregion

    df = (
        df.filter(
            F.col("__component__").isNull()
            | (F.col("__component__") == F.col(index_column))
        )
        .drop("__component__")
        .persist(pyspark.StorageLevel.DISK_ONLY)
    )
    FINAL_SIZE = df.count()

    # region: Output
    partitioned_save(df, MAX_WRITE_CHUNK_SIZE, MAX_WRITE_PARTITIONS, args.output)
    df.unpersist()

    # endregion

    log.debug("-" * 120)
    log.debug(f"Number of rows before:    {DATA_SIZE}")
    log.debug(f"Number of rows after:     {FINAL_SIZE}")
    log.debug(f"Percentage of rows kept:  {FINAL_SIZE / max(0, DATA_SIZE) * 100:.2f}%")
    log.debug(f"Output:                   {args.output}")
    log.debug(f"Time:                     {time.time() - start_time:.2f}s")
    log.debug("-" * 120)

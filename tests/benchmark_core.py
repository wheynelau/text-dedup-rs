import json
import os
import pickle  # nosec
import subprocess  # nosec
import sys
from collections import defaultdict

import click
import datasets
import pandas as pd
from datasets import Features
from datasets import Sequence
from datasets import Value

from text_dedup.dedup_rs import UnionFind as UnionFindRS
from text_dedup.minhash import main as minhash_main
from text_dedup.minhash_pure_rs import main as minhash_pure_rs_main
from text_dedup.minhash_rust import main as minhash_rust_main
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils import MinHashArgs
from text_dedup.utils import SimHashArgs
from text_dedup.utils import Timer
from text_dedup.utils import UnionFind
from text_dedup.utils import UniSimArgs

NUM_PROC = os.cpu_count()


def _recall(row):
    labelled_dups = set(row["duplicates"])
    LEN_LABELLED_DUPLICATES = len(labelled_dups)
    if LEN_LABELLED_DUPLICATES == 0:
        return 1
    dups = set(row["predictions"])
    return len(dups & labelled_dups) / LEN_LABELLED_DUPLICATES


def _precision(row):
    labelled_dups = set(row["duplicates"])
    dups = set(row["predictions"])
    LEN_DUPLICATES = len(dups)
    if LEN_DUPLICATES == 0:
        return 0
    return len(dups & labelled_dups) / LEN_DUPLICATES


def uf2results(path: str, name: str, time: float):
    id2cluster = defaultdict(set)
    try:
        with open(path, "rb") as f:
            uf = pickle.load(f)  # nosec
    except:  # noqa
        uf = UnionFindRS.load(path)

    for idx, cluster in uf.parent.items():
        id2cluster[cluster].add(idx)

    predictions = {
        id2core_id[x["id"]]: {id2core_id[neighbor] for neighbor in id2cluster[uf.find(x["id"])] if neighbor != x["id"]}
        for x in truth
    }
    df = (
        pd.Series(labels)
        .to_frame("duplicates")
        .reset_index()
        .merge(pd.Series(predictions).to_frame("predictions").reset_index(), on="index")
    )
    df["Correct"] = df.apply(lambda row: set(row["duplicates"]) == set(row["predictions"]), axis=1).astype(int)
    prediction_summary = {
        "Correct": df["Correct"].sum(),
        "Incorrect": df.shape[0] - df["Correct"].sum(),
    }
    prediction_summary["Accuracy"] = round(prediction_summary["Correct"] / df.shape[0], 4)
    recalls = df.apply(_recall, axis=1)
    prediction_summary["Recall"] = round(recalls.mean(), 4)
    precisions = df.apply(_precision, axis=1)
    prediction_summary["Precision"] = round(precisions.mean(), 4)

    df["Class"] = df.apply(classify_in_paper, axis=1)
    df["Class_"] = df.apply(lambda row: inverse(row["Class"]), axis=1)

    f1s = {}
    precisions = {}
    recalls = {}
    for col in ["Class", "Class_"]:
        label_counts = df[col].value_counts()
        precision = label_counts["TP"] / (label_counts["TP"] + label_counts["FP"])
        recall = label_counts["TP"] / (label_counts["TP"] + label_counts["FN"])
        f1 = 2 * precision * recall / (precision + recall)

        precisions[col] = precision
        recalls[col] = recall
        f1s[col] = f1

    table.append(
        [
            name,
            f"{precisions['Class']:.4f}",
            f"{recalls['Class']:.4f}",
            f"{precisions['Class_']:.4f}",
            f"{recalls['Class_']:.4f}",
            f"{(f1s['Class'] + f1s['Class_']) / 2:.4f}",
            f"{df['Correct'].mean():.4f}",
            f"{time:.2f}s",
        ]
    )


def classify_in_paper(record):
    duplicates = set(record["duplicates"])
    predictions = set(record["predictions"])

    LEN_PREDICTIONS = len(predictions)
    LEN_DUPLICATES = len(duplicates)

    # if len(predictions) == 0 it is Negative whether True or not.
    # Hopefully True is more common and short circuit ifs
    if LEN_PREDICTIONS == 0:
        if LEN_DUPLICATES == 0:
            return "TN"
        if LEN_DUPLICATES > 0:
            return "FN"

    # If len(predictions) > 0 it is Positive whether True or not.
    # Hopefully True is more common and short circuit ifs
    # python uses short circuiting so this is more readable and faster
    if LEN_PREDICTIONS > 0:
        if LEN_DUPLICATES > 0 and duplicates.issubset(predictions):
            return "TP"
        if LEN_DUPLICATES == 0 or not duplicates.issubset(predictions):
            return "FP"

    raise ValueError(f"This should not happen {duplicates} {predictions} {len(duplicates)=} {len(predictions)=}")


def inverse(label: str) -> str:
    # inverts the results basically N->P and P->N
    return {"TP": "TN", "FN": "FP", "FP": "FN", "TN": "TP"}[label]


# Commented out for testing purposes
# def spark_assignment_to_uf(path: str):
#     df = pd.read_parquet(path)
#     uf = UnionFind()
#     for _, row in df.iterrows():
#         uf.union(row["id"], row["component"])

#     with open(f"{spark_output}/uf.pkl", "wb") as f:
#         pickle.dump(uf, f)
#     return uf


def exact_title_results(ds, name: str):
    title2core_ids = defaultdict(set)
    for record in ds:
        title = record["processed_title"]
        core_id = int(record["core_id"])
        title2core_ids[title].add(core_id)

    matches = ds.map(
        lambda row: {"matches": {x for x in title2core_ids[row["processed_title"]] if x != int(row["core_id"])}}
    )
    matches = {int(x["core_id"]): x["matches"] for x in matches}
    ddf = (
        pd.Series(matches)
        .to_frame("predictions")
        .reset_index()
        .merge(pd.Series(labels).to_frame("duplicates").reset_index(), on="index")
    )
    ddf["Correct"] = ddf.apply(lambda row: set(row["duplicates"]) == set(row["predictions"]), axis=1).astype(int)
    ddf["Class"] = ddf.apply(lambda row: classify_in_paper(row), axis=1)
    ddf["Class_"] = ddf.apply(lambda row: inverse(row["Class"]), axis=1)

    f1s = {}
    precisions = {}
    recalls = {}
    for col in ["Class", "Class_"]:
        label_counts = ddf[col].value_counts().to_dict()
        precision = label_counts["TP"] / (label_counts["TP"] + label_counts["FP"])
        recall = label_counts["TP"] / (label_counts["TP"] + label_counts["FN"])
        f1 = 2 * precision * recall / (precision + recall)
        precisions[col] = precision
        recalls[col] = recall
        f1s[col] = f1

    table.append(
        [
            name,
            f"{precisions['Class']:.4f}",
            f"{recalls['Class']:.4f}",
            f"{precisions['Class_']:.4f}",
            f"{recalls['Class_']:.4f}",
            f"{(precisions['Class'] + precisions['Class_']) / 2:.4f}",
            f"{ddf['Correct'].mean():.4f}",
            "-",
        ]
    )


if __name__ == "__main__":
    t = Timer()

    table = []

    (
        datasets.load_dataset(
            "pinecone/core-2020-05-10-deduplication",
            split="train",
            cache_dir="./cache",
            num_proc=NUM_PROC,
        )
        .map(
            lambda x: {"text": " ".join((x["processed_title"], x["processed_abstract"])).lower()},
            num_proc=NUM_PROC,
        )
        .save_to_disk("temp_files/temp_inp_ds")
    )

    os.makedirs("temp_files/temp_inp_paruqet", exist_ok=True)
    (
        datasets.load_dataset(
            "pinecone/core-2020-05-10-deduplication",
            split="train",
            cache_dir="./cache",
            num_proc=NUM_PROC,
        )
        .map(
            lambda x, i: {
                "text": " ".join((x["processed_title"], x["processed_abstract"])).lower(),
                "id": i,
            },
            num_proc=NUM_PROC,
            with_indices=True,
        )
        .to_pandas()
        .to_parquet("temp_files/temp_inp_paruqet/data.parquet")
    )

    ds = datasets.load_from_disk("temp_files/temp_inp_ds")
    truth = ds.map(
        lambda x, idx: {
            "core_id": x["core_id"],
            "id": idx,
            "duplicates": x["labelled_duplicates"],
        },
        remove_columns=ds.column_names,
        with_indices=True,
        num_proc=NUM_PROC,
        features=Features(
            {
                "core_id": Value("string"),
                "id": Value("int64"),
                "duplicates": Sequence(Value("string")),
            }
        ),
    )
    id2core_id = {x["id"]: int(x["core_id"]) for x in truth}
    labels = {int(x["core_id"]): set(map(int, x["duplicates"])) if x["duplicates"] else set() for x in truth}

    io_args = IOArgs(
        path="./temp_files/temp_inp_ds",
        local=True,
        num_proc=NUM_PROC,
        cache_dir=".cache",
        output="./temp_files/temp_output_minhash",
        debug=True,
        clean_cache=True,
    )
    meta_args = MetaArgs(column="text", batch_size=10000)

    with t("MinHash"):
        ctx = click.Context(minhash_main)
        minhash_args = MinHashArgs(num_perm=200, ngram=2, threshold=0.5, b=50, r=4)
        io_args.output = minhash_output = "./temp_files/temp_output_minhash"
        ctx.invoke(
            minhash_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
        )
    with t("MinRust"):
        ctx = click.Context(minhash_rust_main)
        minhash_args = MinHashArgs(num_perm=200, ngram=2, threshold=0.5, b=50, r=4)
        io_args.output = minhash_output_rust = "./temp_files/temp_output_minhash_rust"
        ctx.invoke(
            minhash_rust_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
        )

    with t("MinHash Pure RS"):
        ctx = click.Context(minhash_pure_rs_main)
        io_args.output = minhash_output_rs = "./temp_files/temp_output_minhash_rs"
        minhash_args = MinHashArgs(num_perm=200, ngram=2, threshold=0.5, b=50, r=4)
        ctx.invoke(
            minhash_pure_rs_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
            parquet_path="temp_files/temp_inp_paruqet/data.parquet",
        )

    try:
        uf2results(f"{minhash_output}/uf.pkl", "MinHash", t.elapsed_times.get("MinHash"))
        uf2results(f"{minhash_output_rust}/uf.json", "MinHashRust", t.elapsed_times.get("MinRust"))
        uf2results(f"{minhash_output_rs}/uf.json", "MinHashPureRS", t.elapsed_times.get("MinHash Pure RS"))
    except FileNotFoundError:
        print(f"Unable to find uf.pkl in {minhash_output} or {minhash_output_rust}")

    exact_title_results(ds=ds, name="Exact Title")

    print(
        pd.DataFrame(
            table,
            columns=[
                "Algorithm",
                "Precision (Duplicates)",
                "Recall (Duplicates)",
                "Precision (Non Duplicates)",
                "Recall (Non Duplicates)",
                "Macro F1 score",
                "Accuracy",
                "Time",
            ],
        ).to_markdown(index=False)
    )

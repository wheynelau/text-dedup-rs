import os
import pickle  # nosec
import json
import subprocess  # nosec
import sys
from collections import defaultdict
from scipy.optimize import brute

import click
import datasets
import pandas as pd
from datasets import Features, Sequence, Value

from text_dedup.minhash_pure_rs import main as minhash_pure_rs_main
from text_dedup.dedup_rs import UnionFind as UnionFindRS
from text_dedup.utils import (IOArgs, MetaArgs, MinHashArgs, SimHashArgs,
                              Timer, UnionFind, UniSimArgs)

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

    raise ValueError(
        f"This should not happen {duplicates} {predictions} {len(duplicates)=} {len(predictions)=}"
    )


def inverse(label: str) -> str:
    # inverts the results basically N->P and P->N
    return {"TP": "TN", "FN": "FP", "FP": "FN", "TN": "TP"}[label]



def uf2results(path: str):

    id2cluster = defaultdict(set)
    try:
        with open(path, "rb") as f:
            uf = pickle.load(f)  # nosec
    except: # noqa
        uf = UnionFindRS.load(path)
    
    for idx, cluster in uf.parent.items():
        id2cluster[cluster].add(idx)

    predictions = {
        id2core_id[x["id"]]: {
            id2core_id[neighbor]
            for neighbor in id2cluster[uf.find(x["id"])]
            if neighbor != x["id"]
        }
        for x in truth
    }
    df = (
        pd.Series(labels)
        .to_frame("duplicates")
        .reset_index()
        .merge(pd.Series(predictions).to_frame("predictions").reset_index(), on="index")
    )
    df["Correct"] = df.apply(
        lambda row: set(row["duplicates"]) == set(row["predictions"]), axis=1
    ).astype(int)
    prediction_summary = {
        "Correct": df["Correct"].sum(),
        "Incorrect": df.shape[0] - df["Correct"].sum(),
    }
    prediction_summary["Accuracy"] = round(
        prediction_summary["Correct"] / df.shape[0], 4
    )
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

    macro_f1 = (f1s['Class'] + f1s['Class_']) / 2

    return macro_f1

if __name__ == "__main__":

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
                "text": " ".join(
                    (x["processed_title"], x["processed_abstract"])
                ).lower(),
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
    labels = {
        int(x["core_id"]): set(map(int, x["duplicates"])) if x["duplicates"] else set()
        for x in truth
    }

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

    def objective_function(params):
        num_perm, bands, r = params
        minhash_args = MinHashArgs(num_perm=int(num_perm), ngram=2, b=int(bands), r=int(r))
        ctx = click.Context(minhash_pure_rs_main)
        io_args.output = "./temp_files/temp_output_minhash_rs"
        ctx.invoke(
            minhash_pure_rs_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
            parquet_path="temp_files/temp_inp_paruqet/data.parquet"
        )
        macrof1 = uf2results("temp_files/temp_output_minhash_rs/uf.json")
        return -macrof1  # Negative because we need to minimize

    # Define ranges for the parameters
    ranges = (slice(100, 201, 10), slice(10, 51, 5), slice(1, 5, 1))

    # Perform the brute force optimization
    result = brute(objective_function, ranges, finish=None)

    # Print the best parameters and the corresponding macro F1 score
    print("Best parameters:")
    print(f"num_perm: {int(result[0])}, bands: {int(result[1])}, rows: {int(result[2])}")
    print(f"Best Macro F1 score: {-objective_function(result)}")
#!/usr/bin/env python
# @Date    : 2022-12-26 15:42:09
# @Author  : Chenghao Mou (mouchenghao@gmail.com)

from text_dedup.utils.analysis import optimal_param
from text_dedup.utils.args import (
    BloomFilterArgs,
    ExactHashArgs,
    IOArgs,
    MetaArgs,
    MinHashArgs,
    SAArgs,
    SimHashArgs,
    UniSimArgs,
)
from text_dedup.utils.const import CLUSTER_COLUMN, INDEX_COLUMN
from text_dedup.utils.hashfunc import (
    md5,
    md5_digest,
    md5_hexdigest,
    sha1_hash,
    sha256,
    sha256_digest,
    sha256_hexdigest,
    xxh3_16hash,
    xxh3_32hash,
    xxh3_64,
    xxh3_64_digest,
    xxh3_128,
    xxh3_128_digest,
    xxh3_hash,
)
from text_dedup.utils.inspect import random_samples
from text_dedup.utils.load import load_hf_dataset
from text_dedup.utils.memory import DisableReferenceCount
from text_dedup.utils.preprocess import news_copy_preprocessing, normalize
from text_dedup.utils.timer import Timer
from text_dedup.utils.tokenization import ngrams
from text_dedup.utils.union_find import UnionFind

__all__ = [
    "IOArgs",
    "MetaArgs",
    "BloomFilterArgs",
    "ExactHashArgs",
    "MinHashArgs",
    "SimHashArgs",
    "UniSimArgs",
    "SAArgs",
    "Timer",
    "ngrams",
    "UnionFind",
    "sha1_hash",
    "xxh3_hash",
    "load_hf_dataset",
    "DisableReferenceCount",
    "random_samples",
    "normalize",
    "news_copy_preprocessing",
    "INDEX_COLUMN",
    "CLUSTER_COLUMN",
    "md5",
    "sha256",
    "sha1_hash",
    "xxh3_64",
    "xxh3_64_digest",
    "xxh3_128",
    "xxh3_128_digest",
    "xxh3_hash",
    "xxh3_16hash",
    "xxh3_32hash",
    "optimal_param",
    "md5_digest",
    "md5_hexdigest",
    "sha256_digest",
    "sha256_hexdigest",
]

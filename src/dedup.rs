use arrow::array::{Int64Array, RecordBatch, StringArray};
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::Path,
};

mod args;
mod embed;
mod union;

use args::Args;

const MODULE_PRIME: u64 = 2u64.pow(61) - 1;

type Bytes = Vec<u8>;
type HashTable = Vec<HashMap<Vec<u8>, HashSet<u32>>>;

fn get_chunk_size() -> usize {
    std::env::var("CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000)
}

fn process_batch(batch: &RecordBatch, main_col: &str, idx_col: &str) -> (StringArray, Int64Array) {
    let text_col = batch
        .column(batch.schema().index_of(main_col).unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let id_col = batch
        .column(batch.schema().index_of(idx_col).unwrap())
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    (text_col.clone(), id_col.clone())
}

fn batch_add(hashes: Vec<Bytes>, key: u32, hash_tables: &mut HashTable) {
    hashes.into_iter().enumerate().for_each(|(index, hash)| {
        if let Some(table) = hash_tables.get_mut(index) {
            let entry = table.entry(hash).or_insert_with(HashSet::new);
            entry.insert(key);
        }
    });
}

fn cluster(hash_tables: HashTable) -> union::UnionFind {
    let mut uf = union::UnionFind::new();
    // Disable rayon for testing

    hash_tables.iter().for_each(|table| {
        // Lock the table to read its contents

        // BTreeMap iteration is deterministic (sorted by key)
        // BTreeSet iteration is also deterministic (sorted)
        for cluster in table.values() {
            if cluster.len() <= 1 {
                continue;
            }
            let idx: u32 = *cluster.iter().min().expect("Cluster should not be empty");
            for &x in cluster {
                uf.union(x as usize, idx as usize);
            }
        }
    });

    uf
}

fn generate_hash_rangs(b: u32, r: u32) -> Vec<(u32, u32)> {
    (0..b).map(|i| (i * r, (i + 1) * r)).collect()
}

fn main() {
    let args = Args::parse();

    // Check if B is too high
    let b: u32 = {
        let max_b = args.num_perm / args.r;
        if args.b > max_b {
            println!(
                "Warning: Provided B value is too high. Adjusting B from {} to {}",
                args.b, max_b
            );
            max_b
        } else {
            args.b
        }
    };

    let hash_ranges: Vec<(u32, u32)> = generate_hash_rangs(b, args.r);

    let mut hash_tables: Vec<HashMap<Vec<u8>, HashSet<u32>>> = vec![HashMap::new(); b as usize];
    let permutations = embed::generate_permutations(MODULE_PRIME as usize, args.num_perm);

    // Setup conditions
    let start_time = std::time::Instant::now();
    let path = Path::new(&args.parquet_path);
    // check if file exists
    if !path.exists() {
        panic!("File does not exist");
    }
    let file = File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();

    let reader = builder
        .with_row_groups(vec![0])
        .with_batch_size(100000)
        .build()
        .unwrap();
    let mut signatures: Vec<String> = Vec::with_capacity(1000000);
    let mut indices: Vec<u32> = Vec::with_capacity(1000000);
    let mut total_len = 0;
    for result in reader {
        let batch = result.unwrap();
        total_len += batch.num_rows();
        let (sigs, idxs) = process_batch(&batch, &args.main_col, &args.idx_col);
        signatures.extend(sigs.iter().map(|x| x.unwrap().to_string()));
        indices.extend(idxs.iter().map(|x| x.unwrap() as u32));
    }

    let elapsed_time = start_time.elapsed();
    println!("Time to read parquet file: {:?}", elapsed_time);

    let start_time = std::time::Instant::now();

    let chunk_size = get_chunk_size();
    let text_idx: Vec<(Vec<Bytes>, u32)> = signatures
        .par_chunks(chunk_size)
        .zip(indices.par_chunks(chunk_size))
        .flat_map(|(sig_chunk, idx_chunk)| {
            sig_chunk
                .iter()
                .zip(idx_chunk.iter())
                .map(|(text, idx)| {
                    let (a, b) = (&permutations.0, &permutations.1);
                    let hs: Vec<Bytes> = embed::py_embed_func(
                        text,
                        &args.n_grams,
                        &(a, b),
                        &hash_ranges,
                        &args.min_len,
                    );
                    (hs, *idx)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    println!("Time to embed to HS: {:?}", start_time.elapsed());
    let start_time = std::time::Instant::now();
    text_idx.iter().for_each(|(sig, i)| {
        batch_add(sig.clone(), *i, &mut hash_tables);
    });

    println!("Time to hash: {:?}", start_time.elapsed());

    let start_time = std::time::Instant::now();

    let mut uf: union::UnionFind = cluster(hash_tables);
    let uf_path = Path::new(&args.uf_output);
    // create directory if it doesn't exist
    if let Some(parent) = uf_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create directories");
    } else {
        panic!("The provided path does not have a parent directory. This should not happen");
    }
    println!("Time to cluster: {:?}", start_time.elapsed());
    println!("Number of edges: {}", uf.edges);

    uf.dump(&args.uf_output).unwrap();

    let start_time = std::time::Instant::now();
    let cluster_column: Vec<u32> = {
        indices
            .iter_mut()
            .map(|x| uf.find(*x as usize) as u32)
            .collect()
    };

    // filter

    let chunk_size = get_chunk_size();
    let final_vec: Vec<(String, u32)> = cluster_column
        .par_chunks(chunk_size)
        .zip(indices.par_chunks(chunk_size))
        .zip(signatures.par_chunks(chunk_size))
        .flat_map(|((cluster_chunk, idx_chunk), sig_chunk)| {
            cluster_chunk
                .iter()
                .zip(idx_chunk.iter())
                .zip(sig_chunk.iter())
                .filter(|((cluster, idx), _text)| cluster == idx)
                .map(|((_, idx), text)| (text.clone(), *idx))
                .collect::<Vec<_>>()
        })
        .collect();

    println!("Time to filter: {:?}", start_time.elapsed());

    let data = json!({
        "before" : total_len,
        "after": final_vec.len(),
    });
    // save data
    print!("{}", &data.to_string());
    let file = File::create("rs_output.json").unwrap();
    serde_json::to_writer_pretty(&file, &data).unwrap();
}

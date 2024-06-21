use rayon::prelude::*;
use serde_json::json;
use utils::parquet_utils;
use std::{collections::{HashMap, HashSet}, fs::File, path::Path, sync::{Arc,Mutex}};
use clap::Parser;

mod utils {
    pub mod embed;
    pub mod unionfind;
    pub mod dedup_utils;
    pub mod parquet_utils;
}

use crate::utils::dedup_utils;
use crate::utils::unionfind::UnionFind;
use crate::utils::embed;
use crate::utils::parquet_utils::*;

const MODULE_PRIME: u64 = 2u64.pow(61) - 1;

   

fn main() {

    let args = dedup_utils::Args::parse();
    
    // Check if B is too high
    let b: i32 = {
        let max_b = args.num_perm / args.r;
        if args.b > max_b {
            println!("Warning: Provided B value is too high. Adjusting B from {} to {}", args.b, max_b);
            max_b
        } else {
            args.b
        }
    };
    
    let hash_ranges: Vec<(i32, i32)> = dedup_utils::generate_hash_rangs(b, args.r);

    let hash_tables = Arc::new((0..b).map(|_| Mutex::new(HashMap::<String, HashSet<i32>>::new())).collect::<Vec<_>>());

    let permutations = embed::generate_permutations(MODULE_PRIME as usize, args.num_perm);

    // Setup conditions
    let start_time = std::time::Instant::now();
    let reader = parquet_utils::get_reader(100000, &args.parquet_path);
    let mut signatures: Vec<String> = Vec::with_capacity(1000000);
    let mut indices: Vec<i32> = Vec::with_capacity(1000000);
    let mut total_len = 0;
    
    for result in reader {
        let batch = result.unwrap();
        total_len += batch.num_rows();
        let (sigs, idxs) = process_batch(&batch, &args.main_col, &args.idx_col);
        signatures.extend(sigs.iter().map(|x| x.unwrap().to_string()));
        indices.extend(idxs.iter().map(|x| {
            x.unwrap() as i32
    }));
    }

    let elapsed_time = start_time.elapsed();
    println!("Time to read parquet file: {:?}", elapsed_time);

    let start_time = std::time::Instant::now();

    let text_idx:Vec<(Vec<String>,i32)> = signatures.par_iter().zip(indices.par_iter())
        .map(|(text, idx)| {
            let hs: Vec<String> = embed::py_embed_func(&text, args.n_grams, permutations.clone(), hash_ranges.clone());
            (hs, *idx)
        }).collect();

    println!("Time to embed to HS: {:?}", start_time.elapsed());
    let start_time = std::time::Instant::now();
    text_idx.par_iter().for_each(|(sig, i)| {
        dedup_utils::batch_add(sig.clone(), *i, &hash_tables);
    });

    println!("Time to hash: {:?}", start_time.elapsed());

    let start_time = std::time::Instant::now();
    
    let uf: UnionFind = dedup_utils::cluster(hash_tables);

    let uf_path = Path::new(&args.uf_output);
    // create directory if it doesn't exist
    if let Some(parent) = uf_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create directories");
    } else {
        panic!("The provided path does not have a parent directory. This should not happen");
    }
    println!("Time to cluster: {:?}", start_time.elapsed());

    uf.dump(&args.uf_output).unwrap();


    let start_time = std::time::Instant::now();
    let cluster_column: Vec<i32> = {
        let uf = Mutex::new(uf);

        indices.par_iter_mut().map(|x| {
            // Lock the mutex and perform the find operation
            let mut uf = uf.lock().unwrap();
            uf.find(*x as usize) as i32
        }).collect()
    };

    // filter

    let final_vec: Vec<(String, i32)> = cluster_column.par_iter()
                                        .zip(indices.par_iter())
                                        .zip(signatures.par_iter())
                                        .filter(|((cluster, idx), _text)| cluster == idx)
                                        .map(|((_, idx), text)| (text.clone(), *idx))
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
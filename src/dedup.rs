
use ndarray::ArcArray1;
///
/// Contains the main functions for running the main

use rayon::prelude::*;
use serde_json::json;
use std::{collections::{HashMap, HashSet}, error::Error, path::Path, sync::{Arc,Mutex}};
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;

use crate::utils::dedup_utils;
use crate::utils::unionfind::UnionFind;
use crate::utils::embed;
use crate::utils::parquet_utils::*;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {


    #[arg(short, long, default_value = "50")]
    pub b:i32,

    #[arg(short, long, default_value="4")]
    pub r:i32,

    #[arg(short, long, default_value="200")]
    pub num_perm: i32,

    #[arg(short, long, default_value="2")]
    pub n_grams: i32,

    #[arg(short, long, default_value="text")]
    pub main_col: String,

    #[arg(short, long)]
    pub parquet_path: String,

    #[arg(short, long, default_value="id")]
    pub idx_col: String,

    #[arg(short, long, default_value="uf_output")]
    pub uf_output: String,

    #[arg(short,long, default_value="false")]
    pub streaming: bool
    
}

pub fn parse_args() -> Args {
    Args::parse()
}

pub fn default (args:Args,
                reader:ParquetRecordBatchReader,
                permutations:(ArcArray1<u64>, ArcArray1<u64>),
                hash_ranges:Vec<(i32, i32)> ) -> Result<(), Box<dyn Error>> {
    
    let hash_tables = Arc::new((0..args.b).map(|_| Mutex::new(HashMap::<String, HashSet<i32>>::new())).collect::<Vec<_>>());
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

    let text_idx:Vec<(Vec<String>,i32)> = signatures.par_iter().zip(indices.par_iter())
        .map(|(text, idx)| {
            let hs: Vec<String> = embed::py_embed_func(&text, args.n_grams, permutations.clone(), hash_ranges.clone());
            (hs, *idx)
        }).collect();

    text_idx.par_iter().for_each(|(sig, i)| {
        dedup_utils::batch_add(sig.clone(), *i, &hash_tables);
    });

    
    let uf = Arc::new(Mutex::new(UnionFind::new()));

    let start_time = std::time::Instant::now();
    dedup_utils::cluster(&hash_tables, uf.clone());
    let cluster_column: Vec<i32> = {

        indices.par_iter_mut().map(|x| {
            // Lock the mutex and perform the find operation
            let mut uf = uf.lock().unwrap();
            uf.find(*x as usize) as i32
        }).collect()
    };

    println!("Time to cluster: {:?}", start_time.elapsed());

    let uf: UnionFind = Arc::try_unwrap(uf).unwrap().into_inner().unwrap();

    let uf_path = Path::new(&args.uf_output);
    // create directory if it doesn't exist
    if let Some(parent) = uf_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create directories");
    } else {
        panic!("The provided path does not have a parent directory. This should not happen");
    }

    uf.dump(&args.uf_output).unwrap();
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
    
    Ok(())
} 

pub fn streaming (args:Args,
    reader:ParquetRecordBatchReader,
    permutations:(ArcArray1<u64>, ArcArray1<u64>),
    hash_ranges:Vec<(i32, i32)>) -> Result<(), Box<dyn Error>>{

    // For streaming, we don't load the full signatures into memory
    let mut total_len = 0;
    let hash_tables = Arc::new((0..args.b).map(|_| Mutex::new(HashMap::<String, HashSet<i32>>::new())).collect::<Vec<_>>());
    let uf = Arc::new(Mutex::new(UnionFind::new()));

    // indices don't really take up much space
    let mut out_indices: Vec<i32> = Vec::new();
    for result in reader {
        let batch = result.unwrap();
        total_len += batch.num_rows();
        let (sigs, idxs) = process_batch(&batch, &args.main_col, &args.idx_col);

        let signatures: Vec<String> = sigs.iter().map(|x| {
            x.unwrap().to_string()
        }).collect();
        
        let indices: Vec<i32> = idxs.iter().map(|x| {
            x.unwrap() as i32
        }).collect();

        out_indices.extend(indices.iter());
        let text_idx:Vec<(Vec<String>,i32)> = signatures.par_iter().zip(indices.par_iter())
        .map(|(text, idx)| {
            let hs: Vec<String> = embed::py_embed_func(&text, args.n_grams, permutations.clone(), hash_ranges.clone());
            (hs, *idx)
        }).collect();

        text_idx.par_iter().for_each(|(sig, i)| {
            dedup_utils::batch_add(sig.clone(), *i, &hash_tables);
        });

        dedup_utils::cluster(&hash_tables, uf.clone());

    }

    // get size of out_indices
    let out_indices_size = out_indices.capacity() as f64 * std::mem::size_of::<i32>() as f64 / 1024_f64.powi(2);
    println!("Size of out_indices: {} MB", out_indices_size);
    let _ = total_len;

    let cluster_column: Vec<i32> = {
        out_indices.par_iter_mut().map(|x| {
            // Lock the mutex and perform the find operation
            let mut uf = uf.lock().unwrap();
            uf.find(*x as usize) as i32
        }).collect()
    };

    let uf_path = Path::new(&args.uf_output);
    // create directory if it doesn't exist

    if let Some(parent) = uf_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create directories");
    } else {
        panic!("The provided path does not have a parent directory. This should not happen");
    }
    let uf: UnionFind = Arc::try_unwrap(uf).unwrap().into_inner().unwrap();

    uf.dump(&args.uf_output)?;

    let final_vec: Vec<i32> = cluster_column.iter()
                .zip(out_indices.iter())
                .filter(|(cluster, idx)| cluster == idx)
                .map(|(_, idx)| *idx)
            .collect();

    let data = json!({
        "before" : total_len,
        "after": final_vec.len(),
    });
    // save data
    print!("{}", &data.to_string());
    Ok(())
}
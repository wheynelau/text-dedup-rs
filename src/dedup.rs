use arrow::array::{Int64Array, RecordBatch, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde_json::json;
use std::{collections::{HashMap, HashSet}, fs::File, path::Path, sync::Mutex};
use clap::Parser;

mod embed;
mod union;
const MODULE_PRIME: u64 = 2u64.pow(61) - 1;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {


    #[arg(short, long, default_value = "50")]
    b:i32,

    #[arg(short, long, default_value="4")]
    r:i32,

    #[arg(short, long, default_value="200")]
    num_perm: i32,

    #[arg(short, long, default_value="text")]
    main_col: String,

    #[arg(short, long, default_value="id")]
    idx_col: String,

    #[arg(short, long, default_value="uf_output")]
    uf_output: String
    
}


fn process_batch(batch: &RecordBatch, main_col: &str, idx_col: &str) -> (StringArray, Int64Array) {
    let text_col = batch.column(batch.schema().index_of(main_col).unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let id_col = batch.column(batch.schema().index_of(idx_col).unwrap())
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    (text_col.clone(), id_col.clone())
}

fn batch_add(hashes: Vec<String>, key: i32, hash_tables: &mut Vec<HashMap<String, HashSet<i32>>>) {
    hashes.into_iter().enumerate().for_each(|(index, hash)| {
        if let Some(table) = hash_tables.get_mut(index) {
            let entry = table.entry(hash).or_insert_with(HashSet::new);
            entry.insert(key);
        }
    });
}

fn cluster(hash_tables: Vec<HashMap<String, HashSet<i32>>>)
    -> union::UnionFind {
    let mut uf = union::UnionFind::new();
    for table in &hash_tables {
        for cluster in table.values() {
            if cluster.len() <= 1 {
                continue;
            }
            let idx: i32 = *cluster.iter().min().expect("Cluster should not be empty");
            for &x in cluster {
                // self.edges.push((x, idx)); // Doesn't seem necessary
                uf.union(x as usize, idx as usize);
            }
        }
    }
    uf
}
   

fn main() {

    let args = Args::parse();
    let hash_ranges: Vec<(i32, i32)> = (0..args.b)
                        .map(|i| (i * args.r, (i + 1) * args.r))
                        .collect();

    let mut hash_tables: Vec<HashMap<String, HashSet<i32>>> = vec![HashMap::new(); args.b as usize];

    let permutations = embed::generate_permutations(MODULE_PRIME as usize, args.num_perm);

    // Setup conditions
    let path = "temp_files/temp_inp_paruqet/data.parquet";
    let file = File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let reader = builder.with_row_groups(vec![0])
                                            .build()
                                            .unwrap();
    let mut signatures: Vec<String> = Vec::with_capacity(1000000);
    let mut indices: Vec<i32> = Vec::with_capacity(1000000);
    for result in reader {
        let batch = result.unwrap();
        let (sigs, idxs) = process_batch(&batch, &args.main_col, &args.idx_col);
        signatures.extend(sigs.iter().map(|x| x.unwrap().to_string()));
        indices.extend(idxs.iter().map(|x| {
            x.unwrap() as i32
    }));
    }

    let text_idx:Vec<(Vec<String>,i32)> = signatures.par_iter().zip(indices.par_iter())
        .map(|(text, idx)| {
            let hs: Vec<String> = embed::py_embed_func(&text, permutations.clone(), hash_ranges.clone());
            (hs, *idx)
        }).collect();
    text_idx.iter().for_each(|(sig, i)| {
        batch_add(sig.clone(), *i, &mut hash_tables);
    });
    
    let uf: union::UnionFind = cluster(hash_tables);
    let uf_path = Path::new(&args.uf_output);
    // create directory if it doesn't exist
    if let Some(parent) = uf_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create directories");
    } else {
        panic!("The provided path does not have a parent directory. This should not happen");
    }
    uf.dump(&args.uf_output).unwrap();

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

    let data = json!({
        "len": final_vec.len(),
    });
    println!("{}", data.to_string());
}
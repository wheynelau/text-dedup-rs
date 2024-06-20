use arrow::{array::{Int64Array, RecordBatch, StringArray}};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde_json::json;
use std::{collections::{HashMap, HashSet}, fs::File, path::Path, sync::{Arc,Mutex}};
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

    #[arg(short, long, default_value="2")]
    n_grams: i32,

    #[arg(short, long, default_value="text")]
    main_col: String,

    #[arg(short, long)]
    parquet_path: String,

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

fn cluster(hash_tables: Vec<HashMap<String, HashSet<i32>>>) -> union::UnionFind {
    let uf = union::UnionFind::new();
    let uf = Arc::new(Mutex::new(uf));

    hash_tables.par_iter().for_each(|table| {
        let mut uf = uf.lock().unwrap(); // Lock the UnionFind for each operation
        for cluster in table.values() {
            if cluster.len() <= 1 {
                continue;
            }
            let idx: i32 = *cluster.iter().min().expect("Cluster should not be empty");
            for &x in cluster {
                uf.union(x as usize, idx as usize);
            }
        }
    });

    // Extract the UnionFind from Arc<Mutex<>>. This is safe because no other threads are using it now.
    Arc::try_unwrap(uf).ok().expect("Failed to unwrap Arc").into_inner().unwrap()
}

fn generate_hash_rangs(b: i32, r: i32) -> Vec<(i32, i32)> {
    (0..b)
        .map(|i| (i * r, (i + 1) * r))
        .collect()
}
   

fn main() {

    let args = Args::parse();
    
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
    
    let hash_ranges: Vec<(i32, i32)> = generate_hash_rangs(b, args.r);

    let mut hash_tables: Vec<HashMap<String, HashSet<i32>>> = vec![HashMap::new(); b as usize];

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
    let reader = builder.with_row_groups(vec![0])
                                            .with_batch_size(100000)
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

    let elapsed_time = start_time.elapsed();
    println!("Time to read parquet file: {:?}", elapsed_time);

    let start_time = std::time::Instant::now();

    let text_idx:Vec<(Vec<String>,i32)> = signatures.par_iter().zip(indices.par_iter())
        .map(|(text, idx)| {
            let hs: Vec<String> = embed::py_embed_func(&text, args.n_grams, permutations.clone(), hash_ranges.clone());
            (hs, *idx)
        }).collect();
    text_idx.iter().for_each(|(sig, i)| {
        batch_add(sig.clone(), *i, &mut hash_tables);
    });

    println!("Time to hash: {:?}", start_time.elapsed());

    let start_time = std::time::Instant::now();
    
    let uf: union::UnionFind = cluster(hash_tables);
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
        "len": final_vec.len(),
    });
    // save data
    let file = File::create("rs_output.json").unwrap();
    serde_json::to_writer_pretty(&file, &data).unwrap();
    print!("{}", &data.to_string());
}
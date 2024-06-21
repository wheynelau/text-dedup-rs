///
/// This utils library is specific to the dedup.rs file, where it
/// is mainly for the rust bindings
/// 
use std::{collections::{HashMap, HashSet}, sync::{Arc, Mutex}};
use rayon::prelude::*;
use clap::Parser;

use crate::utils::unionfind::UnionFind;

pub fn batch_add(hashes: Vec<String>, key: i32, hash_tables: &Arc<Vec<Mutex<HashMap<String, HashSet<i32>>>>>) {
    hashes.into_iter().enumerate().for_each(|(index, hash)| {
        if let Some(table_lock) = hash_tables.get(index) {
            let mut table = table_lock.lock().unwrap();
            let entry = table.entry(hash).or_insert_with(HashSet::new);
            entry.insert(key);
        }
    });
}

pub fn cluster(hash_tables: Arc<Vec<Mutex<HashMap<String, HashSet<i32>>>>>) -> UnionFind {
    let uf = UnionFind::new();
    let uf = Arc::new(Mutex::new(uf));

    hash_tables.par_iter().for_each(|table_mutex| {
        let table = table_mutex.lock().unwrap(); // Lock the table to read its contents
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
/// Generates a vector of tuples representing hash ranges based on the provided parameters.
///
///
/// # Arguments
/// * `b` - The number of buckets into which the hash ranges are divided.
/// * `r` - The number of rows per bucket.
/// * `num_perm` - The total number of permutations available.
///
/// # Returns
/// A vector of tuples, where each tuple `(start, end)` represents the range of indices for a bucket.
/// The `start` is inclusive, and `end` is exclusive.
///
/// # Examples
/// ```
/// hash_ranges = generate_hash_ranges(10, 5)
/// ```
pub fn generate_hash_rangs(b: i32, r: i32) -> Vec<(i32, i32)> {
    (0..b)
        .map(|i| (i * r, (i + 1) * r))
        .collect()
}
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_generate_hash_ranges() {

        let b = 10;
        let r = 5;
        let expected = vec![
            (0, 5),
            (5, 10),
            (10, 15),
            (15, 20),
            (20, 25),
            (25, 30),
            (30, 35),
            (35, 40),
            (40, 45),
            (45, 50),
        ];

        let result = generate_hash_rangs(b, r);
        assert_eq!(result, expected);
    }
}
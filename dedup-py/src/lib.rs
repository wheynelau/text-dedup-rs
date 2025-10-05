use dedup_core::{embed as coreembed, utils as coreutils};
use numpy::PyReadonlyArrayDyn;
use pyo3::{prelude::*, types::PyType};
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

pub mod embed;
pub mod utils;

use dedup_core::union::UnionFind as CoreUnionFind;
use utils::UnionFind;

const MODULE_PRIME: u64 = (1u64 << 61) - 1;
type Bytes = Vec<u8>;

/// Input sequence types for efficient string handling
#[derive(Debug, Clone)]
pub enum InputSequence<'s> {
    /// Borrowed string slice
    Borrowed(Cow<'s, str>),
    /// Owned string
    Owned(String),
}

impl<'s> From<&'s str> for InputSequence<'s> {
    fn from(s: &'s str) -> Self {
        Self::Borrowed(Cow::Borrowed(s))
    }
}

impl From<String> for InputSequence<'_> {
    fn from(s: String) -> Self {
        Self::Owned(s)
    }
}

impl InputSequence<'_> {
    /// Get the string slice, borrowing when possible
    #[inline]
    #[allow(dead_code)]
    fn as_str(&self) -> &str {
        match self {
            Self::Borrowed(cow) => cow.as_ref(),
            Self::Owned(s) => s.as_str(),
        }
    }
}

fn get_chunk_size() -> usize {
    std::env::var("CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000)
}

#[pyclass]
struct EmbedFunc {
    hash_values: Vec<(u32, u32)>,
    #[allow(dead_code)]
    main_col: String,
    #[allow(dead_code)]
    idx_col: String,
    n_grams: u32,
    #[pyo3(get)]
    hash_tables: Vec<HashMap<Vec<u8>, HashSet<u32>>>,
    #[pyo3(get)]
    edges: Vec<(u32, u32)>,
    permutations: [Vec<u64>; 2],
    #[allow(dead_code)]
    dtype: Option<String>,
    min_len: Option<u32>,
}

#[derive(IntoPyObject, IntoPyObjectRef)]
pub enum Sig {
    Signature(Vec<Vec<u8>>),
    Index(u32),
}

// TODO: Need to tidy up
#[pymethods]
impl EmbedFunc {
    ///
    /// Create a new EmbedFunc object
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold for the similarity
    /// * `num_perm` - The number of permutations
    /// * `false_positive` - The false positive rate
    /// * `false_negative` - The false negative rate
    /// * `main_col` - The name of the main column
    /// * `idx_col` - The name of the index column
    ///
    #[allow(clippy::too_many_arguments)]
    #[new]
    fn new(
        threshold: f64,
        num_perm: u32,
        n_grams: u32,
        false_positive: f64,
        false_negative: f64,
        main_col: &str,
        idx_col: &str,
        dtype: Option<String>,
        min_len: Option<u32>,
    ) -> Self {
        let (b, r) = coreutils::optimal_param(threshold, num_perm, false_positive, false_negative);
        Self::shared_init(b, r, n_grams, num_perm, main_col, idx_col, dtype, min_len)
    }
    ///
    /// Create a new EmbedFunc object with the known B and R values
    ///
    /// # Arguments
    ///
    /// * `b`
    /// * `r`
    /// * `num_perm` - The number of permutations
    /// * `main_col` - The name of the main column
    /// * `idx_col` - The name of the index column
    ///
    #[allow(clippy::too_many_arguments)]
    #[classmethod]
    fn from_b_r(
        _cls: &Bound<'_, PyType>,
        b: u32,
        r: u32,
        n_grams: u32,
        num_perm: u32,
        main_col: &str,
        idx_col: &str,
        dtype: Option<String>,
        min_len: Option<u32>,
    ) -> Self {
        Self::shared_init(b, r, n_grams, num_perm, main_col, idx_col, dtype, min_len)
    }
    #[classmethod]
    fn from_permutations(
        _cls: &Bound<'_, PyType>,
        n_grams: u32,
        min_len: u32,
        hashranges: Vec<(u32, u32)>,
        permutations: (Vec<u64>, Vec<u64>),
    ) -> Self {
        let b = hashranges.len() as u32;
        let hash_tables: Vec<HashMap<Vec<u8>, HashSet<u32>>> = vec![HashMap::new(); b as usize];
        let edges: Vec<(u32, u32)> = Vec::new();
        let dtype = None;
        EmbedFunc {
            hash_values: hashranges,
            main_col: "__signatures__".to_string(),
            idx_col: "__index__".to_string(),
            n_grams,
            hash_tables,
            edges,
            permutations: [permutations.0, permutations.1],
            dtype,
            min_len: Some(min_len),
        }
    }
    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    fn shared_init(
        b: u32,
        r: u32,
        n_grams: u32,
        num_perm: u32,
        main_col: &str,
        idx_col: &str,
        dtype: Option<String>,
        min_len: Option<u32>,
    ) -> Self {
        let b = {
            let max_b = num_perm / r;
            if b > max_b {
                println!("Number of permutations: {}, r: {}", num_perm, r);
                println!(
                    "Warning: Provided B value is too high. Adjusting B from {} to {}",
                    b, max_b
                );
                max_b
            } else {
                b
            }
        };
        let hash_ranges: Vec<(u32, u32)> = (0..b).map(|i| (i * r, (i + 1) * r)).collect();

        let hash_tables: Vec<HashMap<Vec<u8>, HashSet<u32>>> = vec![HashMap::new(); b as usize];
        let edges: Vec<(u32, u32)> = Vec::new();
        let permutations = coreembed::generate_permutations(MODULE_PRIME as usize, num_perm);
        EmbedFunc {
            hash_values: hash_ranges,
            main_col: main_col.to_string(),
            idx_col: idx_col.to_string(),
            n_grams,
            hash_tables,
            edges,
            permutations: [permutations.0, permutations.1],
            dtype,
            min_len,
        }
    }
    ///
    /// Add the signature to the hash table
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the hash table
    /// * `hash` - The signature to be added
    /// * `i` - The index of the signature
    fn batch_add(&mut self, hashes: Vec<Bytes>, key: u32) {
        hashes.into_iter().enumerate().for_each(|(index, hash)| {
            if let Some(table) = self.hash_tables.get_mut(index) {
                let entry = table.entry(hash).or_insert_with(HashSet::new);
                entry.insert(key);
            }
        });
    }
    ///
    /// This function embeds the text and adds the signature to the hash table
    ///
    /// Optimized version using borrowed string slices and releasing the GIL for parallel processing
    ///
    fn batch_embed_shard<'py>(
        &mut self,
        py: Python<'py>,
        text: Vec<String>,
        idx: PyReadonlyArrayDyn<'py, u32>,
    ) {
        let min_len = self.min_len.unwrap_or(5); // Default to 5 if min_len is None
        let chunk_size = get_chunk_size();
        let idx_slice = idx.as_slice().unwrap();

        let n_grams = self.n_grams;
        let permutations_a = &self.permutations[0];
        let permutations_b = &self.permutations[1];

        // release GIL
        let text_idx: Vec<(Vec<Vec<u8>>, u32)> = py.detach(|| {
            text.par_chunks(chunk_size)
                .zip(idx_slice.par_chunks(chunk_size))
                .flat_map(|(text_chunk, idx_chunk)| {
                    // Pre-allocate the result vector for this chunk
                    let mut chunk_results = Vec::with_capacity(text_chunk.len());

                    for (s, &i) in text_chunk.iter().zip(idx_chunk.iter()) {
                        // Use as_str() to work with &str instead of &String
                        let mapped = coreembed::py_embed_func(
                            s.as_str(),
                            &n_grams,
                            &(permutations_a, permutations_b),
                            &self.hash_values,
                            &min_len,
                        );
                        chunk_results.push((mapped, i));
                    }
                    chunk_results
                })
                .collect()
        });

        // GIL is automatically reacquired here
        // Batch insert into hash tables
        for (sig, i) in text_idx {
            self.batch_add(sig, i);
        }
    }
    ///
    /// Cluster the hash tables
    ///
    /// # Returns
    ///
    /// A UnionFind data structure representing the clusters
    ///
    /// Iterates the hash tables and clusters the signatures
    fn cluster(&mut self) -> UnionFind {
        let mut uf = CoreUnionFind::new();
        for table in &self.hash_tables {
            for cluster in table.values() {
                if cluster.len() <= 1 {
                    continue;
                }
                let idx: u32 = *cluster.iter().min().expect("Cluster should not be empty");
                for &x in cluster {
                    // self.edges.push((x, idx)); // Doesn't seem necessary
                    uf.union(x as usize, idx as usize);
                }
            }
        }
        UnionFind { inner: uf }
    }
    ///
    /// Filter duplicates using UnionFind and return indices to keep
    ///
    /// # Arguments
    ///
    /// * `py` - Python GIL token
    /// * `uf` - The UnionFind data structure
    /// * `indices` - Array of document indices
    ///
    /// # Returns
    ///
    /// A vector of positions in the original dataset to keep (non-duplicates)
    ///
    fn filter_duplicates<'py>(
        &self,
        py: Python<'py>,
        mut uf: UnionFind,
        indices: PyReadonlyArrayDyn<'py, u32>,
    ) -> Vec<usize> {
        let indices_slice = indices.as_slice().unwrap();

        // Pre-compute all cluster assignments sequentially (find() is mutable)
        let clusters: Vec<u32> = indices_slice
            .iter()
            .map(|&idx| uf.inner.find(idx as usize) as u32)
            .collect();

        let indices_owned = indices_slice;
        let chunk_size = get_chunk_size();

        let result = py.detach(|| {
            clusters
                .par_chunks(chunk_size)
                .zip(indices_owned.par_chunks(chunk_size))
                .enumerate()
                .flat_map(|(chunk_idx, (cluster_chunk, idx_chunk))| {
                    let base_pos = chunk_idx * chunk_size;
                    cluster_chunk
                        .iter()
                        .zip(idx_chunk.iter())
                        .enumerate()
                        .filter(|(_, (&cluster, &idx))| cluster == idx)
                        .map(move |(i, _)| base_pos + i)
                        .collect::<Vec<_>>()
                })
                .collect()
        });
        result
    }
}

///
/// def embed_func(
// content: str,
// idx: int,
// *,
// num_perm: int,
// ngram_size: int,
// min_length: int,
// hashranges: list[tuple[int, int]],
// permutations: np.ndarray,
// hash_func: Callable,
// dtype: type,
// max_hash: np.uint,
// modulo_prime: np.uint,
// ) -> dict[str, Any]:
#[pyfunction]
fn embed_func(
    content: String,
    ngram_size: u32,
    permutations: (Vec<u64>, Vec<u64>),
    hashranges: Vec<(u32, u32)>,
    min_length: u32,
    idx: Option<u32>,
) -> HashMap<String, Sig> {
    let hash_ranges: Vec<(u32, u32)> = hashranges;
    let (a, b) = (&permutations.0, &permutations.1);
    let hashes =
        coreembed::py_embed_func(&content, &ngram_size, &(a, b), &hash_ranges, &min_length);
    let mut result = HashMap::new();
    result.insert("__signatures__".to_string(), Sig::Signature(hashes));
    let idx = idx.unwrap_or(0);
    result.insert("__index__".to_string(), Sig::Index(idx));
    result
}

#[pymodule]
fn dedup_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EmbedFunc>()?;
    m.add_class::<UnionFind>()?;
    m.add_function(wrap_pyfunction!(embed_func, m)?)?;
    Ok(())
}

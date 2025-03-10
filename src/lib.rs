use ndarray::ArcArray1;
use pyo3::{prelude::*, types::PyType};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

mod embed;
mod union;
mod utils;

const MODULE_PRIME: u64 = 2u64.pow(61) - 1;

#[pyclass]
struct EmbedFunc {
    hash_values: Vec<(u32, u32)>,
    main_col: String,
    idx_col: String,
    n_grams: u32,
    #[pyo3(get)]
    hash_tables: Vec<HashMap<Vec<u8>, HashSet<u32>>>,
    #[pyo3(get)]
    edges: Vec<(u32, u32)>,
    permutations: (ArcArray1<u32>, ArcArray1<u32>),
}
#[derive(IntoPyObject, IntoPyObjectRef)]
enum Sig {
    Signature(Vec<Vec<u8>>),
    Index(u32),
}

// impl IntoPy<PyObject> for Sig {
//     fn into_py(self, py: Python) -> PyObject {
//         match self {
//             Sig::Signature(vec) => {
//                 // Convert each Vec<u8> to PyBytes
//                 let py_list = vec.into_iter()
//                     .map(|bytes| PyBytes::new(py, &bytes).into_py(py))
//                     .collect::<Vec<PyObject>>();

//                 // Convert the Vec<PyObject> to a Python list
//                 py_list.into_py(py)
//             }
//             Sig::Index(index) => {
//                 let py_int: PyObject = index.into_py(py); // Convert u32 to Python integer
//                 py_int
//             }
//         }
//     }
// }

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
    #[new]
    fn new(
        threshold: f64,
        num_perm: u32,
        n_grams: u32,
        false_positive: f64,
        false_negative: f64,
        main_col: &str,
        idx_col: &str,
    ) -> Self {
        let (b, r) = utils::optimal_param(threshold, num_perm, false_positive, false_negative);
        Self::shared_init(b, r, n_grams, num_perm, main_col, idx_col)
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
    #[classmethod]
    fn from_b_r(
        _cls: &Bound<'_, PyType>,
        b: u32,
        r: u32,
        n_grams: u32,
        num_perm: u32,
        main_col: &str,
        idx_col: &str,
    ) -> Self {
        Self::shared_init(b, r, n_grams, num_perm, main_col, idx_col)
    }
    #[staticmethod]
    fn shared_init(
        b: u32,
        r: u32,
        n_grams: u32,
        num_perm: u32,
        main_col: &str,
        idx_col: &str,
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
        let permutations = embed::generate_permutations(MODULE_PRIME as usize, num_perm);
        EmbedFunc {
            hash_values: hash_ranges,
            main_col: main_col.to_string(),
            idx_col: idx_col.to_string(),
            n_grams,
            hash_tables,
            edges,
            permutations,
        }
    }
    ///
    /// Not in use unless its for single line
    ///
    fn embed_func(&self, text: &str, idx: u32) -> HashMap<String, Sig> {
        let hs: Vec<Vec<u8>> = embed::py_embed_func(
            text,
            self.n_grams,
            self.permutations.clone(),
            self.hash_values.to_vec(),
        );

        let mut map = HashMap::new();
        map.insert(self.main_col.to_string(), Sig::Signature(hs));
        map.insert(self.idx_col.to_string(), Sig::Index(idx));
        map
    }

    ///
    /// Add the signature to the hash table
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the hash table
    /// * `hash` - The signature to be added
    /// * `i` - The index of the signature
    fn batch_add(&mut self, hashes: Vec<Vec<u8>>, key: u32) {
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
    fn batch_embed_shard(&mut self, text: Vec<String>, idx: Vec<u32>) {
        let text_idx: Vec<(Vec<Vec<u8>>, u32)> = text
            .par_iter()
            .zip(idx.par_iter())
            .map(|(s, &i)| {
                let mapped = embed::py_embed_func(
                    s,
                    self.n_grams,
                    self.permutations.clone(),
                    self.hash_values.to_vec(),
                );
                (mapped, i)
            })
            .collect();

        text_idx.iter().for_each(|(sig, i)| {
            self.batch_add(sig.clone(), *i);
        });
    }
    ///
    /// Cluster the hash tables
    ///
    /// # Returns
    ///
    /// A UnionFind data structure representing the clusters
    ///
    /// Iterates the hash tables and clusters the signatures
    fn cluster(&mut self) -> union::UnionFind {
        let mut uf = union::UnionFind::new();
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
        uf
    }
}

#[pymodule]
fn dedup_rs(_py: Python, m: Bound<PyModule>) -> PyResult<()> {
    m.add_class::<EmbedFunc>()?;
    m.add_class::<union::UnionFind>()?;
    Ok(())
}

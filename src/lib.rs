use ndarray::{Array, Array1};
use pyo3::{exceptions,prelude::*, types::{PyBytes,PyTuple, PyType}};
use serde::{Serialize,Deserialize};
use std::collections::{HashMap,HashSet};
use rayon::prelude::*;

mod embed;
mod union;
mod utils;

const MODULE_PRIME: u64 = 2u64.pow(61) - 1;

#[pyclass]
#[derive(Clone, Deserialize, Serialize)]
struct EmbedFunc {
    hash_values: Vec<(i32,i32)>,
    main_col: String,
    idx_col: String,
    #[pyo3(get)]
    hash_tables: Vec<HashMap<String, HashSet<i32>>>,
    #[pyo3(get)]
    edges: Vec<(i32, i32)>,
    permutations: (Array1<u64>, Array1<u64>),
}

enum SIG {
    SIGNATURE(Vec<String>),
    INDEX(i32)
}

impl IntoPy<PyObject> for SIG {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            SIG::SIGNATURE(vec) => {
                let py_list: PyObject = vec.into_py(py);  // Convert Vec<Vec<u8>> to Python list[list]
                py_list
            },
            SIG::INDEX(index) => {
                let py_int: PyObject = index.into_py(py);  // Convert i32 to Python integer
                py_int
            },
        }
    }
}

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
    fn new(threshold:f64, num_perm:i32,false_positive:f64, false_negative:f64,
        main_col: &str, idx_col: &str, ) -> Self {
        let (b, r) = utils::optimal_param(threshold, num_perm, false_positive, false_negative);
        Self::shared_init(b, r, num_perm, main_col, idx_col)
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
    fn from_b_r(_cls: &Bound<'_, PyType>, b:i32, r:i32, num_perm:i32, main_col: &str, idx_col: &str) -> Self {

        Self::shared_init(b, r, num_perm, main_col, idx_col)
    }
    #[staticmethod]
    fn shared_init(b:i32, r:i32, num_perm:i32, main_col: &str, idx_col: &str) -> Self {

        let hash_ranges: Vec<(i32, i32)> = (0..b)
                        .map(|i| (i * r, (i + 1) * r))
                        .collect();

        let hash_tables: Vec<HashMap<String, HashSet<i32>>> = vec![HashMap::new(); b as usize];
        let edges: Vec<(i32, i32)> = Vec::new();
        let permutations = embed::generate_permutations(MODULE_PRIME as usize, num_perm);
        EmbedFunc {
            hash_values: hash_ranges,
            main_col: main_col.to_string(),
            idx_col: idx_col.to_string(),
            hash_tables,
            edges,
            permutations,
        }
    }
    ///
    /// Not in use unless its for single line
    /// 
    fn embed_func(&self, text:&str, idx: i32) -> HashMap<String, SIG>{
        let hs: Vec<String> = embed::py_embed_func(&text, self.permutations.clone(),self.hash_values.to_vec());

        let mut map = HashMap::new();
        map.insert(self.main_col.to_string(), SIG::SIGNATURE(hs));
        map.insert(self.idx_col.to_string(), SIG::INDEX(idx));
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
    fn batch_add(&mut self, hashes: Vec<String>, key: i32) {
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
    fn batch_embed_shard(&mut self, text: Vec<String>, idx: Vec<i32>) {
        let text_idx: Vec<(Vec<String>,i32)> = text.par_iter().zip(idx.par_iter())
            .map( |(s, &i)| {
                let mapped =  embed::py_embed_func(&s, self.permutations.clone(),self.hash_values.to_vec());
                (mapped , i)
            }).collect();
        
        text_idx.iter().for_each(|(sig, i)| {
            self.batch_add(sig.clone() , *i);
        });
    }

    fn pyspark_hash(&mut self, text: String, idx: i32) -> Vec<(i32, String, i32)> {
        let hs: Vec<String> = embed::py_embed_func(&text, self.permutations.clone(),self.hash_values.to_vec());
        hs.iter().enumerate().map(|(i, hash)| {
            (i as i32, hash.clone(), idx)
        }).collect()
    }
    ///
    /// Cluster the hash tables
    /// 
    /// # Returns
    /// 
    /// A UnionFind data structure representing the clusters
    /// 
    /// Iterates the hash tables and clusters the signatures
    fn cluster(&mut self) -> union::UnionFind{
        let mut uf = union::UnionFind::new();
        for table in &self.hash_tables {
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
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(self).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle EmbedFunc: {}",
                e
            ))
        })?;
        Ok(PyBytes::new_bound(py, data.as_bytes()).to_object(py))
    }
    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                // Deserialize into a new instance of EmbedFunc
                let new_self: EmbedFunc = serde_json::from_slice(s.as_bytes()).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle EmbedFunc: {}",
                        e
                    ))
                })?;
    
                // Assign the new instance to self
                *self = new_self;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
    
}
#[pyfunction]
fn pyspark_hash(text: String, idx: i32, a: Vec<u64>, b: Vec<u64>, hash_ranges: Vec<(i32, i32)>) -> Vec<(i32, String, i32)> {

    let permutations = (Array1::from_vec(a), Array1::from_vec(b));
    let hs: Vec<String> = embed::py_embed_func(&text, permutations , hash_ranges);
    hs.iter().enumerate().map(|(i, hash)| {
        (i as i32, hash.clone(), idx)
    }).collect()
}
#[pyfunction]
fn pyspark_edges(nodes: Vec<i32>) -> Vec<(i32,i32)> {
    if nodes.len() < 2 {
        Vec::new()
    } else {
        let min_node: i32 = *nodes.iter().min().unwrap();
        let result: Vec<(i32, i32)> = nodes.iter()
            .filter(|&&node| node != min_node)
            .map(|&node| (node, min_node))
            .collect();
        result
    }
}

#[pymodule]
fn dedup_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<EmbedFunc>()?;
    m.add_class::<union::UnionFind>()?;
    m.add_function(wrap_pyfunction!(pyspark_hash, m)?)?;
    m.add_function(wrap_pyfunction!(pyspark_edges, m)?)?;
    Ok(())
}
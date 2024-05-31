use pyo3::{prelude::*, types::PyType};
use std::collections::{HashMap,HashSet};
use rayon::prelude::*;

mod embed;
mod union;

#[pyclass]
struct EmbedFunc {
    hash_values: Vec<(i32,i32)>,
    main_col: String,
    idx_col: String,
    #[pyo3(get)]
    hash_tables: Vec<HashMap<String, HashSet<i32>>>,
    #[pyo3(get)]
    edges: Vec<(i32, i32)>,
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
    #[new]
    fn new(threshold:f64, num_perm:i32,false_positive:f64, false_negative:f64,
        main_col: &str, idx_col: &str, ) -> Self {
        let (b, r) = embed::optimal_param(threshold, num_perm, false_positive, false_negative);
        Self::shared_init(b, r, main_col, idx_col)
    }
    #[classmethod]
    fn from_b_r(_cls: &Bound<'_, PyType>, b:i32, r:i32, main_col: &str, idx_col: &str) -> Self {

        Self::shared_init(b, r, main_col, idx_col)
    }
    #[staticmethod]
    fn shared_init(b:i32, r:i32, main_col: &str, idx_col: &str) -> Self {

        let hash_ranges: Vec<(i32, i32)> = (0..b)
                        .map(|i| (i * r, (i + 1) * r))
                        .collect();

        let hash_tables: Vec<HashMap<String, HashSet<i32>>> = vec![HashMap::new(); b as usize];
        let edges: Vec<(i32, i32)> = Vec::new();
        EmbedFunc {
            hash_values: hash_ranges,
            main_col: main_col.to_string(),
            idx_col: idx_col.to_string(),
            hash_tables: hash_tables,
            edges: edges,
        }
    }

    fn embed_func(&self, text:&str, idx: i32) -> HashMap<String, SIG>{
        let hs: Vec<String> = embed::py_embed_func(&text, self.hash_values.to_vec());

        let mut map = HashMap::new();
        map.insert(self.main_col.to_string(), SIG::SIGNATURE(hs));
        map.insert(self.idx_col.to_string(), SIG::INDEX(idx));
        map

    }

    fn batched_embed_func(&self, text: Vec<String>, idx: Vec<i32>) -> HashMap<String, Vec<SIG>> {

        let new_text : Vec<SIG> = text.par_iter()
            .map(|s| {
                let mapped = embed::py_embed_func(&s, self.hash_values.to_vec());
                SIG::SIGNATURE(mapped)
            }).collect();
        
        HashMap::from([(self.main_col.to_string(), new_text), (self.idx_col.to_string(), idx.into_iter().map(SIG::INDEX).collect())])
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

    fn cluster(&mut self) -> union::UnionFind{
        let mut uf = union::UnionFind::new();
        for table in &self.hash_tables {
            for cluster in table.values() {
                if cluster.len() <= 1 {
                    continue;
                }
                let idx = *cluster.iter().min().expect("Cluster should not be empty");
                for &x in cluster {
                    self.edges.push((x, idx));
                    uf.union(x as usize, idx as usize);
                }
            }
        }
        uf
    }

}

#[pymodule]
fn dedup_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(shingle, m)?)?;
    // // m.add_function(wrap_pyfunction!(shuffle, m)?)?;
    // m.add_function(wrap_pyfunction!(jaccard, m)?)?;
    // m.add_function(wrap_pyfunction!(one_hot, m)?)?;
    m.add_class::<EmbedFunc>()?;
    Ok(())
}

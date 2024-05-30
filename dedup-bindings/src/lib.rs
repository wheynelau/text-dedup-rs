
use pyo3::prelude::*;
use std::collections::BTreeMap;
use rayon::prelude::*;

mod embed;
mod union;

#[pyclass]
struct EmbedFunc {
    b: i32,
    r: i32,
    hash_values: Vec<(i32,i32)>,
    main_col: String,
    idx_col: String
}

enum SIG {
    SIGNATURE(Vec<String>),
    INDEX(i32)
}

impl IntoPy<PyObject> for SIG {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            SIG::SIGNATURE(vec) => {
                let py_list: PyObject = vec.into_py(py);  // Convert Vec<String> to Python list
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
        main_col: &str, idx_col: &str) -> Self {
        let (B, R) = embed::optimal_param(threshold, num_perm, false_positive, false_negative);
        let hash_ranges: Vec<(i32, i32)> = (0..B)
                        .map(|i| (i * R, (i + 1) * R))
                        .collect();
        
        EmbedFunc {
            b: 50,
            r: 4,
            hash_values: hash_ranges,
            main_col: main_col.to_string(),
            idx_col: idx_col.to_string()

        }
    }
    pub fn embed_func(&self, text:&str, idx: i32) -> BTreeMap<String, SIG>{
        let hs: Vec<String> = embed::py_embed_func(&text, self.hash_values.to_vec());

        let mut map = BTreeMap::new();
        map.insert(self.main_col.to_string(), SIG::SIGNATURE(hs));
        map.insert(self.idx_col.to_string(), SIG::INDEX(idx));
        map

    }

    fn batched_embed_func(&self, text: Vec<String>, idx: Vec<i32>) -> BTreeMap<String, Vec<SIG>> {

        let new_text : Vec<SIG> = text.par_iter()
            .map(|s| {
                let mapped = (embed::py_embed_func(&s, self.hash_values.to_vec()));
                SIG::SIGNATURE(mapped)
            }).collect();
        
        BTreeMap::from([(self.main_col.to_string(), new_text)])
    }

}
// #[pyfunction]
// fn shingle(text: String, k: usize) -> PyResult<HashSet<String>> {
//     let text = text.to_lowercase();
//     let mut shingles: HashSet<String> = HashSet::new();

//     // Directly iterate over character windows
//     let chars: Vec<char> = text.chars().collect();
//     if chars.len() < k {
//         return Ok(shingles);
//     }

//     for i in 0..=chars.len() - k {
//         let shingle: String = chars[i..i + k].iter().collect();
//         shingles.insert(shingle);
//     }

//     Ok(shingles)
// }

// #[pyfunction]
// fn one_hot(a: HashSet<String>, vocab: Vec<String>) -> PyResult<Vec<i32>> {
//     let mut one_hot_vec = vec![0; vocab.len()];
//     for (i, word) in vocab.iter().enumerate() {
//         if a.contains(word) {
//             one_hot_vec[i] = 1;
//         }
//     }
//     Ok(one_hot_vec)
// }

// fn embed_func() {

//     let modulo_prime: i32 = i32::pow(2, 61) -1;
//     let seed: u8 = 42;


//     let mut rng: SmallRng = SeedableRng::from_seed([seed; 32]);
//     let a: Vec<i32> = (1..modulo_prime).map(|_| rng.gen_range(1..modulo_prime)).collect();

//     // Generate the second part of the permutation
//     let b: Vec<i32> = (0..modulo_prime).map(|_| rng.gen_range(0..modulo_prime)).collect();

//     // Combine both parts into a tuple
//     let permutations = (a, b);
// }

// // #[pyfunction]
// // fn shuffle(py: Python, list: &PyList) -> PyResult<PyObject> {
// //     let mut rng = thread_rng();
// //     // Extract elements from PyList and convert them to Vec<usize>
// //     let mut vec: Vec<usize> = list
// //         .extract::<Vec<usize>>()?; // Attempt to convert Python list to Vec<usize>

// //     // Shuffle the vector
// //     vec.shuffle(&mut rng); 

// //     // Convert the shuffled Vec<usize> back to a Python list
// //     let shuffled_list = PyList::new(py, &vec);
// //     Ok(shuffled_list.to_object(py))
// // }
// // #[pyfunction]
// // fn hash_vector(size:usize) {

// // }

// #[pyfunction]
// fn jaccard(x: HashSet<i32>, y: HashSet<i32>) -> f32 {
//     let intersection = x.intersection(&y).count() as f32;
//     let union = x.union(&y).count() as f32;
//     intersection / union
// }
/// A Python module implemented in Rust.
#[pymodule]
fn dedup_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(shingle, m)?)?;
    // // m.add_function(wrap_pyfunction!(shuffle, m)?)?;
    // m.add_function(wrap_pyfunction!(jaccard, m)?)?;
    // m.add_function(wrap_pyfunction!(one_hot, m)?)?;
    m.add_class::<EmbedFunc>()?;
    Ok(())
}

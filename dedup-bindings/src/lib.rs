use ndarray::Array1;
use pyo3::prelude::*;
use std::{sync::Arc};

mod embed;

#[pyclass]
struct EmbedFunc {
    b: i32,
    r: i32,
    hash_values: Vec<(i32,i32)>,
}

#[pymethods]
impl EmbedFunc {
    #[new]
    fn new(threshold:f64, num_perm:i32,false_positive:f64, false_negative:f64) -> Self {

        let (B, R) = embed::optimal_param(threshold, num_perm, false_positive, false_negative);
        let hash_ranges: Vec<(i32, i32)> = (0..B)
                        .map(|i| (i * R, (i + 1) * R))
                        .collect();
        
        EmbedFunc {
            b: B,
            r: R,
            hash_values: hash_ranges
        }
    }
    fn embed_func(&self, text:&str) -> Vec<String>{
        embed::py_embed_func(&text, self.hash_values.to_vec())
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

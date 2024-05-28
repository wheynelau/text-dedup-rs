use ndarray::Array1;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::{rngs::SmallRng, Rng, SeedableRng}; // Add the missing import
use std::collections::HashSet;
use rand::{seq::SliceRandom, thread_rng};

mod embed;

use sha1::{Sha1, Digest};

#[derive(Debug)]
struct EmbedFuncParams {
    num_perm: usize,
    ngram_size: usize,
    min_length: usize,
    hashranges: Vec<(usize, usize)>,
    permutations: (Array1<u32>, Array1<u32>),
    max_hash: u32,
    modulo_prime: u32,
}


#[pyfunction]
fn shingle(text: String, k: usize) -> PyResult<HashSet<String>> {
    let text = text.to_lowercase();
    let mut shingles: HashSet<String> = HashSet::new();

    // Directly iterate over character windows
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < k {
        return Ok(shingles);
    }

    for i in 0..=chars.len() - k {
        let shingle: String = chars[i..i + k].iter().collect();
        shingles.insert(shingle);
    }

    Ok(shingles)
}

#[pyfunction]
fn one_hot(a: HashSet<String>, vocab: Vec<String>) -> PyResult<Vec<i32>> {
    let mut one_hot_vec = vec![0; vocab.len()];
    for (i, word) in vocab.iter().enumerate() {
        if a.contains(word) {
            one_hot_vec[i] = 1;
        }
    }
    Ok(one_hot_vec)
}

fn embed_func() {

    let modulo_prime: i32 = i32::pow(2, 61) -1;
    let seed: u8 = 42;


    let mut rng: SmallRng = SeedableRng::from_seed([seed; 32]);
    let a: Vec<i32> = (1..modulo_prime).map(|_| rng.gen_range(1..modulo_prime)).collect();

    // Generate the second part of the permutation
    let b: Vec<i32> = (0..modulo_prime).map(|_| rng.gen_range(0..modulo_prime)).collect();

    // Combine both parts into a tuple
    let permutations = (a, b);
}

// #[pyfunction]
// fn shuffle(py: Python, list: &PyList) -> PyResult<PyObject> {
//     let mut rng = thread_rng();
//     // Extract elements from PyList and convert them to Vec<usize>
//     let mut vec: Vec<usize> = list
//         .extract::<Vec<usize>>()?; // Attempt to convert Python list to Vec<usize>

//     // Shuffle the vector
//     vec.shuffle(&mut rng); 

//     // Convert the shuffled Vec<usize> back to a Python list
//     let shuffled_list = PyList::new(py, &vec);
//     Ok(shuffled_list.to_object(py))
// }
// #[pyfunction]
// fn hash_vector(size:usize) {

// }

#[pyfunction]
fn jaccard(x: HashSet<i32>, y: HashSet<i32>) -> f32 {
    let intersection = x.intersection(&y).count() as f32;
    let union = x.union(&y).count() as f32;
    intersection / union
}
/// A Python module implemented in Rust.
#[pymodule]
fn dedup_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shingle, m)?)?;
    // m.add_function(wrap_pyfunction!(shuffle, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(one_hot, m)?)?;
    Ok(())
}

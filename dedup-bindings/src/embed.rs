use std::{collections::BTreeSet, sync::Arc, time::Instant};
use rand::{distributions::Uniform, Rng, SeedableRng};
use regex::Regex;
use sha1::{Sha1, Digest};
use byteorder::{ByteOrder, LittleEndian};
use ndarray::{ArcArray, ArcArray1, ArcArray2};
use rayon::prelude::*;

#[derive(Debug)]
pub struct EmbedFuncParams {
    // content: String,
    // idx: usize,
    // num_perm: usize,
    // ngram_size: usize,
    // min_length: usize,
    // hashranges: Vec<(usize, usize)>,
    // permutations: (Vec<u32>, Vec<u32>),
    // max_hash: u32,
    // modulo_prime: u32,
}

pub fn embed_func(params: &EmbedFuncParams) {

    let modulo_prime: i32 = i32::pow(2, 61) -1;
    let seed: u8 = 42;

    let mut rng: rand::rngs::StdRng = SeedableRng::from_seed([seed; 32]);
    let a: Vec<i32> = (1..modulo_prime)
                    .map(|_| rng.gen_range(1..modulo_prime))
                    .collect();

    // Generate the second part of the permutation
    let b: Vec<i32> = (0..modulo_prime)
                    .map(|_| rng.gen_range(0..modulo_prime))
                    .collect();

    // Combine both parts into a tuple
    let permutations = (a, b);
}

fn ngrams(sequence: Vec<&str>, n: usize, min_length: usize) -> Vec<Vec<&str>> {
    if sequence.len() < min_length {
        return vec![];
    }
    if sequence.len() < n {
        return vec![sequence];
    }
    sequence.windows(n).map(|window| window.to_vec()).collect()
}

fn sha1_hash(data: &[u8], d: u32) -> u64 {
    let mut hasher = Sha1::new();
    hasher.update(data);
    let result = hasher.finalize();

    match d {
        32 => LittleEndian::read_u32(&result[0..4]) as u64,
        64 => LittleEndian::read_u64(&result[0..8]),
        _ => {
            let bytes_to_read = (d / 8) as usize;
            let mut buffer = vec![0; bytes_to_read];
            buffer.copy_from_slice(&result[0..bytes_to_read]);
            LittleEndian::read_uint(&buffer, bytes_to_read)
        },
    }
}
pub fn tokenize(text: String, n: usize, min_length: usize) -> BTreeSet<Arc<[u8]>> {
    let text: String = text.to_lowercase();

    let re = Regex::new(r"\W+").unwrap();

    let filtered_content: Vec<&str> = re.split(&text)
                                        .filter(|s| !s.is_empty())
                                        .collect();
    
    let tokens: BTreeSet<Arc<[u8]>> = ngrams(filtered_content, n, min_length)
    .into_iter()
    .map(|vec| vec.join(" "))
    .map(|s| s.into_bytes()) // Convert String to Vec<u8>
    .map(Arc::from)    // Convert Vec<u8> to Arc<[u8]>
    .collect();
    
    tokens
}
pub fn hash_tokens(tokens: BTreeSet<Arc<[u8]>>, d: u32) -> Vec<u64> {
    tokens.iter()
        .map(|token| sha1_hash(token, d))
        .collect()
}

fn generate_permutations(modulo_prime: usize) -> (ArcArray1<u64>, ArcArray1<u64>) {
    let mut rng: rand::rngs::StdRng = SeedableRng::from_seed([42; 32]);
    let dist_a = Uniform::new(1, modulo_prime); // Range is [1, modulo_prime)
    let dist_b = Uniform::new(0, modulo_prime); // Range is [0, modulo_prime)

    let a: ArcArray1<u64> = (0..200) // Assuming you want 200 elements as per previous context
        .map(|_| rng.sample(&dist_a) as u64)
        .collect::<Vec<_>>()
        .into_iter()
        .collect();

    let b: ArcArray1<u64> = (0..200) // Assuming you want 200 elements as per previous context
        .map(|_| rng.sample(&dist_b) as u64)
        .collect::<Vec<_>>()
        .into_iter()
        .collect();

    (a, b)
}

fn riemann_sum(f: fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let sum = (1..n)
        .map(|i| f(a + i as f64 * h))
        .sum::<f64>();
    h * ((f(a) + f(b)) / 2.0 + sum)
}

// fn optimal_param(threshold: f64, 
//     num_perm: usize, 
//     false_positive_weight:f64,
//     false_negative_weight: f64) -> (usize, usize) {

    
// }

fn main() {
    let text = "But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness. No one rejects, dislikes, or avoids pleasure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally encounter consequences that are extremely painful. Nor again is there anyone who loves or pursues or desires to obtain pain of itself, because it is pain, but because occasionally circumstances occur in which toil and pain can procure him some great pleasure. To take a trivial example, which of us ever undertakes laborious physical exercise, except to obtain some advantage from it? But who has any right to find fault with a man who chooses to enjoy a pleasure that has no annoying consequences, or one who avoids a pain that produces no resultant pleasure?".to_string();
    let n = 3;
    let min_length = 5;

    // Start timing the tokenization process
    let start_tokenization = Instant::now();
    let tokens = tokenize(text, n, min_length);
    let tokenization_duration = start_tokenization.elapsed();

    dbg!(tokenization_duration);

    println!("total number= {}", tokens.len());

    let d = 64;

    // Start timing the hashing process
    let start_hashing = Instant::now();

    let modulo_prime = 2u64.pow(61) - 1;
    let max_hash = 2u32.pow(d) - 1;
    let (a, b) = generate_permutations(modulo_prime as usize);

    let hashing_duration = start_hashing.elapsed();
    dbg!(hashing_duration);
    dbg!(a.len());
    dbg!(b.len());

    let hashes: Vec<u64> = hash_tokens(tokens, d);

    dbg!(hashes.len());

    let start_hashing = Instant::now();

    let mut result: Vec<Vec<u64>> = Vec::new();

    for hash in hashes.iter() {
        let mut inner_vec: Vec<u64> = Vec::new();  // Create a vector to hold the computed hashes for this iteration
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let computed_hash = ((hash * a_val + b_val) % modulo_prime as u64) & max_hash as u64;
            inner_vec.push(computed_hash);  // Store each computed hash in the inner vector
        }
        result.push(inner_vec);  // Add the inner vector to the outer vector after each iteration of the inner loop
    }

    let hashing_duration = start_hashing.elapsed();

    dbg!(hashing_duration);

    // for ((hash, a_row), &b_val) in hashes.iter_mut().zip(genrows(&a)).zip(b.iter()) {
    //     *hash = ((*hash * a_row + b_val) % modulo_prime) & max_hash;
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_function(x: f64) -> f64 {
        x * x  // A simple function, f(x) = x^2
    }

    #[test]
    fn test_riemann_sum() {
        let a = 0.0;
        let b = 1.0;
        let n = 1000;
        let result = riemann_sum(test_function, a, b, n);
        let expected = 1.0 / 3.0;  // The integral of x^2 from 0 to 1 is 1/3

        // Assert that the result is close to the expected value
        let tolerance = 0.001;
        println!("Result: {}, Expected: {}", result, expected);
        assert!((result - expected).abs() < tolerance, "The calculated integral was not within the expected tolerance");
    }
}

use lazy_static::lazy_static;
use std::collections::BTreeSet;
use rand::{distributions::Uniform, Rng, SeedableRng};
use regex::Regex;
use sha1::{Sha1, Digest};
use byteorder::{ByteOrder, LittleEndian};
use ndarray::ArcArray1;


const D :u32 = 32;
const RIEMANN_DIVISIONS: u32 = 100;
const MODULE_PRIME: u64 = 2u64.pow(61) - 1;
const MAX_HASH:u64 = 2u64.pow(D) - 1;
const N: i32 = 3;
const MIN_LENGTH:i32 = 5;

lazy_static! {
    static ref RE:Regex = Regex::new(r"\W+").unwrap();
}

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
fn ngrams(sequence: Vec<&str>, n: i32, min_length: i32) -> Vec<Vec<&str>> {
    if sequence.len() < min_length as usize {
        return vec![];
    }
    if sequence.len() < n as usize {
        return vec![sequence];
    }
    sequence.windows(n as usize).map(|window| window.to_vec()).collect()
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
pub fn tokenize(text: &str, n: i32, min_length: i32) -> BTreeSet<Vec<u8>> {
    // let text: String = text.to_lowercase();

    let filtered_content: Vec<&str> = RE.split(&text)
                                        .filter(|s| !s.is_empty())
                                        .collect();
    
    let tokens: BTreeSet<Vec<u8>> = ngrams(filtered_content, n, min_length)
    .into_iter()
    .map(|vec| vec.join(" "))
    .map(|s| s.into_bytes()) // Convert String to Vec<u8>
    .collect();
    
    tokens
}
pub fn hash_tokens(tokens: BTreeSet<Vec<u8>>, d: u32) -> Vec<u64> {
    tokens.iter()
        .map(|token| sha1_hash(token, d))
        .collect()
}

fn generate_permutations(module_prime: usize) -> (ArcArray1<u64>, ArcArray1<u64>) {
    let mut rng: rand::rngs::StdRng = SeedableRng::from_seed([42; 32]);
    let dist_a = Uniform::new(1, module_prime); // Range is [1, modulo_prime)
    let dist_b = Uniform::new(0, module_prime); // Range is [0, modulo_prime)

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
fn riemann_sum(f: impl Fn(f64) -> f64, a: f64, b: f64, n: u32) -> f64 {
    let h = (b - a) / n as f64;
    let sum = (1..n)
        .map(|i| f(a + i as f64 * h))
        .sum::<f64>();
    h * ((f(a) + f(b)) / 2.0 + sum)
}

fn false_positive_area(threshold: f64, b:i32, r: i32) -> f64 {
    let proba = |s: f64| -> f64 {
        1.0 - (1.0 - s.powi(r)).powi(b)
    };
    riemann_sum(proba, 0.0, threshold, RIEMANN_DIVISIONS)
}

fn false_negative_area(threshold: f64, b: i32, r: i32) -> f64 {
    let proba = |s: f64| -> f64 {
        1.0 - (1.0 - (1.0 - s.powi(r)).powi(b))
    };
    riemann_sum(proba, threshold, 1.0, RIEMANN_DIVISIONS)
}

pub fn optimal_param(threshold: f64, 
    num_perm: i32, 
    false_positive_weight:f64,
    false_negative_weight: f64) -> (i32, i32) {

    let mut min_error:f64 = f64::INFINITY;

    let mut opt: (i32, i32) = (0,0);

    for b in 1..(num_perm + 1) as i32 {
        let max_r: i32 = num_perm / b;
        for r in 1..max_r + 1 as i32 {
            let false_positive = false_positive_area(threshold, b, r);
            let false_negative = false_negative_area(threshold, b, r);
            let error = false_positive_weight * false_positive + false_negative_weight * false_negative;
            if error < min_error {
                min_error = error;
                opt = (b, r);
            }
        }
    }
    opt
}


pub fn py_embed_func(text: &str, hash_ranges:Vec<(i32,i32)>) -> Vec<String> {

    let tokens = tokenize(&text, N, MIN_LENGTH);

    let (a, b) = generate_permutations(MODULE_PRIME as usize);

    let hashes: Vec<u64> = hash_tokens(tokens, D);

    // dbg!(&hashes[0]);

    let mut result: Vec<Vec<u64>> = Vec::with_capacity(hashes.len() + 1);

    let mut inner_vec: Vec<u64> = Vec::with_capacity(a.len());
    for hash in hashes.iter() {
        inner_vec.clear();  // Create a vector to hold the computed hashes for this iteration
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let computed_hash = hash.wrapping_mul(*a_val).wrapping_add(*b_val) % MODULE_PRIME & MAX_HASH ;
            inner_vec.push(computed_hash);  // Store each computed hash in the inner vector
        }
        result.push(inner_vec.clone());  // Add the inner vector to the outer vector after each iteration of the inner loop
    }

    let max_hash_vec: Vec<u64> = vec![MAX_HASH as u64; a.len()];
    result.push(max_hash_vec);

    // dbg!(&result[0]);

    // find the min of each column
    let num_cols = result[0].len();
    let mut hashvalues: Vec<u64> = Vec::with_capacity(num_cols);

    for col in 0..num_cols {
        let min_value = result.iter()
                             .map(|row| row[col])
                             .min()
                             .expect("Expected at least one row");
        hashvalues.push(min_value);
    }

    let hs: Vec<String> = hash_ranges.iter().map(|(start, end)| {
        let start = *start as usize;
        let end = *end as usize;
        let inner_vec: Vec<u8> = hashvalues[start..end].iter().flat_map(|&x| x.to_le_bytes().to_vec()).collect();
        hex::encode(&inner_vec)
    }).collect();

    hs
}

fn main() {
    let text = "But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness. No one rejects, dislikes, or avoids pleasure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally encounter consequences that are extremely painful. Nor again is there anyone who loves or pursues or desires to obtain pain of itself, because it is pain, but because occasionally circumstances occur in which toil and pain can procure him some great pleasure. To take a trivial example, which of us ever undertakes laborious physical exercise, except to obtain some advantage from it? But who has any right to find fault with a man who chooses to enjoy a pleasure that has no annoying consequences, or one who avoids a pain that produces no resultant pleasure?".to_string();

    // These functions are fixed for all iterations

    let (b, r) = optimal_param(0.5, 200, 0.5, 0.5);

    let hash_ranges: Vec<(i32,i32)> = (0..b)
                    .map(|i| (i*r, (i+1)*r))
                    .collect();

    for _ in 0..100000 {
        py_embed_func(&text, hash_ranges.clone());
    }
    // py_embed_func(&text, hash_ranges.clone());
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

    #[test]
    fn test_embed_fn() {
        let threshold = 0.5;
        let num_perm = 200;
        let false_positive_weight = 0.5;
        let false_negative_weight = 0.5;

        let (b, r) = optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight);
        let hash_ranges: Vec<(i32, i32)> = (0..b)
                        .map(|i| (i * r, (i + 1) * r))
                        .collect();

        let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";
        py_embed_func(text, hash_ranges);
    }
}

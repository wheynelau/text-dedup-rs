use lazy_static::lazy_static;
use std::collections::HashSet;
use rand::{distributions::Uniform, Rng, SeedableRng};
use regex::Regex;
use sha1::{Sha1, Digest};
use byteorder::{ByteOrder, LittleEndian};
use ndarray::ArcArray1;
use base64::{Engine as _, engine::general_purpose};

use crate::utils;

/// TODO: Remove hardcodes
const D :u32 = 32;
const MODULE_PRIME: u64 = 2u64.pow(61) - 1;
const MAX_HASH:u64 = 2u64.pow(D) - 1;
const N: i32 = 2;
const MIN_LENGTH:i32 = 5;

lazy_static! {
    static ref RE:Regex = Regex::new(r"\W+").unwrap();
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

fn sha1_hash(data: &[u8]) -> u64 {
    let d = D;
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
fn tokenize(text: &str, n: i32, min_length: i32) -> HashSet<Vec<u8>> {
    // let text: String = text.to_lowercase();

    let filtered_content: Vec<&str> = RE.split(&text)
                                        .filter(|s| !s.is_empty())
                                        .collect();
    
    let tokens: HashSet<Vec<u8>> = ngrams(filtered_content, n, min_length)
    .into_iter()
    .map(|vec| {
        let mut bytes = Vec::new();
        for (i, s) in vec.iter().enumerate() {
            if i > 0 {
                bytes.push(32); // Append a space before each word except the first
            }
            bytes.extend_from_slice(s.as_bytes());
        }
        bytes
    })
    .collect();
    
    tokens
}
pub fn hash_tokens(tokens: HashSet<Vec<u8>>) -> Vec<u64> {
    tokens.iter()
        .map(|token| sha1_hash(token))
        .collect()
}

pub fn generate_permutations(module_prime: usize, num_perm:i32) -> (ArcArray1<u64>, ArcArray1<u64>) {
    let mut rng: rand::rngs::StdRng = SeedableRng::from_seed([42; 32]);
    let dist_a = Uniform::new(1, module_prime); // Range is [1, modulo_prime)
    let dist_b = Uniform::new(0, module_prime); // Range is [0, modulo_prime)

    let a: ArcArray1<u64> = (0..num_perm) // Assuming you want NUM_PERM elements as per previous context
        .map(|_| rng.sample(&dist_a) as u64)
        .collect::<Vec<_>>()
        .into_iter()
        .collect();

    let b: ArcArray1<u64> = (0..num_perm) // Assuming you want NUM_PERM elements as per previous context
        .map(|_| rng.sample(&dist_b) as u64)
        .collect::<Vec<_>>()
        .into_iter()
        .collect();

    (a, b)
}


pub fn py_embed_func(text: &str, permutations: (ArcArray1<u64>, ArcArray1<u64>), hash_ranges:Vec<(i32,i32)>) -> Vec<String> {

    let (a, b) = permutations;

    let tokens = tokenize(&text, N, MIN_LENGTH);

    let hashes: Vec<u64> = hash_tokens(tokens);

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
        let inner_vec : Vec<u8> = hashvalues[start..end].iter().flat_map(|&x| x.to_le_bytes().to_vec()).collect();
        general_purpose::STANDARD.encode(&inner_vec)
    }).collect();
    // let hs: Vec<Vec<u8>> = hash_ranges.iter().map(|&(start, end)| {
    //     let slice = &hashvalues[start as usize..end as usize]; // Convert the range to usize
    //     let mut swapped_bytes = Vec::new();
    //     for &value in slice {
    //         // Assuming `hashvalues` is a slice of u32 or similar, adjust the type accordingly
    //         swapped_bytes.extend_from_slice(&value.to_be_bytes());
    //     }
    //     swapped_bytes
    // }).collect();
    hs
}

#[allow(dead_code)]
fn main() {
    let text = "But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness. No one rejects, dislikes, or avoids pleasure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally encounter consequences that are extremely painful. Nor again is there anyone who loves or pursues or desires to obtain pain of itself, because it is pain, but because occasionally circumstances occur in which toil and pain can procure him some great pleasure. To take a trivial example, which of us ever undertakes laborious physical exercise, except to obtain some advantage from it? But who has any right to find fault with a man who chooses to enjoy a pleasure that has no annoying consequences, or one who avoids a pain that produces no resultant pleasure?".to_string();

    // These functions are fixed for all iterations
    let num_perm = 200;
    let (b, r) = utils::optimal_param(0.5, num_perm, 0.5, 0.5);
    let permutations = generate_permutations(MODULE_PRIME as usize, num_perm);
    let hash_ranges: Vec<(i32,i32)> = (0..b)
                    .map(|i| (i*r, (i+1)*r))
                    .collect();

    for _ in 0..100000 {
        py_embed_func(&text, permutations.clone() ,hash_ranges.clone());
    }
    // py_embed_func(&text, hash_ranges.clone());
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_fn() {
        let threshold = 0.5;
        let num_perm = 200;
        let false_positive_weight = 0.5;
        let false_negative_weight = 0.5;

        let (b, r) = utils::optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight);
        let permutations = generate_permutations(MODULE_PRIME as usize, num_perm);
        let hash_ranges: Vec<(i32, i32)> = (0..b)
                        .map(|i| (i * r, (i + 1) * r))
                        .collect();

        let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";
        py_embed_func(text,permutations, hash_ranges);
    }
}

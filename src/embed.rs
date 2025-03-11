use byteorder::{ByteOrder, LittleEndian};
use lazy_static::lazy_static;
use ndarray::ArcArray1;
use rand::{distr::Uniform, Rng, SeedableRng};
use regex::Regex;
use sha1::Sha1;
use sha3::{Digest, Sha3_256};
use std::{collections::HashSet, ops::{BitAnd, Rem}};
use num_traits::{WrappingMul, WrappingAdd};

/// TODO: Remove hardcodes
const D: u32 = 32;
const MODULE_PRIME: u64 = (1u64 << 61) - 1;
const MAX_HASH: u64 = (1u64 << 32) - 1; // no difference from u32::MAX?

lazy_static! {
    static ref RE: Regex = Regex::new(r"\W").unwrap();
}

fn ngrams(sequence: Vec<String>, n: &u32, min_length: &u32) -> Vec<Vec<String>> {
    if sequence.len() < *min_length as usize {
        return vec![];
    }
    if sequence.len() < *n as usize {
        return vec![sequence];
    }
    sequence
        .windows(*n as usize)
        .map(|window| window.to_vec())
        .collect()
}

#[allow(dead_code)]
fn sha1_hash(data: &[u8]) -> u32 {
    let mut hasher = Sha1::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut buf = [0; 4];
    buf.copy_from_slice(&result[0..4]);
    u32::from_le_bytes(buf)
}

#[allow(dead_code)]
fn sha3_hash(data: &[u8]) -> u64 {
    let d = D;
    let mut hasher = Sha3_256::new();
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
        }
    }
}

fn split_text(text: &str) -> Vec<String> {
    RE.split(text).map(|s| s.to_ascii_lowercase()).collect()
}

fn tokenize(text: &str, n: &u32, min_length: &u32) -> HashSet<Vec<u8>> {
    // let text: String = text.to_lowercase();

    let filtered_content: Vec<String> = split_text(text);

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
fn hash_tokens(tokens: HashSet<Vec<u8>>) -> Vec<u64> {
    tokens.iter().map(|token| sha1_hash(token) as u64).collect()
}

pub fn generate_permutations(
    module_prime: usize,
    num_perm: u32,
) -> (ArcArray1<u64>, ArcArray1<u64>) {
    let mut rng: rand::rngs::StdRng = SeedableRng::from_seed([42; 32]);
    let dist_a = Uniform::try_from(1..module_prime).expect("Distribution failed?"); // Range is [1, modulo_prime)
    let dist_b = Uniform::try_from(0..module_prime).expect("Distribution failed?"); // Range is [0, modulo_prime)

    let a: ArcArray1<u64> = (0..num_perm)
        .map(|_| rng.sample(dist_a) as u64)
        .collect::<Vec<_>>()
        .into_iter()
        .collect();

    let b: ArcArray1<u64> = (0..num_perm)
        .map(|_| rng.sample(dist_b) as u64)
        .collect::<Vec<_>>()
        .into_iter()
        .collect();

    (a, b)
}
/// Permutate the hash
///
/// Reference python code
/// ```python
///  hashvalues = (hashvalues * a + b) % modulo_prime & max_hash
/// ```
/// Also note that it adds the max hash
#[allow(dead_code)]
fn permute_hashes(
    hashvalues: Vec<u32>,
    a: &ArcArray1<u32>,
    b: &ArcArray1<u32>,
    modulo_prime: u32,
    max_hash: u32,
) -> Vec<Vec<u32>> {
    let mut new_hashvalues: Vec<Vec<u32>> = Vec::with_capacity(hashvalues.len());
    let mut inner_vec: Vec<u32> = Vec::with_capacity(a.len());
    for hash in hashvalues.iter() {
        inner_vec.clear(); // Create a vector to hold the computed hashes for this iteration
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let computed_hash =
                (hash.wrapping_mul(a_val).wrapping_add(*b_val) % modulo_prime) & max_hash;
            inner_vec.push(computed_hash); // Store each computed hash in the inner vector
        }
        new_hashvalues.push(inner_vec.clone()); // Add the inner vector to the outer vector after each iteration of the inner loop
    }
    let max_hash_vec: Vec<u32> = vec![max_hash; a.len()];
    new_hashvalues.push(max_hash_vec);

    new_hashvalues
}
// Keep this code as reference
#[allow(dead_code)]
fn find_min(hashvalues: Vec<Vec<u32>>) -> Vec<u32> {
    let num_cols = hashvalues[0].len();
    let mut min_values: Vec<u32> = Vec::with_capacity(num_cols);
    for col in 0..num_cols {
        let min_value = hashvalues
            .iter()
            .map(|row| row[col])
            .min()
            .expect("Expected at least one row");
        min_values.push(min_value);
    }
    min_values
}
// helper function
fn hash_helper<T>(hash: T, a: T, b: T, modulo_prime: T, max_hash: T) -> T
where T: WrappingMul + WrappingAdd + Copy + Rem<Output = T> + BitAnd<Output = T>{
    (hash.wrapping_mul(&a).wrapping_add(&b) % modulo_prime) & max_hash
}

// fused min_hash, uses a single function, with flat datastructure
fn min_hash_fused(
    hashvalues: &[u64],
    a: &ArcArray1<u64>,
    b: &ArcArray1<u64>,
    modulo_prime: u64,
    max_hash: u64,
) -> Vec<u64> {
    let mut min_values = vec![max_hash; a.len()];

    // Restructured loop order for better cache locality
    for &hash in hashvalues {
        for ((&a_val, &b_val), min_val) in a.iter().zip(b.iter()).zip(min_values.iter_mut()) {
            let computed_hash = hash_helper::<u64>(hash, a_val, b_val, modulo_prime, max_hash);
            if computed_hash < *min_val {
                *min_val = computed_hash;
            }
        }
    }

    min_values
}
fn swap_bytes(hashvalues: &[u64], hash_ranges: &[(u32, u32)]) -> Vec<Vec<u8>> {
    hash_ranges
        .iter()
        .map(|(start, end)| {
            let start = *start as usize;
            let end = *end as usize;
            if start >= hashvalues.len() {
                return Vec::new();
            }
            let actual_end = std::cmp::min(end, hashvalues.len());
            let inner_vec: Vec<u8> = hashvalues[start..actual_end]
                .iter()
                .flat_map(|&x| x.swap_bytes().to_le_bytes().to_vec())
                .collect();
            inner_vec
        })
        .collect()
}

pub fn py_embed_func(
    text: &str,
    n_grams: &u32,
    permutations: &(ArcArray1<u64>, ArcArray1<u64>),
    hash_ranges: &[(u32, u32)],
    min_length: &u32,
) -> Vec<Vec<u8>> {
    let (a, b) = permutations;

    let tokens = tokenize(text, n_grams, min_length);

    let hashes: Vec<u64> = hash_tokens(tokens);

    let hashvalues = min_hash_fused(&hashes, a, b, MODULE_PRIME, MAX_HASH);

    swap_bytes(&hashvalues, hash_ranges)
}
// TESTS ARE BROKEN
// More info: u32 was testing to be precise
// But realised that in the benchmarks, python was done in u64
// Therefore, need to modify the test for u64 as well
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_fn() {
        let num_perm = 200;

        let (b, r) = (50, 4);
        let permutations = generate_permutations(MODULE_PRIME as usize, num_perm);
        let hash_ranges: Vec<(u32, u32)> = (0..b).map(|i| (i * r, (i + 1) * r)).collect();
        let n = 2;
        let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";
        py_embed_func(text, &n, &permutations, &hash_ranges, &5);
    }
    #[test]
    fn test_split_text() {
        let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
        let baseline = vec![
            "lorem",
            "ipsum",
            "dolor",
            "sit",
            "amet",
            "",
            "consectetur",
            "adipiscing",
            "elit",
            "",
        ];
        let result = split_text(text);
        assert_eq!(result, baseline);
    }
    #[test]
    fn test_ngrams() {
        let n = 3;
        let min_length = 5;
        let text = vec![
            "lorem",
            "ipsum",
            "dolor",
            "sit",
            "amet",
            "",
            "consectetur",
            "adipiscing",
            "elit",
            "",
        ];
        let text = text.iter().map(|s| s.to_string()).collect();
        let ngrams = ngrams(text, &n, &min_length);
        // from python
        assert_eq!(ngrams.len(), 8);
    }
    #[test]
    fn test_tokenize() {
        // the vec values were produced by the python function
        let baseline: HashSet<Vec<u8>> = HashSet::from_iter(vec![
            vec![97, 109, 101, 116, 32, 32],
            vec![
                105, 112, 115, 117, 109, 32, 100, 111, 108, 111, 114, 32, 115, 105, 116,
            ],
            vec![
                100, 111, 108, 111, 114, 32, 115, 105, 116, 32, 97, 109, 101, 116,
            ],
            vec![
                108, 111, 114, 101, 109, 32, 105, 112, 115, 117, 109, 32, 100, 111, 108, 111, 114,
            ],
            vec![115, 105, 116, 32, 97, 109, 101, 116, 32],
        ]);

        let n = 3;
        let min_length = 5;
        let text = "Lorem ipsum dolor sit amet, ";
        let tokens = tokenize(text, &n, &min_length);
        assert_eq!(tokens, baseline);
        let baseline: HashSet<Vec<u8>> = HashSet::from_iter(vec![
            vec![
                108, 111, 114, 101, 109, 32, 105, 112, 115, 117, 109, 32, 100, 111, 108, 111, 114,
                32, 115, 105, 116,
            ],
            vec![115, 105, 116, 32, 97, 109, 101, 116, 32, 32],
            vec![
                105, 112, 115, 117, 109, 32, 100, 111, 108, 111, 114, 32, 115, 105, 116, 32, 97,
                109, 101, 116,
            ],
            vec![
                100, 111, 108, 111, 114, 32, 115, 105, 116, 32, 97, 109, 101, 116, 32,
            ],
        ]);
        let n = 4;
        let min_length = 5;
        let text = "Lorem ipsum dolor sit amet, ";
        let tokens = tokenize(text, &n, &min_length);
        assert_eq!(tokens, baseline);
    }
    #[test]
    fn test_hash_function() {
        let tokens: HashSet<Vec<u8>> = HashSet::from_iter(vec![
            vec![
                108, 111, 114, 101, 109, 32, 105, 112, 115, 117, 109, 32, 100, 111, 108, 111, 114,
            ],
            vec![
                105, 112, 115, 117, 109, 32, 100, 111, 108, 111, 114, 32, 115, 105, 116,
            ],
        ]);
        let hashes: Vec<u64> = hash_tokens(tokens);
        // use hashset for better comparison
        let hashes: HashSet<u64> = HashSet::from_iter(hashes);
        let baseline: HashSet<u64> = HashSet::from_iter(vec![3201882886, 3634006389]);
        assert_eq!(hashes, baseline);
    }
    #[test]
    fn test_permute_hash_unit() {
        // set a and b for deterministic results
        let original_hashvalues: Vec<u32> = vec![3201882886, 3634006389];
        // note this computes element wise and is meant for testing
        let a: u32 = 1608637543;
        let b: u32 = 3421126067;
        let max_hash: u32 = u32::MAX;
        let py_max_hash: u32 = ((1u64 << 32) - 1) as u32;
        assert_eq!(py_max_hash, max_hash);

        let modulo_prime: u32 = u32::MAX - 4; // python implementation uses (1 << 32) - 5
        let py_modulo_prime: u32 = ((1u64 << 32) - 5) as u32;
        assert_eq!(py_modulo_prime, modulo_prime);

        // we only use one
        let new_hash: HashSet<u32> = original_hashvalues
            .iter()
            .map(|&hash| {
                let computed_hash =
                    (hash.wrapping_mul(a).wrapping_add(b) % modulo_prime) & max_hash;
                computed_hash
            })
            .collect();
        // baseline is calculated from python
        let baseline: HashSet<u32> = HashSet::from_iter(vec![2880013597, 4211939270]);
        assert_eq!(new_hash, baseline);
    }
    #[test]
    fn test_function_permute_hash() {
        let original_hashvalues: Vec<u32> = vec![3201882886, 3634006389];
        // note this computes element wise and is meant for testing
        let a: ArcArray1<u32> = ArcArray1::from(vec![3143890027, 3348747336]);
        let b: ArcArray1<u32> = ArcArray1::from(vec![2571218620, 2563451924]);
        let max_hash: u32 = u32::MAX;
        let modulo_prime: u32 = u32::MAX - 4;
        let new_hashvalues = permute_hashes(original_hashvalues, &a, &b, modulo_prime, max_hash);
        // random values generated by the permutations in python
        let baseline: HashSet<Vec<u32>> = HashSet::from_iter(vec![
            vec![603517502, 1807728068],
            vec![203138723, 2550380796],
            vec![4294967295, 4294967295],
        ]);
        let new_hashvalues: HashSet<Vec<u32>> = HashSet::from_iter(new_hashvalues);
        assert_eq!(new_hashvalues, baseline);
    }
    #[test]
    fn test_min_hash_original() {
        let original_hashvalues: Vec<u32> = vec![3201882886, 3634006389];
        let a: ArcArray1<u32> = ArcArray1::from(vec![3143890027, 3348747336]);
        let b: ArcArray1<u32> = ArcArray1::from(vec![2571218620, 2563451924]);
        let max_hash: u32 = u32::MAX;
        let modulo_prime: u32 = u32::MAX - 4;
        let result: Vec<Vec<u32>> =
            permute_hashes(original_hashvalues, &a, &b, modulo_prime, max_hash);

        // find the min of each column
        let hashvalues = find_min(result);
        let baseline: Vec<u32> = vec![203138723, 1807728068];
        assert_eq!(hashvalues, baseline);
    }
    #[test]
    fn test_fused_min_hash() {
        let original_hashvalues: Vec<u64> = vec![ 684415160,  659044179,  971394434, 2591406015, 1557223710,
        827156816, 3839636002, 1313217433,  334402827, 3601442597];
        let a: ArcArray1<u64> = ArcArray1::from(vec![2297359619001564596, 1396682528897996047]);
        let b: ArcArray1<u64> = ArcArray1::from(vec![1973689801170867271, 1819927849474927636]);
        let max_hash: u64 = (1u64 << 32) - 1;
        let modulo_prime: u64 = (1u64 << 61) - 1;
        let result = min_hash_fused(&original_hashvalues, &a, &b, modulo_prime, max_hash);
        let baseline: Vec<u64> = vec![ 307409119, 1040993984];
        assert_eq!(result, baseline);
    }
    #[test]
    fn test_byte_swap() {
        // The hashvalues are generated with the python code
        let hashvalues: Vec<u64> = vec![ 307409119, 1040993984];
        let hash_ranges: Vec<(u32, u32)> = vec![(0, 1), (1, 2)];
        let result = swap_bytes(&hashvalues, &hash_ranges);
        let baseline = vec![vec![0, 0, 0, 0, 18, 82, 176, 223], vec![0, 0, 0, 0, 62, 12, 78, 192]];
        assert_eq!(result, baseline);
        let hash_ranges: Vec<(u32, u32)> = vec![(0, 2)];
        let result = swap_bytes(&hashvalues, &hash_ranges);
        let baseline = vec![vec![0, 0, 0, 0, 18, 82, 176, 223, 0, 0, 0, 0, 62, 12, 78, 192]];
        assert_eq!(result, baseline);
    }
    #[test]
    fn test_full_functional() {
        let file = "tests/assets/sonnets.txt";
        let content = std::fs::read_to_string(file).expect("Unable to read file");
        let tokens = tokenize(&content, &3, &5);
        println!("{:?}", tokens.len());
        // from python
        assert_eq!(tokens.len(), 19775);
        let hashvalues = hash_tokens(tokens);
        let a: ArcArray1<u64> = ArcArray1::from(vec![2297359619001564596, 1396682528897996047, 1973689801170867272,
            1819927849474927637,  572192888165898362]);
        let b: ArcArray1<u64> = ArcArray1::from(vec![ 571748048327668950, 1071453510346823114, 2143071682933157236,
            1865242737500154727, 1532418594269339778]);
        let max_hash: u64 = (1u64 << 32) - 1;
        let modulo_prime: u64 = (1u64 << 61) - 1;
        let result = min_hash_fused(&hashvalues, &a, &b, modulo_prime, max_hash);
        assert_eq!(result, vec![796983, 189220, 151464, 153940, 155229]);
        let hash_ranges: Vec<(u32, u32)> = vec![(0, 4), (4, 8), (8, 12), (12, 16)];
        let result = swap_bytes(&result, &hash_ranges);
        let baseline:Vec<Vec<u8>> = vec![
            vec![0, 0, 0, 0, 0, 12, 41, 55, 0, 0, 0, 0, 0, 2, 227, 36, 0, 0, 0, 0, 0, 2, 79, 168, 0, 0, 0, 0, 0, 2, 89, 84],
            vec![0, 0, 0, 0, 0, 2, 94, 93],
            vec![],
            vec![],
        ];
        assert_eq!(result, baseline);

    }
}

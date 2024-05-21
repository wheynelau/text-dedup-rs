use std::{collections::BTreeSet, sync::Arc};
use rand::{Rng, SeedableRng};
use regex::Regex;

#[derive(Debug)]
pub struct EmbedFuncParams {
    content: String,
    idx: usize,
    num_perm: usize,
    ngram_size: usize,
    min_length: usize,
    hashranges: Vec<(usize, usize)>,
    permutations: (Vec<u32>, Vec<u32>),
    max_hash: u32,
    modulo_prime: u32,
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
pub fn tokenize(text: String, n: usize, min_length: usize) -> BTreeSet<Arc<str>> {
    let text: String = text.to_lowercase();

    let re: Regex = Regex::new(r"\W+").unwrap();

    let filtered_content: Vec<&str> = re.split(&text)
                                        .filter(|s| !s.is_empty())
                                        .collect();
    
    let tokens: BTreeSet<Arc<str>> = ngrams(filtered_content, n, min_length)
        .into_iter()
        .map(|vec| vec.join(" ")) // Join Vec<&str> into a single String
        .map(Arc::from)    // Convert each String to Arc<str>
        .collect();
    
    tokens
}

fn main() {
    let text = "But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness. No one rejects, dislikes, or avoids pleasure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally encounter consequences that are extremely painful. Nor again is there anyone who loves or pursues or desires to obtain pain of itself, because it is pain, but because occasionally circumstances occur in which toil and pain can procure him some great pleasure. To take a trivial example, which of us ever undertakes laborious physical exercise, except to obtain some advantage from it? But who has any right to find fault with a man who chooses to enjoy a pleasure that has no annoying consequences, or one who avoids a pain that produces no resultant pleasure?".to_string();
    let n = 3;
    let min_length = 5;
    let tokens = tokenize(text, n, min_length);
    println!("{:?} total number= {}", tokens, tokens.len());
}
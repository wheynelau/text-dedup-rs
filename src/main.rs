use std::error::Error;

pub mod utils;
use crate::utils::prelude::*;
mod dedup;

const MODULE_PRIME: u64 = 2u64.pow(61) - 1;

fn main() -> Result<(), Box<dyn Error>>{

    let mut args = dedup::parse_args();
    
    // Check if B is too high
    let max_b = args.num_perm / args.r;
    if args.b > max_b {
        println!("Warning: Provided B value is too high. Adjusting B from {} to {}", args.b, max_b);
        args.b = max_b; // Directly modify `args.b` if `args` is mutable
    };
    
    let hash_ranges: Vec<(i32, i32)> = utils::dedup_utils::generate_hash_rangs(args.b, args.r);

    let permutations: Permutations= utils::embed::generate_permutations(MODULE_PRIME as usize, args.num_perm);

    let reader = utils::parquet_utils::get_reader(100000, &args.parquet_path);

    if args.streaming {
        return dedup::streaming(args, reader, permutations, hash_ranges);
    }
    else {
        return dedup::default(args, reader, permutations, hash_ranges);
    }

}
use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[arg(short, long, default_value = "50")]
    pub b: u32,

    #[arg(short, long, default_value = "4")]
    pub r: u32,

    #[arg(short, long, default_value = "200")]
    pub num_perm: u32,

    #[arg(short, long, default_value = "2")]
    pub n_grams: u32,

    #[arg(short, long, default_value = "text")]
    pub main_col: String,

    #[arg(short, long)]
    pub parquet_path: String,

    #[arg(short, long, default_value = "id")]
    pub idx_col: String,

    #[arg(short, long, default_value = "5")]
    pub min_len: u32,

    #[arg(short, long, default_value = "uf_output")]
    pub uf_output: String,
}

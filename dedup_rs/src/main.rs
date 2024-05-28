use std::fs::File;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

fn shingle(text:String, k:usize) -> Vec<String> {
    // 
    let mut shingles: Vec<String> = Vec::new();
    let text: String = text.to_lowercase();
    for i in 0..text.len()-k+1 {
        shingles.push(text[i..i+k].to_string());
    }
    shingles
}
fn main() {

    let file = File::open("/home/wayne/kioxia/github/text-dedup/temp_inp_paruqet/data.parquet").unwrap();

    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    println!("Converted arrow schema is: {}", builder.schema());

    let mut reader = builder.build().unwrap();

    // Initialize a counter for the number of rows
    let mut total_rows: usize = 0;

    while let Some(record_batch_result) = reader.next() {
        match record_batch_result {
            Ok(record_batch) => {
                // Increment the total row count by the number of rows in the current batch
                total_rows += record_batch.num_rows();
            }
            Err(e) => {
                eprintln!("Error reading record batch: {:?}", e);
                break;
            }
        }
    }
    
    println!("Read {} records.", total_rows);
}
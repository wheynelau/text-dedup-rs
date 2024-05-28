use std::fs::File;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetFileArrowReader;
use parquet::file::reader::{FileReader, SerializedFileReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path to the Parquet file
    let file_path = "/home/wayne/kioxia/github/text-dedup/temp_inp_paruqet/data.parquet";

    // Open the Parquet file
    let file = File::open(file_path)?;

    // Create a Parquet file reader
    let file_reader = Arc::new(SerializedFileReader::new(file)?);

    // Create an Arrow reader
    let mut arrow_reader = ParquetFileArrowReader::new(file_reader);

    // Get the schema
    let schema = arrow_reader.get_schema()?;
    println!("Schema: {:?}", schema);

    // Read the record batches
    let record_batch_reader = arrow_reader.get_record_reader(2048)?;
    for maybe_record_batch in record_batch_reader {
        let record_batch: RecordBatch = maybe_record_batch?;
        println!("Record Batch: {:?}", record_batch);
    }

    Ok(())
}
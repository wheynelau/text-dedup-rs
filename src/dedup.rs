/// Pure rust implementation of the deduplication algorithm
/// 
/// Requires files to be in a parquet format
/// 
/// 
use arrow::{
    record_batch::RecordBatch,
    datatypes::Schema,
};
use parquet::{
    arrow::{ArrowReader, ArrowWriter, ParquetFileArrowReader},
    file::reader::SerializedFileReader,
};

use std::sync::Arc;
use std::fs::File;
use std::path::Path;
mod embed;

// The Table struct. This object will represent the data read from the
// parquet files and it will be our entry point to any value in the file
pub struct Table {
    // We mantain a copy of the RecordBatch schema to keep handy the
    // file's metadata information.
    schema: Schema,
    data: Vec<RecordBatch>,
    rows: usize,
}

impl Table {
    pub fn read_parquet<T: AsRef<Path>>(path: T, chunk_size: usize) -> Self {
        // Using the parquet Arrow reader to extract batches of data
        // from the file to keep them in memory
        let file = File::open(path).unwrap();
        let file_reader = SerializedFileReader::new(file).unwrap();
        let mut arrow_reader = ParquetFileArrowReader::new(Arc::new(file_reader));

        let schema = arrow_reader.get_schema().unwrap();
        let record_batch_reader = arrow_reader.get_record_reader(chunk_size).unwrap();
        let mut data: Vec<RecordBatch> = Vec::new();

        let mut rows = 0;
        for maybe_batch in record_batch_reader {
            let record_batch = maybe_batch.unwrap();
            rows += record_batch.num_rows();

            data.push(record_batch);
        }

        Self { schema, data, rows }
    }

    // Simple writer to store the table data into a parquet file
    pub fn to_parquet<T: AsRef<Path>>(&self, path: T) {
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::new(self.schema.clone()), None).unwrap();

        for batch in self.data.iter() {
            writer.write(&batch).unwrap();
        }

        writer.close().unwrap();
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn data(&self) -> &Vec<RecordBatch> {
        &self.data
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
}

fn main() {
    let table = Table::read_parquet("data/olympics.parquet", 2000);
    println!("Number of rows: {}", table.rows())
}


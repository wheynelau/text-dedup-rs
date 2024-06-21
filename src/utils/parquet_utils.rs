use std::{fs::File, path::Path};

use arrow::array::{Int64Array, RecordBatch, StringArray};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};

pub fn process_batch(batch: &RecordBatch, main_col: &str, idx_col: &str) -> (StringArray, Int64Array) {
    let text_col = batch.column(batch.schema().index_of(main_col).unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let id_col = batch.column(batch.schema().index_of(idx_col).unwrap())
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    (text_col.clone(), id_col.clone())
}

pub fn get_reader(batch_size: usize, path: &str) -> ParquetRecordBatchReader {
    let path = Path::new(path);
    // check if file exists
    if !path.exists() {
        panic!("File does not exist");
    }
    let file = File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();

    builder.with_row_groups(vec![0])
            .with_batch_size(batch_size)
            .build()
            .unwrap()
}
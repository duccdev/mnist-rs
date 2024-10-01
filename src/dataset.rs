use crate::types::vector::Vector;
use flate2::read::GzDecoder;
use std::error::Error;
use std::fs::File;
use std::io::Read;

type Dataset = (Vec<Vector>, Vector, Vec<Vector>, Vector);

const TRAIN_SIZE: usize = 60000;
const X_ENTRY_SIZE: usize = 784;
const TEST_SIZE: usize = 10000;

pub fn load_from_binary(bytes: Vec<u8>) -> Dataset {
  let mut train_x: Vec<Vector> = vec![];
  let mut train_y: Vec<f32> = vec![];
  let mut test_x: Vec<Vector> = vec![];
  let mut test_y: Vec<f32> = vec![];

  let train_x_offset = 0;
  let train_y_offset = train_x_offset + (TRAIN_SIZE * X_ENTRY_SIZE);
  let test_x_offset = train_y_offset + (TRAIN_SIZE);
  let test_y_offset = test_x_offset + (TEST_SIZE * X_ENTRY_SIZE);

  for i in 0..TRAIN_SIZE {
    let start = train_x_offset + (i * X_ENTRY_SIZE);
    let end = start + X_ENTRY_SIZE;
    let vector_data = &bytes[start..end];

    let normalized_data: Vec<f32> = vector_data.iter().map(|&b| b as f32 / 255.0).collect();
    train_x.push(Vector::new(Some(normalized_data), None).unwrap());
  }

  for i in 0..TRAIN_SIZE {
    let y_value = bytes[train_y_offset + i] as f32;
    train_y.push(y_value);
  }

  for i in 0..TEST_SIZE {
    let start = test_x_offset + (i * X_ENTRY_SIZE);
    let end = start + X_ENTRY_SIZE;
    let vector_data = &bytes[start..end];

    let normalized_data: Vec<f32> = vector_data.iter().map(|&b| b as f32 / 255.0).collect();
    test_x.push(Vector::new(Some(normalized_data), None).unwrap());
  }

  for i in 0..TEST_SIZE {
    let y_value = bytes[test_y_offset + i] as f32;
    test_y.push(y_value);
  }

  (
    train_x,
    Vector::new(Some(train_y), None).unwrap(),
    test_x,
    Vector::new(Some(test_y), None).unwrap(),
  )
}

pub fn load_from_gzip(path: &str) -> Result<Dataset, Box<dyn Error>> {
  let file = File::open(path)?;
  let mut decoder = GzDecoder::new(file);
  let mut buffer = Vec::new();
  decoder.read_to_end(&mut buffer)?;

  Ok(load_from_binary(buffer))
}

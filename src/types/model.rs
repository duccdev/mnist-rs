use super::vector::Vector;
use crate::constants::*;
use serde::{Deserialize, Serialize};

pub struct Model {
  weights: Vec<Vector>,
  bias: Vector,
  epoch: usize,
}

impl Model {
  pub fn new() -> Self {
    let mut weights: Vec<Vector> = Vec::new();

    for _ in 0..OUTPUT_SIZE {
      weights.push(Vector::new(Some(vec![0.0; INPUT_SIZE]), None).unwrap());
    }

    Self {
      weights,
      bias: Vector::new(None, Some(OUTPUT_SIZE)).unwrap(),
      epoch: 0,
    }
  }

  fn softmax(x: Vector) -> Vector {
    x.exp().divide_scalar(x.exp().sum())
  }

  pub fn predict(&self, x: Vector) -> Result<Vector, String> {
    if x.len() != INPUT_SIZE {
      return Err("x must be of len 784 (28 * 28)".to_owned());
    }

    let mut result: Vec<f32> = vec![];

    for weights in &self.weights {
      result.push(weights.dot(&x)?);
    }

    Ok(Self::softmax(
      Vector::new(Some(result), None)?.add(&self.bias)?,
    ))
  }

  pub fn weights(&self) -> &Vec<Vector> {
    &self.weights
  }

  pub fn bias(&self) -> &Vector {
    &self.bias
  }

  pub fn epoch(&self) -> usize {
    self.epoch
  }
}

#[derive(Deserialize, Serialize)]
pub struct DeSerializableModel {
  weights: Vec<Vec<f32>>,
  bias: Vec<f32>,
  epoch: usize,
}

impl DeSerializableModel {
  pub fn from_model(model: Model) -> Self {
    let mut weights: Vec<Vec<f32>> = vec![];

    for vector in model.weights() {
      weights.push(vector.to_vec().clone());
    }

    Self {
      weights,
      bias: model.bias().to_vec().clone(),
      epoch: model.epoch().clone(),
    }
  }

  pub fn to_model(&self) -> Model {
    let mut weights: Vec<Vector> = vec![];

    for w_vec in self.weights.clone() {
      weights.push(Vector::new(Some(w_vec), None).unwrap());
    }

    Model {
      weights,
      bias: Vector::new(Some(self.bias.clone()), None).unwrap(),
      epoch: self.epoch,
    }
  }
}

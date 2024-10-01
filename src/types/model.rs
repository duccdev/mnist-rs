use super::vector::Vector;
use crate::constants::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{error::Error, fs};

pub struct Model {
  weights: Vec<Vector>,
  bias: Vector,
  epoch: usize,
}

impl Model {
  pub fn new() -> Self {
    let mut rng = rand::thread_rng();
    let mut weights: Vec<Vector> = Vec::new();

    for _ in 0..OUTPUT_SIZE {
      let mut data: Vec<f32> = vec![];

      for _ in 0..INPUT_SIZE {
        data.push(rng.gen());
      }

      weights.push(Vector::new(Some(data), None).unwrap());
    }

    Self {
      weights,
      bias: Vector::new(None, Some(OUTPUT_SIZE)).unwrap(),
      epoch: 0,
    }
  }

  #[inline]
  fn softmax(x: Vector) -> Vector {
    x.exp().divide_scalar(x.exp().sum())
  }

  pub fn predict(&self, x: &Vector) -> Result<Vector, String> {
    let mut result: Vec<f32> = vec![];

    for weights in &self.weights {
      result.push(weights.dot(&x)?);
    }

    Ok(Self::softmax(
      Vector::new(Some(result), None)?.add(&self.bias)?,
    ))
  }

  pub fn loss(&self, y_true: &Vector, y_pred: &Vector) -> Result<f32, String> {
    let y_pred_clipped = y_pred
      .to_vec()
      .iter()
      .map(|&p| p.clamp(1e-12, 1.0))
      .collect::<Vec<f32>>();

    let mut sum_loss = 0.0;
    for i in 0..y_true.len() {
      sum_loss += -y_true.get(i) * y_pred_clipped[i].ln();
    }

    Ok(sum_loss / y_true.len() as f32)
  }

  pub fn loss_deriv(&self, y_true: &Vector, y_pred: &Vector) -> Result<Vector, String> {
    let y_pred_clipped = y_pred
      .to_vec()
      .iter()
      .map(|&p| p.clamp(1e-12, 1.0))
      .collect::<Vec<f32>>();

    let mut deriv = vec![0.0; y_true.len()];
    for i in 0..y_true.len() {
      deriv[i] = y_pred_clipped[i] - y_true.get(i);
    }

    Vector::new(Some(deriv), None)
  }

  pub fn evaluate(&self, test_x: Vec<Vector>, test_y: Vec<Vector>) -> Result<(f32, f32), String> {
    let mut total_loss = 0.0;
    let mut correct_predictions = 0.0;

    for (x, y) in test_x.iter().zip(test_y.iter()) {
      let output = self.predict(x)?;

      let loss = self.loss(y, &output)?;
      total_loss += loss;

      let predicted_class = output
        .to_vec()
        .iter()
        .cloned()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0;

      let true_class = y
        .to_vec()
        .iter()
        .cloned()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0;

      if predicted_class == true_class {
        correct_predictions += 1.0;
      }
    }

    let avg_loss = total_loss / test_x.len() as f32;
    let accuracy = correct_predictions / test_x.len() as f32;

    Ok((avg_loss, accuracy))
  }

  pub fn train(
    &mut self,
    train_x: Vec<Vector>,
    train_y: Vec<Vector>,
    epochs: usize,
    learning_rate: f32,
  ) -> Result<(), String> {
    for epoch in 0..epochs {
      let mut total_loss = 0.0;
      let mut accuracy = 0.0;

      for (x, y) in train_x.iter().zip(train_y.iter()) {
        let pred = self.predict(x)?;

        let loss = self.loss(y, &pred)?;
        total_loss += loss;

        let y_true = y.argmax();
        let y_pred = pred.argmax();

        if y_true == y_pred {
          accuracy += 1.0;
        }

        let grad = self.loss_deriv(y, &pred)?;

        for (i, weight) in self.weights.iter_mut().enumerate() {
          let grad_w = x
            .multiply_scalar(grad.get(i))
            .to_vec()
            .iter()
            .map(|&g| g.clamp(-1.0, 1.0))
            .collect::<Vec<f32>>();

          *weight =
            weight.subtract(&Vector::new(Some(grad_w), None)?.multiply_scalar(learning_rate))?;
        }

        let grad_b = grad
          .to_vec()
          .iter()
          .map(|&g| g.clamp(-1.0, 1.0))
          .collect::<Vec<f32>>();

        self.bias = self
          .bias
          .subtract(&Vector::new(Some(grad_b), None)?.multiply_scalar(learning_rate))?;
      }

      total_loss /= train_x.len() as f32;

      if accuracy > 0.0 {
        accuracy /= train_x.len() as f32;
      }

      self.epoch += 1;
      println!(
        "epoch: {} | loss: {} | accuracy: {}",
        self.epoch, total_loss, accuracy
      );

      if epoch >= epochs - 1 {
        println!("stopping training");
        break;
      }
    }

    Ok(())
  }

  #[inline]
  pub fn weights(&self) -> &Vec<Vector> {
    &self.weights
  }

  #[inline]
  pub fn bias(&self) -> &Vector {
    &self.bias
  }

  #[inline]
  pub fn epoch(&self) -> usize {
    self.epoch
  }

  #[inline]
  pub fn load_from_file(path: &str) -> Result<Self, Box<dyn Error>> {
    let ckpt = fs::read_to_string(path)?;
    let ckpt: DeSerializableModel = serde_json::from_str(&ckpt)?;
    Ok(ckpt.to_model())
  }

  #[inline]
  pub fn load_from_string(ckpt: String) -> Result<Self, Box<dyn Error>> {
    let ckpt: DeSerializableModel = serde_json::from_str(&ckpt)?;
    Ok(ckpt.to_model())
  }

  #[inline]
  pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn Error>> {
    Ok(fs::write(
      path,
      serde_json::to_string(&DeSerializableModel::from_model(self))?,
    )?)
  }

  #[inline]
  pub fn dump_ckpt(&self) -> Result<String, Box<dyn Error>> {
    Ok(serde_json::to_string(&DeSerializableModel::from_model(
      self,
    ))?)
  }
}

#[derive(Deserialize, Serialize)]
pub struct DeSerializableModel {
  weights: Vec<Vec<f32>>,
  bias: Vec<f32>,
  epoch: usize,
}

impl DeSerializableModel {
  pub fn from_model(model: &Model) -> Self {
    let mut weights: Vec<Vec<f32>> = vec![];

    for vector in model.weights() {
      weights.push(vector.to_vec().clone());
    }

    Self {
      weights,
      bias: model.bias().to_vec().clone(),
      epoch: model.epoch(),
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

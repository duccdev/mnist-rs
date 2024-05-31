use super::vector::Vector;
use serde::{Deserialize, Serialize};

pub struct Model {
  layer_1: Vector,
  layer_2: Vector,
  bias_1: Vector,
  bias_2: Vector,
}

impl Model {
  pub fn new() -> Self {
    Self {
      layer_1: Vector::new(None, Some(28 * 28)).unwrap(),
      layer_2: Vector::new(None, Some(28 * 28)).unwrap(),
      bias_1: Vector::new(None, Some(28 * 28)).unwrap(),
      bias_2: Vector::new(None, Some(28 * 28)).unwrap(),
    }
  }

  pub fn train(
    &mut self,
    x_train: Vec<Vector>,
    y_train: Vec<u8>,
    epochs: Option<usize>,
    stop_at_loss: Option<f32>,
    learning_rate: Option<f32>,
  ) {
    todo!();
  }

  pub fn layer_1(&self) -> &Vector {
    &self.layer_1
  }

  pub fn layer_2(&self) -> &Vector {
    &self.layer_2
  }

  pub fn bias_1(&self) -> &Vector {
    &self.bias_1
  }

  pub fn bias_2(&self) -> &Vector {
    &self.bias_2
  }
}

#[derive(Deserialize, Serialize)]
pub struct DeSerializableModel {
  layer_1: Vec<f32>,
  layer_2: Vec<f32>,
  bias_1: Vec<f32>,
  bias_2: Vec<f32>,
}

impl DeSerializableModel {
  pub fn from_model(model: Model) -> Self {
    Self {
      layer_1: model.layer_1().to_vec().clone(),
      layer_2: model.layer_2().to_vec().clone(),
      bias_1: model.bias_1().to_vec().clone(),
      bias_2: model.bias_2().to_vec().clone(),
    }
  }

  pub fn to_model(&self) -> Model {
    Model {
      layer_1: Vector::new(Some(self.layer_1.clone()), None).unwrap(),
      layer_2: Vector::new(Some(self.layer_2.clone()), None).unwrap(),
      bias_1: Vector::new(Some(self.bias_1.clone()), None).unwrap(),
      bias_2: Vector::new(Some(self.bias_2.clone()), None).unwrap(),
    }
  }
}

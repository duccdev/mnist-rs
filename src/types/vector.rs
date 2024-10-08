pub struct Vector {
  data: Vec<f32>,
  size: usize,
}

impl Vector {
  pub fn new(data: Option<Vec<f32>>, size: Option<usize>) -> Result<Vector, String> {
    if data.is_none() && size.is_none() {
      return Err("Either data or size must be provided".to_owned());
    }

    if data.is_some() && size.is_some() {
      let data = data.unwrap();
      let size = size.unwrap();

      if data.len() != size {
        return Err("Data and size must be the same length".to_owned());
      }

      return Ok(Vector { data, size });
    }

    if data.is_some() {
      let data = data.unwrap();
      return Ok(Vector {
        data: data.clone(),
        size: data.len(),
      });
    }

    let size = size.unwrap();
    Ok(Vector {
      data: vec![0.0; size],
      size,
    })
  }

  pub fn multiply(&self, other: &Vector) -> Result<Vector, String> {
    if self.size != other.size {
      return Err("Vectors must be the same size".to_owned());
    }

    let mut data = vec![0.0; self.size];

    for i in 0..self.size {
      data[i] = self.data[i] * other.data[i];
    }

    Ok(Vector {
      data,
      size: self.size,
    })
  }

  pub fn multiply_scalar(&self, scalar: f32) -> Vector {
    let mut data = vec![0.0; self.size];

    for i in 0..self.size {
      data[i] = self.data[i] * scalar;
    }

    Vector {
      data,
      size: self.size,
    }
  }

  pub fn divide(&self, other: &Vector) -> Result<Vector, String> {
    if self.size != other.size {
      return Err("Vectors must be the same size".to_owned());
    }

    let mut data = vec![0.0; self.size];

    for i in 0..self.size {
      data[i] = self.data[i] / other.data[i];
    }

    Ok(Vector {
      data,
      size: self.size,
    })
  }

  pub fn divide_scalar(&self, scalar: f32) -> Vector {
    let mut data = vec![0.0; self.size];

    for i in 0..self.size {
      data[i] = self.data[i] / scalar;
    }

    Vector {
      data,
      size: self.size,
    }
  }

  pub fn add(&self, other: &Vector) -> Result<Vector, String> {
    if self.size != other.size {
      return Err("Vectors must be the same size".to_owned());
    }

    let mut data = vec![0.0; self.size];

    for i in 0..self.size {
      data[i] = self.data[i] + other.data[i];
    }

    Ok(Vector {
      data,
      size: self.size,
    })
  }

  pub fn add_scalar(&self, scalar: f32) -> Vector {
    let mut data = vec![0.0; self.size];

    for i in 0..self.size {
      data[i] = self.data[i] + scalar;
    }

    Vector {
      data,
      size: self.size,
    }
  }

  pub fn subtract(&self, other: &Vector) -> Result<Vector, String> {
    if self.size != other.size {
      return Err("Vectors must be the same size".to_owned());
    }

    let mut data = vec![0.0; self.size];

    for i in 0..self.size {
      data[i] = self.data[i] - other.data[i];
    }

    Ok(Vector {
      data,
      size: self.size,
    })
  }

  pub fn subtract_scalar(&self, scalar: f32) -> Vector {
    let mut data = vec![0.0; self.size];

    for i in 0..self.size {
      data[i] = self.data[i] - scalar;
    }

    Vector {
      data,
      size: self.size,
    }
  }

  pub fn dot(&self, other: &Vector) -> Result<f32, String> {
    if self.size != other.size {
      return Err("Vectors must be the same size".to_owned());
    }

    let mut sum: f32 = 0.0;

    for i in 0..self.size {
      sum += self.data[i] * other.data[i];
    }

    Ok(sum)
  }

  #[inline]
  pub fn exp(&self) -> Vector {
    Vector {
      data: self.data.iter().map(|x| x.exp()).collect(),
      size: self.size,
    }
  }

  pub fn sum(&self) -> f32 {
    let mut sum: f32 = 0.0;

    for value in &self.data {
      sum += value;
    }

    sum
  }

  pub fn argmax(&self) -> usize {
    let mut max_index = 0;
    let mut max_value = self.data[0];

    for (i, &value) in self.data.iter().enumerate() {
      if value > max_value {
        max_value = value;
        max_index = i;
      }
    }

    max_index
  }

  #[inline]
  pub fn set(&mut self, index: usize, value: f32) {
    self.data[index] = value;
  }

  #[inline]
  pub fn get(&self, index: usize) -> f32 {
    self.data[index]
  }

  #[inline]
  pub fn len(&self) -> usize {
    self.size
  }

  #[inline]
  pub fn is_empty(&self) -> bool {
    false
  }

  #[inline]
  pub fn to_vec(&self) -> &Vec<f32> {
    &self.data
  }
}

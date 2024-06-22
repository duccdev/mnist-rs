pub struct Vector {
  data: Vec<f32>,
  size: usize,
}

impl Vector {
  pub fn new(data: Option<Vec<f32>>, size: Option<usize>) -> Result<Vector, String> {
    if data.is_none() && size.is_none() {
      return Err("Either data or size must be provided".to_string());
    }

    if data.is_some() && size.is_some() {
      let data = data.unwrap();
      let size = size.unwrap();

      if data.len() != size {
        return Err("Data and size must be the same length".to_string());
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
      return Err("Vectors must be the same size".to_string());
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

  pub fn add(&self, other: &Vector) -> Result<Vector, String> {
    if self.size != other.size {
      return Err("Vectors must be the same size".to_string());
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

  pub fn subtract(&self, other: &Vector) -> Result<Vector, String> {
    if self.size != other.size {
      return Err("Vectors must be the same size".to_string());
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

  pub fn dot(&self, other: &Vector) -> Result<f32, String> {
    if self.size != other.size {
      return Err("Vectors must be the same size".to_string());
    }

    let mut sum = 0.0;

    for i in 0..self.size {
      sum += self.data[i] * other.data[i];
    }

    Ok(sum)
  }

  pub fn set(&mut self, index: usize, value: f32) {
    self.data[index] = value;
  }

  pub fn size(&self) -> usize {
    self.size
  }

  pub fn to_vec(&self) -> &Vec<f32> {
    &self.data
  }
}

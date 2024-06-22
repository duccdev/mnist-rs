pub struct XorShift32 {
  state: u32,
}

impl XorShift32 {
  fn new(seed: u32) -> Self {
    Self { state: seed }
  }

  pub fn next(&mut self) -> u32 {
    let mut x = self.state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    self.state = x;
    x
  }

  pub fn next_float(&mut self) -> f32 {
    let rand_int = self.next();
    let rand_float = rand_int as f32 / std::u32::MAX as f32;
    rand_float
  }

  pub fn next_float_range(&mut self, min: f32, max: f32) -> f32 {
    let rand_float = self.next_float();
    rand_float * (max - min) + min
  }
}

//! WGPU plan descriptors.

use num_complex::Complex32;

/// Metadata-preserving WGPU plan descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CztWgpuPlan {
    input_len: usize,
    output_len: usize,
    a_re: u32,
    a_im: u32,
    w_re: u32,
    w_im: u32,
}

impl CztWgpuPlan {
    /// Create a WGPU plan descriptor carrying the full spiral parameters.
    #[must_use]
    pub const fn new(input_len: usize, output_len: usize, a: [u32; 2], w: [u32; 2]) -> Self {
        Self {
            input_len,
            output_len,
            a_re: a[0],
            a_im: a[1],
            w_re: w[0],
            w_im: w[1],
        }
    }

    /// Return the logical input length carried by this descriptor.
    #[must_use]
    pub const fn input_len(self) -> usize {
        self.input_len
    }

    /// Return the logical output length carried by this descriptor.
    #[must_use]
    pub const fn output_len(self) -> usize {
        self.output_len
    }

    /// Return the starting spiral point.
    #[must_use]
    pub fn a(self) -> Complex32 {
        Complex32::new(f32::from_bits(self.a_re), f32::from_bits(self.a_im))
    }

    /// Return the spiral step ratio.
    #[must_use]
    pub fn w(self) -> Complex32 {
        Complex32::new(f32::from_bits(self.w_re), f32::from_bits(self.w_im))
    }

    /// Return whether the descriptor carries zero length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.input_len == 0 || self.output_len == 0
    }
}

//! 1D Fractional Fourier Transform Plan

pub mod plan;

#[cfg(test)]
mod tests;

pub use plan::{frft, frft_leto, frft_leto_typed, FrftPlan};

//! Mnemosyne-backed Leto output construction.

mod dense;
mod vector;

pub use dense::{try_dense_from_array, try_dense_from_slice, try_dense_from_view};
pub use vector::{try_array1_from_slice, try_array1_from_vec};

//! Shared Leto host-array boundaries for Apollo transforms.
//!
//! The crate owns the only workspace-wide conversion path between borrowed Leto
//! views and slice-oriented kernels. Contiguous views remain borrowed; strided
//! views are materialized exactly once in Leto logical row-major order.

mod array;
mod view;

pub use array::{
    try_array1_from_slice, try_array1_from_vec, try_dense_from_array, try_dense_from_slice,
    try_dense_from_view,
};
pub use view::view_cow;

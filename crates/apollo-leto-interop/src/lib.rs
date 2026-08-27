#![doc = include_str!("../README.md")]
#![deny(missing_docs)]

mod array;
mod view;

pub use array::{
    try_array1_from_slice, try_array1_from_vec, try_dense_from_array, try_dense_from_slice,
    try_dense_from_view,
};
pub use view::view_cow;

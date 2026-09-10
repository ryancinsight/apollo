//! Unit coverage for the batched four-step components.
//!
//! The transpose and the batched stage set are verified separately from the
//! assembled transform, so a failure localizes.

mod cache;
mod fold;
mod oracle;
mod sweep;
mod transform;
mod transpose;

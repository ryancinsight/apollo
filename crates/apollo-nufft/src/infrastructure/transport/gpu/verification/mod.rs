//! NUFFT value-semantic verification grouped by operation contract.
//!
//! The direct Type-1/Type-2 pair satisfies the adjoint law
//! `<Type1(c), f> = <c, Type2(f)>` in exact arithmetic. Fast Kaiser--Bessel
//! execution retains the owning plan's derived finite-precision bounds.
//!
//! Direct-operation leaves retain CPU differential, Leto, represented-storage,
//! and rejection contracts. Fast-operation leaves retain independently
//! evaluated gridded CPU comparisons, normalization, and diagnostic-grid
//! contracts. The adjoint statement is a proof sketch; these value-semantic
//! operator tests are empirical finite-precision evidence, not a machine-
//! checked proof.

mod direct_type1_1d;
mod direct_type1_3d;
mod direct_type2_1d;
mod direct_type2_3d;
mod fast_type1_1d;
mod fast_type1_3d;
mod fast_type2_1d;
mod fast_type2_3d;
mod metadata;
mod reusable;
mod support;

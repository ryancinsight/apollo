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

#[cfg(test)]
mod direct_type1_1d;
#[cfg(test)]
mod direct_type1_3d;
#[cfg(test)]
mod direct_type2_1d;
#[cfg(test)]
mod direct_type2_3d;
#[cfg(test)]
mod fast_type1_1d;
#[cfg(test)]
mod fast_type1_3d;
#[cfg(test)]
mod fast_type2_1d;
#[cfg(test)]
mod fast_type2_3d;
#[cfg(test)]
mod metadata;
#[cfg(test)]
mod reusable;
#[cfg(test)]
mod support;

//! NUFFT value-semantic verification grouped by operation contract.
//!
//! The direct Type-1/Type-2 pair satisfies the adjoint law
//! `<Type1(c), f> = <c, Type2(f)>` in exact arithmetic. Fast Kaiser--Bessel
//! execution retains the owning plan's derived finite-precision bounds.

#[cfg(test)]
mod device;
#[cfg(test)]
mod direct_type1_1d;
#[cfg(test)]
mod direct_type1_3d;
#[cfg(test)]
mod direct_type2_1d;
#[cfg(test)]
mod direct_type2_3d;
#[cfg(test)]
mod metadata;
#[cfg(test)]
mod reusable;
#[cfg(test)]
mod support;

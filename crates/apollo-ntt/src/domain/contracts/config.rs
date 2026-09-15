//! Default configuration mappings.

/// NTT-friendly prime 998_244_353 = 119 * 2^23 + 1. Has multiplicative order 2^23, supporting NTT lengths up to 2^23.
pub const DEFAULT_MODULUS: u64 = 998_244_353;

/// 3 is a primitive root modulo 998_244_353. Order = 998_244_352 = (p-1).
pub const DEFAULT_PRIMITIVE_ROOT: u64 = 3;

/// Mersenne prime `2³¹ − 1`, whose `p + 1 = 2³¹` makes the circle group
/// two-adic to order 30: every power-of-two length up to `2³⁰` has a
/// twin-coset domain for [`CircleNttPlan`](crate::CircleNttPlan).
pub const MERSENNE31: u64 = (1 << 31) - 1;

//! Collision-free indices for bounded process-wide cache tables.

/// Number of input lengths retained in bounded cache tables.
pub(crate) const FLAT_CACHE_LIMIT: usize = 1 << KEY_COMPONENT_BITS;
/// Number of length-and-direction entries retained in bounded cache tables.
pub(crate) const DIRECTIONAL_FLAT_CACHE_LIMIT: usize = 2 * FLAT_CACHE_LIMIT;

const KEY_COMPONENT_BITS: u32 = 12;

/// Resolve a bounded length to its direct table index.
#[inline]
pub(crate) const fn bounded_index(length: usize) -> Option<usize> {
    if length < FLAT_CACHE_LIMIT {
        Some(length)
    } else {
        None
    }
}

/// Resolve a bounded length and binary direction to a direct table index.
///
/// # Correctness
///
/// **Theorem (injectivity).** For `length < 4096` and `direction < 2`, equal
/// returned indices imply equal inputs.
///
/// **Proof.** The index `(length << 1) | direction` stores the length above the
/// least-significant bit and the direction in that bit. Right shift and bit
/// masking recover both inputs uniquely. Out-of-domain inputs return `None`. ∎
#[inline]
pub(crate) const fn bounded_directional_index(length: usize, direction: usize) -> Option<usize> {
    if length < FLAT_CACHE_LIMIT && direction < 2 {
        Some((length << 1) | direction)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{bounded_directional_index, bounded_index, FLAT_CACHE_LIMIT};

    #[test]
    fn directional_indices_preserve_each_component() {
        let base = bounded_directional_index(67, 0);

        assert_ne!(base, bounded_directional_index(68, 0));
        assert_ne!(base, bounded_directional_index(67, 1));
        assert_ne!(bounded_index(67), bounded_index(68));
    }

    #[test]
    fn direct_indices_reject_out_of_domain_components() {
        assert_eq!(bounded_index(FLAT_CACHE_LIMIT), None);
        assert_eq!(bounded_directional_index(FLAT_CACHE_LIMIT, 0), None);
        assert_eq!(bounded_directional_index(0, 2), None);
    }
}

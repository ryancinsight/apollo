//! Collision-safe storage for bounded direct-mapped caches.

use std::sync::OnceLock;

/// Number of input lengths retained in bounded cache tables.
pub(crate) const FLAT_CACHE_LIMIT: usize = 1 << KEY_COMPONENT_BITS;
/// Number of length-and-direction entries retained in bounded cache tables.
pub(crate) const DIRECTIONAL_FLAT_CACHE_LIMIT: usize = 2 * FLAT_CACHE_LIMIT;

const KEY_COMPONENT_BITS: u32 = 12;
/// A write-once cache slot that retains the key needed to validate a hit.
///
/// # Correctness
///
/// **Theorem (non-aliasing).** `get(key)` returns a value only if that value was
/// inserted with the same key.
///
/// **Proof.** The write-once cell stores `(key, value)` together. `get` compares
/// the requested key with the stored key before cloning the value, and returns
/// `None` when they differ. Therefore two keys mapped to the same slot cannot
/// observe each other's values. `insert` returns the rejected value on such a
/// collision so the caller can retain it in its sparse cache. ∎
#[repr(transparent)]
pub(crate) struct DirectMappedSlot<K, V>(OnceLock<(K, V)>);

impl<K, V> DirectMappedSlot<K, V> {
    /// Construct an empty slot without runtime initialization.
    pub(crate) const fn new() -> Self {
        Self(OnceLock::new())
    }
}

impl<K: Copy + Eq, V: Clone> DirectMappedSlot<K, V> {
    /// Clone the stored value when its key matches `key`.
    #[inline]
    pub(crate) fn get(&self, key: K) -> Option<V> {
        self.0
            .get()
            .and_then(|(stored_key, value)| (*stored_key == key).then(|| value.clone()))
    }

    /// Retain `value`, or return it when another key already owns the slot.
    #[inline]
    pub(crate) fn insert(&self, key: K, value: V) -> Result<(), V> {
        if let Some((stored_key, _)) = self.0.get() {
            return if *stored_key == key {
                Ok(())
            } else {
                Err(value)
            };
        }

        match self.0.set((key, value)) {
            Ok(()) => Ok(()),
            Err((rejected_key, rejected_value)) => match self.0.get() {
                Some((stored_key, _)) if *stored_key == rejected_key => Ok(()),
                _ => Err(rejected_value),
            },
        }
    }
}

/// Monomorphic access to collision-free and collision-validated flat slots.
///
/// A raw [`OnceLock`] is valid when the table index uniquely identifies the
/// complete key. [`DirectMappedSlot`] retains the omitted key component when
/// more than one semantic key can address the same index.
pub(crate) trait FlatCacheSlot<V> {
    /// Clone the value associated with `tag`, if initialized.
    fn get_cached(&self, tag: usize) -> Option<V>;

    /// Retain `value`, or return it when a different tag owns the slot.
    fn insert_cached(&self, tag: usize, value: V) -> Result<(), V>;
}

impl<V: Clone> FlatCacheSlot<V> for OnceLock<V> {
    #[inline]
    fn get_cached(&self, _tag: usize) -> Option<V> {
        self.get().cloned()
    }

    #[inline]
    fn insert_cached(&self, _tag: usize, value: V) -> Result<(), V> {
        match self.set(value) {
            Ok(()) => Ok(()),
            Err(rejected) => {
                // The table index is the complete key, so a racing winner is
                // semantically identical and the rejected clone is redundant.
                drop(rejected);
                Ok(())
            }
        }
    }
}

impl<V: Clone> FlatCacheSlot<V> for DirectMappedSlot<usize, V> {
    #[inline]
    fn get_cached(&self, tag: usize) -> Option<V> {
        self.get(tag)
    }

    #[inline]
    fn insert_cached(&self, tag: usize, value: V) -> Result<(), V> {
        self.insert(tag, value)
    }
}

/// Resolve two bounded key components into a table index and validation tag.
///
/// # Correctness
///
/// **Theorem (injectivity).** For components smaller than
/// [`FLAT_CACHE_LIMIT`], equal returned `(index, tag)` pairs imply equal input
/// pairs.
///
/// **Proof.** The index is `lower` and the tag is `upper`, so projection
/// recovers both components exactly. Out-of-domain components return `None`. ∎
#[inline]
pub(crate) const fn bounded_pair_coordinates(lower: usize, upper: usize) -> Option<(usize, usize)> {
    if lower < FLAT_CACHE_LIMIT && upper < FLAT_CACHE_LIMIT {
        Some((lower, upper))
    } else {
        None
    }
}

/// Resolve a bounded directional key into a table index and validation tag.
///
/// **Theorem (injectivity).** The index `(length << 1) | direction` preserves
/// the bounded length and binary direction in disjoint bit fields; the tag is
/// the generator. Therefore equal `(index, tag)` pairs imply equal keys.
#[inline]
pub(crate) const fn bounded_directional_coordinates(
    length: usize,
    direction: usize,
    generator: usize,
) -> Option<(usize, usize)> {
    if length < FLAT_CACHE_LIMIT && direction < 2 && generator < FLAT_CACHE_LIMIT {
        Some(((length << 1) | direction, generator))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bounded_directional_coordinates, bounded_pair_coordinates, DirectMappedSlot,
        FLAT_CACHE_LIMIT,
    };

    #[test]
    fn colliding_keys_never_alias_values() {
        let slot = DirectMappedSlot::new();
        let (_, first) = bounded_directional_coordinates(67, 0, 12).expect("bounded key");
        let (_, second) = bounded_directional_coordinates(67, 0, 23).expect("bounded key");

        assert_eq!(slot.insert(first, 11), Ok(()));
        assert_eq!(slot.get(first), Some(11));
        assert_eq!(slot.get(second), None);
        assert_eq!(slot.insert(second, 13), Err(13));
        assert_eq!(slot.get(first), Some(11));
        assert_eq!(slot.get(second), None);
    }

    #[test]
    fn direct_coordinates_preserve_every_semantic_component() {
        let base = bounded_directional_coordinates(67, 0, 12);

        assert_ne!(base, bounded_directional_coordinates(68, 0, 12));
        assert_ne!(base, bounded_directional_coordinates(67, 1, 12));
        assert_ne!(base, bounded_directional_coordinates(67, 0, 13));
        assert_ne!(
            bounded_pair_coordinates(67, 12),
            bounded_pair_coordinates(68, 12)
        );
        assert_ne!(
            bounded_pair_coordinates(67, 12),
            bounded_pair_coordinates(67, 13)
        );
    }

    #[test]
    fn direct_coordinates_reject_out_of_domain_components() {
        assert_eq!(bounded_pair_coordinates(FLAT_CACHE_LIMIT, 0), None);
        assert_eq!(bounded_pair_coordinates(0, FLAT_CACHE_LIMIT), None);
        assert_eq!(bounded_directional_coordinates(0, 2, 0), None);
        assert_eq!(
            bounded_directional_coordinates(0, 0, FLAT_CACHE_LIMIT),
            None
        );
    }
}

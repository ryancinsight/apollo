//! Collision-safe storage for bounded direct-mapped caches.

use std::sync::OnceLock;

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

/// Map every key component into a power-of-two table capacity.
///
/// Collisions remain possible because the key domain can exceed `CAP`; callers
/// use [`DirectMappedSlot`] to distinguish a collision from a hit.
#[inline]
pub(crate) fn flat_index<const CAP: usize, const N: usize>(components: [usize; N]) -> usize {
    debug_assert!(CAP.is_power_of_two());
    const MIX: usize = 0x9e37_79b9;
    let mut state = N.wrapping_mul(MIX);
    for component in components {
        state = (state.rotate_left(5) ^ component).wrapping_mul(MIX);
    }
    state & (CAP - 1)
}

#[cfg(test)]
mod tests {
    use super::{flat_index, DirectMappedSlot};

    #[test]
    fn colliding_keys_never_alias_values() {
        let slot = DirectMappedSlot::new();
        let first = (67, 0, 12);
        let second = (67, 0, 23);

        assert_eq!(slot.insert(first, 11), Ok(()));
        assert_eq!(slot.get(first), Some(11));
        assert_eq!(slot.get(second), None);
        assert_eq!(slot.insert(second, 13), Err(13));
        assert_eq!(slot.get(first), Some(11));
        assert_eq!(slot.get(second), None);
    }

    #[test]
    fn index_uses_each_semantic_component() {
        const CAPACITY: usize = 8192;
        let base = flat_index::<CAPACITY, 3>([67, 0, 12]);

        assert_ne!(base, flat_index::<CAPACITY, 3>([68, 0, 12]));
        assert_ne!(base, flat_index::<CAPACITY, 3>([67, 1, 12]));
        assert_ne!(base, flat_index::<CAPACITY, 3>([67, 0, 13]));
    }
}

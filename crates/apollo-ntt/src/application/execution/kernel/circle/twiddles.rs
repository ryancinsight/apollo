//! Per-level twiddle tables in one contiguous buffer.
//!
//! The circle FFT's `x` twiddles are consumed level by level, one level per
//! butterfly pass. They were one `Vec<u64>` per level, so the domain carried a
//! pointer table and a heap block per level and a transform followed one
//! pointer per pass. Flattening the levels back to back gives the same
//! `&[u64]` per level from one allocation, and the geometric lengths derive
//! their boundaries without an offset allocation. The passes still walk
//! memory in order.

/// The `x` twiddles of one transform direction, level by level.
///
/// `values` holds every level back to back. A valid table has lengths
/// `2^(l-1), 2^(l-2), …, 1`, so `values.len() + 1` is a power of two and the
/// boundaries can be derived directly from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct LevelTwiddles {
    values: Vec<u64>,
}

impl LevelTwiddles {
    /// Assemble from a filled buffer whose runs follow the circle level shape.
    ///
    /// # Panics
    ///
    /// Panics when the value count is not `2^level_count - 1`.
    pub(super) fn from_values(values: Vec<u64>) -> Self {
        let level_shape = values
            .len()
            .checked_add(1)
            .expect("invariant: a twiddle table cannot fill the address space");
        assert!(
            level_shape.is_power_of_two(),
            "invariant: twiddle values follow lengths 2^(L-1), ..., 1"
        );
        Self { values }
    }

    /// The number of levels held.
    pub(super) const fn level_count(&self) -> usize {
        (self.values.len() + 1).trailing_zeros() as usize
    }

    /// The twiddles of level `index`, in order.
    ///
    /// # Panics
    ///
    /// Panics when `index >= self.level_count()`.
    pub(super) fn level(&self, index: usize) -> &[u64] {
        let level_count = self.level_count();
        assert!(index < level_count, "twiddle level index out of bounds");
        let remaining_shape = 1usize << (level_count - index);
        let start = self.values.len() - (remaining_shape - 1);
        let length = remaining_shape / 2;
        &self.values[start..start + length]
    }

    /// Every level, coarsest last.
    pub(super) fn levels(&self) -> impl Iterator<Item = &[u64]> {
        (0..self.level_count()).map(|index| self.level(index))
    }

    /// Every level, coarsest first.
    pub(super) fn levels_reversed(&self) -> impl Iterator<Item = &[u64]> {
        (0..self.level_count()).rev().map(|index| self.level(index))
    }
}

#[cfg(test)]
mod tests {
    use super::LevelTwiddles;
    use std::mem::size_of;

    #[test]
    fn metadata_is_the_single_value_buffer() {
        assert_eq!(size_of::<LevelTwiddles>(), size_of::<Vec<u64>>());
    }

    #[test]
    fn levels_are_the_back_to_back_runs() {
        let twiddles = LevelTwiddles::from_values(vec![1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(twiddles.level_count(), 3);
        assert_eq!(twiddles.level(0), [1, 2, 3, 4]);
        assert_eq!(twiddles.level(1), [5, 6]);
        assert_eq!(twiddles.level(2), [7]);
        assert_eq!(
            twiddles.levels().collect::<Vec<_>>(),
            vec![&[1u64, 2, 3, 4][..], &[5, 6][..], &[7][..]]
        );
        assert_eq!(
            twiddles.levels_reversed().collect::<Vec<_>>(),
            vec![&[7u64][..], &[5, 6][..], &[1, 2, 3, 4][..]]
        );
    }

    #[test]
    fn an_empty_level_set_has_no_boundaries_to_store() {
        let twiddles = LevelTwiddles::from_values(Vec::new());
        assert_eq!(twiddles.level_count(), 0);
        assert_eq!(twiddles.levels().count(), 0);
        assert_eq!(twiddles.levels_reversed().count(), 0);
    }

    #[test]
    #[should_panic(expected = "invariant: twiddle values follow lengths")]
    fn a_non_geometric_level_shape_is_refused() {
        LevelTwiddles::from_values(vec![1, 2]);
    }
}

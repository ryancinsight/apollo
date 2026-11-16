//! Per-level twiddle tables in one contiguous buffer.
//!
//! The circle FFT's `x` twiddles are consumed level by level, one level per
//! butterfly pass. They were one `Vec<u64>` per level, so the domain carried a
//! pointer table and a heap block per level and a transform followed one
//! pointer per pass. Flattening the levels back to back behind an offset table
//! gives the same `&[u64]` per level from a single allocation, and the passes
//! then walk memory in order.

/// The `x` twiddles of one transform direction, level by level.
///
/// `values` holds every level back to back and `offsets` holds `level_count+1`
/// ascending indices into it, so level `l` is
/// `values[offsets[l]..offsets[l+1]]`. The offset table is `level_count`
/// `usize`s, against the per-level `Vec` headers it replaces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct LevelTwiddles {
    values: Vec<u64>,
    offsets: Vec<usize>,
}

impl LevelTwiddles {
    /// Assemble from a filled value buffer and its ascending offset table.
    ///
    /// # Panics
    ///
    /// Panics when `offsets` is empty, does not start at zero, does not end at
    /// `values.len()`, or is not non-decreasing: each is an invariant of the
    /// constructor, not a caller input.
    pub(super) fn from_parts(values: Vec<u64>, offsets: Vec<usize>) -> Self {
        assert!(
            !offsets.is_empty(),
            "invariant: a twiddle offset table carries its leading zero"
        );
        assert_eq!(
            offsets.first().copied(),
            Some(0),
            "invariant: the first twiddle level starts at offset zero"
        );
        assert_eq!(
            offsets.last().copied(),
            Some(values.len()),
            "invariant: the offset table ends at the value buffer's length"
        );
        assert!(
            offsets.windows(2).all(|pair| pair[0] <= pair[1]),
            "invariant: twiddle level offsets ascend"
        );
        Self { values, offsets }
    }

    /// The number of levels held.
    pub(super) fn level_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The twiddles of level `index`, in order.
    ///
    /// # Panics
    ///
    /// Panics when `index >= self.level_count()`.
    pub(super) fn level(&self, index: usize) -> &[u64] {
        &self.values[self.offsets[index]..self.offsets[index + 1]]
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

    #[test]
    fn levels_are_the_back_to_back_runs() {
        let twiddles = LevelTwiddles::from_parts(vec![1, 2, 3, 4, 5, 6], vec![0, 2, 5, 6]);
        assert_eq!(twiddles.level_count(), 3);
        assert_eq!(twiddles.level(0), [1, 2]);
        assert_eq!(twiddles.level(1), [3, 4, 5]);
        assert_eq!(twiddles.level(2), [6]);
        assert_eq!(
            twiddles.levels().collect::<Vec<_>>(),
            vec![&[1u64, 2][..], &[3, 4, 5][..], &[6][..]]
        );
        assert_eq!(
            twiddles.levels_reversed().collect::<Vec<_>>(),
            vec![&[6u64][..], &[3, 4, 5][..], &[1, 2][..]]
        );
    }

    #[test]
    fn an_empty_level_set_is_one_offset() {
        let twiddles = LevelTwiddles::from_parts(Vec::new(), vec![0]);
        assert_eq!(twiddles.level_count(), 0);
        assert_eq!(twiddles.levels().count(), 0);
        assert_eq!(twiddles.levels_reversed().count(), 0);
    }

    #[test]
    #[should_panic(expected = "invariant: the first twiddle level starts at offset zero")]
    fn a_table_not_starting_at_zero_is_refused() {
        LevelTwiddles::from_parts(vec![1, 2], vec![1, 2]);
    }
}

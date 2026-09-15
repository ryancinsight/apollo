//! Between a half spectrum and the full one it stands for: the lane-parallel
//! expansion that writes every bin of the full spectrum once, and the packs
//! that keep only the lower `depth` bins of each lane.
//!
//! A real field's spectrum is Hermitian: along the last axis, bin `n - k` of a
//! lane is the conjugate of bin `k` of the lane's partner (the lane at the
//! negated indices along the other axes), so the `(…, n/2 + 1)` half holds
//! all of it. The routed full-spectrum entries of the 2-D and 3-D real
//! transforms compute the half in a staging role and expand it here, into
//! caller storage or fresh capacity alike, and pack a full spectrum's lanes
//! here before the half inverses.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::plan::fft::lanes;
use core::mem::MaybeUninit;
use eunomia::Complex;

/// A place the expansion writes one bin into: an initialized bin of caller
/// storage, or a slot of fresh capacity.
pub(super) trait Slot<C> {
    fn set(&mut self, value: C);
}

impl<C> Slot<C> for C {
    #[inline]
    fn set(&mut self, value: C) {
        *self = value;
    }
}

impl<C> Slot<C> for MaybeUninit<C> {
    #[inline]
    fn set(&mut self, value: C) {
        self.write(value);
    }
}

/// Writes the full spectrum whose half is `half` into `lanes`, every slot
/// exactly once: lane `l` is lane `l` of the half (`depth` bins) followed by
/// the mirror `X[l, n - k] = conj(X[partner(l), k])` for `k = 1..n/2`, `n`
/// the full lane length.
///
/// Parallel over blocks of about [`EXPANSION_BLOCK_BYTES`] of lanes — enough
/// lanes that scheduling a task costs less than the lanes it writes, which at
/// the shortest lanes the split admits is thousands of them — with the
/// remainder past the last whole block written here. The blocks are what
/// spread a fresh spectrum's page faults over the workers.
pub(super) fn write_expanded<P, S>(
    lanes: &mut [S],
    half: &[Complex<P>],
    lane_len: usize,
    depth: usize,
    partner: impl Fn(usize) -> usize + Send + Sync,
) where
    P: MixedRadixScalar<Complex = Complex<P>>,
    S: Slot<Complex<P>> + Send + 'static,
{
    debug_assert_eq!(lanes.len() / lane_len * depth, half.len());
    let mirrored = lane_len - depth;
    let write_lane = |l: usize, lane: &mut [S]| {
        let head = &half[l * depth..(l + 1) * depth];
        let mirror = partner(l);
        let source = &half[mirror * depth + 1..=mirror * depth + mirrored];
        let (front, tail) = lane.split_at_mut(depth);
        for (slot, &bin) in front.iter_mut().zip(head) {
            slot.set(bin);
        }
        // `tail` is `conj(source[mirrored - 1])`, ..., `conj(source[0])`.
        for (slot, bin) in tail.iter_mut().zip(source.iter().rev()) {
            slot.set(Complex::new(bin.re, -bin.im));
        }
    };
    let lanes_per_block = (EXPANSION_BLOCK_BYTES / (lane_len * core::mem::size_of::<S>())).max(1);
    let block = lanes_per_block * lane_len;
    let scheduled = lanes.len() / block * block;
    let (blocks, remainder) = lanes.split_at_mut(scheduled);
    lanes::each(blocks, block, |b, lanes| {
        for (k, lane) in lanes.chunks_exact_mut(lane_len).enumerate() {
            write_lane(b * lanes_per_block + k, lane);
        }
    });
    for (k, lane) in remainder.chunks_exact_mut(lane_len).enumerate() {
        write_lane(scheduled / lane_len + k, lane);
    }
}

/// The full spectrum whose half is `half`, `lanes` lanes of `lane_len`, in
/// fresh capacity written exactly once.
pub(super) fn fresh_expanded<P>(
    half: &[Complex<P>],
    lanes: usize,
    lane_len: usize,
    depth: usize,
    partner: impl Fn(usize) -> usize + Send + Sync,
) -> Vec<Complex<P>>
where
    P: MixedRadixScalar<Complex = Complex<P>>,
{
    let len = lanes * lane_len;
    let mut full = Vec::with_capacity(len);
    write_expanded(
        &mut full.spare_capacity_mut()[..len],
        half,
        lane_len,
        depth,
        partner,
    );
    // SAFETY: `write_expanded` sets every one of the `len` slots of the spare
    // capacity it was handed — each lane's `depth` head bins and its
    // `lane_len - depth` mirror bins, over every lane — before this point,
    // and `full` held no elements before it.
    unsafe { full.set_len(len) };
    full
}

/// Bytes of lanes one expansion task writes.
const EXPANSION_BLOCK_BYTES: usize = 64 << 10;

/// Output bytes from which an owned forward takes the half route.
///
/// Unlike the caller-owned form, an owned forward pays the half's staging
/// round trip on top of the write, and a fresh spectrum's page faults, which
/// the parallel expansion spreads over the workers, are what buy that back.
/// Measured on the census host for the 2-D pair (`output/probe2d`,
/// 2026-09-15): against the widened route the owned pair ran 0.86-0.97x at
/// 1-2 MiB planes, 1.01-1.41x at 4 MiB and 1.57x at 8 MiB; the caller-owned
/// form wins at every shape and has no floor.
pub(super) const OWNED_ROUTE_BYTES: usize = 4 << 20;

/// Packs the lower `depth` bins of every `lane_len`-long lane of `full` into
/// the half at its front, in place.
///
/// Lane `l`'s packed slot ends before lane `l + 1`'s bins begin, so walking
/// the lanes upward overwrites nothing unread.
pub(super) fn pack_lanes_in_place<C: Copy>(full: &mut [C], lane_len: usize, depth: usize) {
    debug_assert!(full.len().is_multiple_of(lane_len));
    for l in 1..full.len() / lane_len {
        full.copy_within(l * lane_len..l * lane_len + depth, l * depth);
    }
}

/// Packs the lower `depth` bins of every `lane_len`-long lane of `full` into
/// the half `half`.
pub(super) fn pack_lanes<C: Copy + Send + Sync>(
    full: &[C],
    half: &mut [C],
    lane_len: usize,
    depth: usize,
) {
    lanes::paired(half, depth, full, lane_len, |halves, lanes| {
        for (packed, lane) in halves
            .chunks_exact_mut(depth)
            .zip(lanes.chunks_exact(lane_len))
        {
            packed.copy_from_slice(&lane[..depth]);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{fresh_expanded, pack_lanes, pack_lanes_in_place};
    use eunomia::Complex;

    fn half(lanes: usize, depth: usize) -> Vec<Complex<f64>> {
        (0..lanes * depth)
            .map(|k| Complex::new(k as f64, 0.5 - k as f64))
            .collect()
    }

    /// Every bin of the fresh spectrum is the half's or a mirror of it, over
    /// planes with an odd `nx`, an even one (whose Nyquist row mirrors
    /// itself) and one row, and a volume — small enough for miri to walk the
    /// `set_len` site's proof.
    #[test]
    fn the_fresh_spectrum_is_the_half_and_its_mirrors() {
        for (nx, ny) in [(5usize, 8usize), (4, 8), (1, 4), (6, 4)] {
            let depth = ny / 2 + 1;
            let half = half(nx, depth);
            let full = fresh_expanded(&half, nx, ny, depth, |i| (nx - i) % nx);
            assert_eq!(full.len(), nx * ny);
            for i in 0..nx {
                for j in 0..ny {
                    let want = if j < depth {
                        half[i * depth + j]
                    } else {
                        let bin = half[((nx - i) % nx) * depth + ny - j];
                        Complex::new(bin.re, -bin.im)
                    };
                    assert_eq!(full[i * ny + j], want, "{nx}x{ny} bin ({i}, {j})");
                }
            }
        }
        let (nx, ny, nz) = (3usize, 4usize, 8usize);
        let depth = nz / 2 + 1;
        let half = half(nx * ny, depth);
        let full = fresh_expanded(&half, nx * ny, nz, depth, |l| {
            ((nx - l / ny) % nx) * ny + (ny - l % ny) % ny
        });
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let want = if k < depth {
                        half[(i * ny + j) * depth + k]
                    } else {
                        let bin = half[(((nx - i) % nx) * ny + (ny - j) % ny) * depth + nz - k];
                        Complex::new(bin.re, -bin.im)
                    };
                    assert_eq!(full[(i * ny + j) * nz + k], want, "bin ({i}, {j}, {k})");
                }
            }
        }
    }

    /// Both packs keep exactly the lower `depth` bins of every lane.
    #[test]
    fn the_packs_keep_the_lower_bins_of_every_lane() {
        let (lanes, lane_len, depth) = (5usize, 8usize, 5usize);
        let full: Vec<Complex<f64>> = (0..lanes * lane_len)
            .map(|k| Complex::new(k as f64, -(k as f64)))
            .collect();
        let want: Vec<Complex<f64>> = (0..lanes)
            .flat_map(|l| full[l * lane_len..l * lane_len + depth].iter().copied())
            .collect();
        let mut packed = vec![Complex::default(); lanes * depth];
        pack_lanes(&full, &mut packed, lane_len, depth);
        assert_eq!(packed, want);
        let mut in_place = full.clone();
        pack_lanes_in_place(&mut in_place, lane_len, depth);
        assert_eq!(in_place[..lanes * depth], want[..]);
    }
}

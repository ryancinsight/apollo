//! The base kernel's output strategies: where each column-pass register
//! lands.
//!
//! A sink holds only shared inputs — the peer spectrum, the twiddles, the
//! even halves — and writes into the kernel's output surface, which it
//! indexes as the block itself (direct), the block's pair (combine), or its
//! four (final combine). Holding no `&mut` of its own is what lets a block
//! read its samples out of the very surface its sink later writes: the
//! reads finish in the row phase, before the first store, so the split's
//! combining blocks load the parent directly and no gather pass exists.

use super::super::cmul::cmul_chunk;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use core::mem::size_of;
use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

/// Stores `v` at chunk `chunk` of `out`.
///
/// The kernel asserts `out.len() == OUT_BLOCKS * LANES` once at entry and
/// every chunk index a sink forms is `q * LANES / LANE_COUNT + j` with
/// `q < OUT_BLOCKS` and `j < LANES / LANE_COUNT`, so the per-store bound is
/// a debug assertion rather than a checked branch in the column loop.
#[expect(
    clippy::inline_always,
    reason = "the base kernel invokes this once per SIMD chunk"
)]
#[inline(always)]
fn put<T, A>(simd: &Simd<T, A>, v: Vector<T, A>, out: &mut [T], chunk: usize)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    let offset = chunk * lanes;
    debug_assert!(
        offset + lanes <= out.len(),
        "invariant: sink chunk in bounds"
    );
    let _ = simd;
    // SAFETY: `simd` proves the host supports `A`; the kernel's entry
    // assertion on `out.len()` and the chunk arithmetic above keep
    // `offset + LANE_COUNT <= out.len()`, so the store writes a full
    // vector in bounds.
    unsafe { v.store_unaligned(out.as_mut_ptr().add(offset)) }
}

/// Chunk `chunk` of a fixed-size input.
#[expect(
    clippy::inline_always,
    reason = "the base kernel invokes this once per SIMD chunk"
)]
#[inline(always)]
fn input<T, A, const LANES: usize>(
    simd: &Simd<T, A>,
    data: &[T; LANES],
    chunk: usize,
) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    ComplexReg::from_interleaved(Vector::from_view_chunk(&simd.view(data), chunk))
}

/// The split's combining twiddles in the dup-split layout [`cmul_chunk`]
/// reads: for every register of `samples` complex samples, one chunk of
/// the real parts each duplicated over its sample's two lanes, then one of
/// the imaginary parts, so a sink's complex multiply is one `swap_adjacent`,
/// one multiply, and one `fmaddsub` where the interleaved form paid three
/// shuffles. `inner` is `W_{2 BASE}^j` for `j < BASE`; `outer` is
/// `W_{4 BASE}^j` for `j < 2 BASE`, empty when the route has no outer level.
/// The values are the process twiddle cache's, relaid, so the sinks compute
/// exactly what they did from the interleaved table.
pub(crate) struct SplitSinks<T> {
    inner: AlignedLanes<T>,
    /// `W_{3 BASE}^{2 j}` for `j < BASE`, dup-split: the radix-3 step's
    /// second twiddle; empty elsewhere.
    second: AlignedLanes<T>,
    outer: AlignedLanes<T>,
    /// `W_{8 BASE}^{j k}` for `j` in `1..8`, `k < BASE`, interleaved and
    /// chunk-major — for each register chunk of `samples` complexes the
    /// seven twiddle registers `j = 1..8` in turn — so the radix-8 pass
    /// ahead of the blocks reads one contiguous twiddle stream (RustFFT's
    /// table shape); empty elsewhere.
    rows: AlignedLanes<T>,
}

/// Lanes starting on a 64-byte boundary: a `Box<[T]>` lands at the
/// allocator's 16, so a 32-byte table load could split a cache line by
/// the heap's luck ([`crate::application::execution::kernel::components::aligned::CacheLineAligned`]); the table starts
/// at the first boundary inside its buffer instead.
struct AlignedLanes<T> {
    buffer: Box<[T]>,
    start: usize,
    len: usize,
}

impl<T: Copy> AlignedLanes<T> {
    fn empty(zero: T) -> Self {
        Self::new(&[], zero)
    }

    fn new(values: &[T], zero: T) -> Self {
        let slack = 64 / size_of::<T>();
        // `vec![x; n]` allocates exactly `n`, so the boxed slice is the
        // same allocation and keeps the boundary found in it.
        let mut buffer = vec![zero; values.len() + slack];
        let start = buffer.as_ptr().align_offset(64);
        buffer[start..start + values.len()].copy_from_slice(values);
        Self {
            buffer: buffer.into_boxed_slice(),
            start,
            len: values.len(),
        }
    }

    fn as_slice(&self) -> &[T] {
        &self.buffer[self.start..self.start + self.len]
    }
}

impl<T: MixedRadixScalar<Complex = Complex<T>>> SplitSinks<T> {
    /// The tables for a route of `n` samples over `base`-blocks with
    /// `samples` complex samples a register, from the stage-major table
    /// `twiddles` of `n` (the level of half-length `len` at `len - 1`).
    pub(crate) fn build(samples: usize, twiddles: &[Complex<T>], base: usize, n: usize) -> Self {
        let zero = T::from_precise(0.0);
        let inner = if n >= 2 * base {
            AlignedLanes::new(&dup_split(samples, &twiddles[base - 1..2 * base - 1]), zero)
        } else {
            AlignedLanes::empty(zero)
        };
        let outer = if n >= 4 * base {
            AlignedLanes::new(&interleaved(&twiddles[2 * base - 1..3 * base - 1]), zero)
        } else {
            AlignedLanes::empty(zero)
        };
        Self {
            inner,
            second: AlignedLanes::empty(zero),
            outer,
            rows: AlignedLanes::empty(zero),
        }
    }

    /// The tables for the radix-3 step over three `base`-blocks with
    /// `samples` complex samples a register: `W_{3 base}^j` and
    /// `W_{3 base}^{2 j}` for `j < base`, computed here since the
    /// stage-major cache serves powers of two only.
    pub(crate) fn build_radix3<const INVERSE: bool>(samples: usize, base: usize) -> Self {
        let zero = T::from_precise(0.0);
        let dir = if INVERSE { 1.0_f64 } else { -1.0_f64 };
        let n = 3 * base;
        let w = |j: usize| -> Complex<T> {
            let (s, c) = (dir * core::f64::consts::TAU * j as f64 / n as f64).sin_cos();
            Complex::new(T::from_precise(c), T::from_precise(s))
        };
        let first: Vec<Complex<T>> = (0..base).map(w).collect();
        let second: Vec<Complex<T>> = (0..base).map(|j| w(2 * j)).collect();
        Self {
            inner: AlignedLanes::new(&dup_split(samples, &first), zero),
            second: AlignedLanes::new(&dup_split(samples, &second), zero),
            outer: AlignedLanes::empty(zero),
            rows: AlignedLanes::empty(zero),
        }
    }

    /// The table for the radix-8 pass over eight `base`-blocks with
    /// `samples` complex samples a register: `W_{8 base}^{j k}`, `j` in
    /// `1..8`, `k < base`, chunk-major, from the stage-major table
    /// `twiddles` of `8 base` — its last level holds `W_{8 base}^m` for
    /// `m < 4 base`, and the upper half of the circle is that level negated
    /// — so the pass multiplies by exactly the cache's values.
    pub(crate) fn build_radix8(samples: usize, twiddles: &[Complex<T>], base: usize) -> Self {
        let zero = T::from_precise(0.0);
        let n = 8 * base;
        let half = n / 2;
        let level = &twiddles[half - 1..n - 1];
        let power = |m: usize| -> Complex<T> {
            let m = m % n;
            if m < half {
                level[m]
            } else {
                let w = level[m - half];
                Complex::new(-w.re, -w.im)
            }
        };
        debug_assert_eq!(base % samples, 0);
        let rows: Vec<Complex<T>> = (0..base / samples)
            .flat_map(|c| (1..8).flat_map(move |j| (0..samples).map(move |s| (j, samples * c + s))))
            .map(|(j, k)| power(j * k))
            .collect();
        Self {
            inner: AlignedLanes::empty(zero),
            second: AlignedLanes::empty(zero),
            outer: AlignedLanes::empty(zero),
            rows: AlignedLanes::new(&interleaved(&rows), zero),
        }
    }

    /// No tables: the route is one block.
    pub(crate) fn empty() -> Self {
        let zero = T::from_precise(0.0);
        Self {
            inner: AlignedLanes::empty(zero),
            second: AlignedLanes::empty(zero),
            outer: AlignedLanes::empty(zero),
            rows: AlignedLanes::empty(zero),
        }
    }

    /// `W_{3 BASE}^{2 j}`, `j < BASE`, dup-split; empty except under the
    /// radix-3 step.
    pub(crate) fn second(&self) -> &[T] {
        self.second.as_slice()
    }

    /// `W_{2 BASE}^j`, `j < BASE`, dup-split.
    pub(crate) fn inner(&self) -> &[T] {
        self.inner.as_slice()
    }

    /// `W_{4 BASE}^j`, `j < BASE`, interleaved; empty below four blocks.
    pub(crate) fn outer(&self) -> &[T] {
        self.outer.as_slice()
    }

    /// `W_{8 BASE}^{j k}` for `j` in `1..8`, `k < BASE`, interleaved and
    /// chunk-major; empty except under the radix-8 pass.
    pub(crate) fn rows(&self) -> &[T] {
        self.rows.as_slice()
    }
}

/// `w` as interleaved lanes: the outer level's table, kept compact so the
/// 1024 route's working set stays inside L1; the final sink duplicates
/// each twiddle in registers.
fn interleaved<T: Copy>(w: &[Complex<T>]) -> Vec<T> {
    w.iter().flat_map(|c| [c.re, c.im]).collect()
}

/// `w` relaid as dup-split chunk pairs of `samples` complex samples.
fn dup_split<T: Copy>(samples: usize, w: &[Complex<T>]) -> Vec<T> {
    debug_assert_eq!(w.len() % samples, 0);
    let mut lanes = Vec::with_capacity(4 * w.len());
    for chunk in w.chunks_exact(samples) {
        for c in chunk {
            lanes.extend([c.re; 2]);
        }
        for c in chunk {
            lanes.extend([c.im; 2]);
        }
    }
    lanes
}

/// `v` times dup-split twiddle chunk `chunk` of `tw`.
#[expect(
    clippy::inline_always,
    reason = "the base kernel invokes this once per SIMD chunk"
)]
#[inline(always)]
fn twiddled<T, A, const TW_LANES: usize>(
    simd: &Simd<T, A>,
    tw: &[T; TW_LANES],
    v: ComplexReg<T, A>,
    chunk: usize,
) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    cmul_chunk(&simd.view(tw), v, 2 * chunk)
}

/// Output strategy of the column pass.
pub(crate) trait StoreSink<T: LaneScalar> {
    /// Blocks of the base length the output surface spans.
    const OUT_BLOCKS: usize;

    /// Stores column register `reg`, which holds output chunk `chunk` of
    /// this block's spectrum, into `out`.
    fn store<A: SimdArch + SimdKernel<T>>(
        &mut self,
        simd: &Simd<T, A>,
        reg: ComplexReg<T, A>,
        chunk: usize,
        out: &mut [T],
    );
}

/// The spectrum lands in place: `out` is the block.
pub(crate) struct DirectSink;

impl<T: LaneScalar> StoreSink<T> for DirectSink {
    const OUT_BLOCKS: usize = 1;

    #[expect(
        clippy::inline_always,
        reason = "the base kernel invokes this concrete sink once per SIMD chunk"
    )]
    #[inline(always)]
    fn store<A: SimdArch + SimdKernel<T>>(
        &mut self,
        simd: &Simd<T, A>,
        reg: ComplexReg<T, A>,
        chunk: usize,
        out: &mut [T],
    ) {
        put(simd, reg.into_interleaved(), out, chunk);
    }
}

/// The last block of three: the radix-3 step over the three block spectra
/// as its registers leave the kernel. `sub0` and `sub1` are the other two
/// blocks' spectra; with `t1 = W_{3 BASE}^j sub1` and `t2 = W_{3 BASE}^{2 j}
/// reg`, the outputs are `sub0 + t1 + t2`, `m0 + m1`, and `m0 - m1` for
/// `m0 = sub0 - (t1 + t2) / 2` and `m1 = -+ i (sqrt(3) / 2) (t1 - t2)`, the
/// upper sign forward — the three-point butterfly the composite radix-3
/// pass runs, over three spectra of `BASE` at once.
pub(crate) struct FinalRadix3Sink<
    'a,
    T,
    const LANES: usize,
    const TW_LANES: usize,
    const INVERSE: bool,
> {
    pub(crate) sub0: &'a [T; LANES],
    pub(crate) sub1: &'a [T; LANES],
    /// `W_{3 BASE}^j` per chunk, dup-split ([`SplitSinks::inner`]).
    pub(crate) first_tw: &'a [T; TW_LANES],
    /// `W_{3 BASE}^{2 j}` per chunk, dup-split ([`SplitSinks::second`]).
    pub(crate) second_tw: &'a [T; TW_LANES],
    /// `-1 / 2`, the real part of the third root of unity.
    pub(crate) half_negative: T,
    /// `sqrt(3) / 2`, the sine of the third root of unity.
    pub(crate) sine: T,
}

impl<T: LaneScalar, const LANES: usize, const TW_LANES: usize, const INVERSE: bool> StoreSink<T>
    for FinalRadix3Sink<'_, T, LANES, TW_LANES, INVERSE>
{
    const OUT_BLOCKS: usize = 3;

    #[expect(
        clippy::inline_always,
        reason = "the base kernel invokes this concrete sink once per SIMD chunk"
    )]
    #[inline(always)]
    fn store<A: SimdArch + SimdKernel<T>>(
        &mut self,
        simd: &Simd<T, A>,
        reg: ComplexReg<T, A>,
        chunk: usize,
        out: &mut [T],
    ) {
        let block = LANES / <A as SimdStorage<T>>::LANE_COUNT;
        let sub0 = input(simd, self.sub0, chunk);
        let t1 = twiddled(simd, self.first_tw, input(simd, self.sub1, chunk), chunk);
        let t2 = twiddled(simd, self.second_tw, reg, chunk);
        let (sum, diff) = t1.butterfly(t2);
        let half_negative = simd.splat(self.half_negative);
        let sine = simd.splat(self.sine);
        let (out0, _) = sub0.butterfly(sum);
        let m0 = ComplexReg::from_interleaved(
            sum.into_interleaved()
                .mul_add(half_negative, sub0.into_interleaved()),
        );
        let turned = if INVERSE {
            diff.mul_i()
        } else {
            diff.mul_neg_i()
        };
        let m1 = ComplexReg::from_interleaved(turned.into_interleaved() * sine);
        let (out1, out2) = m0.butterfly(m1);
        put(simd, out0.into_interleaved(), out, chunk);
        put(simd, out1.into_interleaved(), out, block + chunk);
        put(simd, out2.into_interleaved(), out, 2 * block + chunk);
    }
}

/// The last block of four: the radix-4 step over the four block spectra
/// as its registers leave the kernel. `sub0`, `sub2`, and `sub1` are the
/// other three blocks' spectra; the even half is `sub0 -+ W_{2 BASE}^j
/// sub2` and the odd half `sub1 -+ W_{2 BASE}^j reg`, the outer level
/// `W_{4 BASE}^j` combining them into the four quarters of `out` at
/// `chunk`. No block writes an intermediate pair: the even half that the
/// two-level form stored and reloaded is formed here from the spectra.
/// The high half of the outer level is the low half rotated a quarter
/// turn (`W_{4 BASE}^{j + BASE} = -+ i W_{4 BASE}^j`), so one table serves
/// both and the route holds half the outer twiddles.
pub(crate) struct FinalRadix4Sink<
    'a,
    T,
    const LANES: usize,
    const TW_LANES: usize,
    const INVERSE: bool,
> {
    /// Subsequence 0's spectrum.
    pub(crate) sub0: &'a [T; LANES],
    /// Subsequence 2's spectrum, the even half's odd block.
    pub(crate) sub2: &'a [T; LANES],
    /// Subsequence 1's spectrum, the odd half's even block.
    pub(crate) sub1: &'a [T; LANES],
    /// `W_{2 BASE}^j` per chunk, dup-split.
    pub(crate) inner_tw: &'a [T; TW_LANES],
    /// `W_{4 BASE}^j` per chunk, `j < BASE`, interleaved.
    pub(crate) outer_tw: &'a [T; LANES],
}

impl<T: LaneScalar, const LANES: usize, const TW_LANES: usize, const INVERSE: bool> StoreSink<T>
    for FinalRadix4Sink<'_, T, LANES, TW_LANES, INVERSE>
{
    const OUT_BLOCKS: usize = 4;

    #[expect(
        clippy::inline_always,
        reason = "the base kernel invokes this concrete sink once per SIMD chunk"
    )]
    #[inline(always)]
    fn store<A: SimdArch + SimdKernel<T>>(
        &mut self,
        simd: &Simd<T, A>,
        reg: ComplexReg<T, A>,
        chunk: usize,
        out: &mut [T],
    ) {
        let block = LANES / <A as SimdStorage<T>>::LANE_COUNT;
        let sub0 = input(simd, self.sub0, chunk);
        let sub2 = input(simd, self.sub2, chunk);
        let (even_low, even_high) = sub0.butterfly(twiddled(simd, self.inner_tw, sub2, chunk));
        let sub1 = input(simd, self.sub1, chunk);
        let (odd_low, odd_high) = sub1.butterfly(twiddled(simd, self.inner_tw, reg, chunk));
        let outer = input(simd, self.outer_tw, chunk);
        let odd_high = if INVERSE {
            odd_high.mul_i()
        } else {
            odd_high.mul_neg_i()
        };
        let (out0, out2) = even_low.butterfly(odd_low * outer);
        let (out1, out3) = even_high.butterfly(odd_high * outer);
        put(simd, out0.into_interleaved(), out, chunk);
        put(simd, out1.into_interleaved(), out, block + chunk);
        put(simd, out2.into_interleaved(), out, 2 * block + chunk);
        put(simd, out3.into_interleaved(), out, 3 * block + chunk);
    }
}

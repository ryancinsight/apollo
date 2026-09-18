//! The base kernel's output strategies: where each column-pass register
//! lands.
//!
//! A sink holds only shared inputs — the peer spectrum, the twiddles, the
//! even halves — and writes into the kernel's output surface, which it
//! indexes as the block itself (direct), the block's pair (combine), or its
//! four (final combine). Holding no `&mut` of its own is what lets a block
//! read its samples out of the very surface its sink later writes: the
//! reads finish in the row phase, before the first store, so the split's
//! four-lane combining blocks load the parent directly; the eight-lane
//! four-block route still gathers.

use super::super::cmul::{cmul, cmul_chunk};
use crate::application::execution::kernel::components::register_butterfly::{radix3, Thirds};
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
    /// The column-first chain's levels, outermost first, each the
    /// chunk-major rows of its radix pass; empty for the sink routes.
    chain: Box<[ChainLevel<T>]>,
}

/// One level of the column-first chain: the radix of its pass over the
/// level above and the pass's twiddles `W_n^{j k}` for `j` in `1..R`,
/// `k < n / R`, chunk-major — for each register chunk of `samples`
/// complexes the `R - 1` twiddles `j = 1..R` in turn, in the level's
/// [`TwiddleLayout`] — so the pass reads one contiguous twiddle stream.
pub(crate) struct ChainLevel<T> {
    radix: usize,
    layout: TwiddleLayout,
    rows: AlignedLanes<T>,
}

impl<T> ChainLevel<T> {
    /// The radix of this level's pass.
    pub(crate) fn radix(&self) -> usize {
        self.radix
    }

    /// How the pass's twiddles are laid out.
    pub(crate) fn layout(&self) -> TwiddleLayout {
        self.layout
    }
}

impl<T: Copy> ChainLevel<T> {
    /// The pass's twiddle lanes, [`TwiddleLayout::registers`] a twiddle.
    pub(crate) fn rows(&self) -> &[T] {
        self.rows.as_slice()
    }
}

/// How a column pass holds each twiddle register.
///
/// Split, the pass multiplies an interleaved register by one swap, one
/// product and one FMA; interleaved, by two duplicates and a swap more, at
/// half the table. Measured by alternating pinned A/B (2026-09-17,
/// `APOLLO-COLUMN-SPLIT-TWIDDLES`): split took `f32` 2048 6% and `f64` 4096
/// 5% faster, and ran the chain passes 16 to 18% slower at `f32` 131072 and
/// 262144 and 10% slower at `f64` 16384 and 32768, where the doubled table
/// leaves the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TwiddleLayout {
    /// `(re, im)` per sample, one register a twiddle.
    Interleaved,
    /// The real parts duplicated, then `(-im, im)`: two registers a
    /// twiddle ([`split_signed`]).
    Split,
}

impl TwiddleLayout {
    /// The largest split table that measured faster: `f32` 16384's top
    /// level (14336 twiddles, 229 KiB split) won; `f64` 16384's (458 KiB)
    /// lost and `f32` 32768's held level.
    const SPLIT_MAX_BYTES: usize = 256 << 10;

    /// The layout for `count` twiddles of `T`.
    pub(crate) fn for_twiddles<T>(count: usize) -> Self {
        if 4 * count * core::mem::size_of::<T>() <= Self::SPLIT_MAX_BYTES {
            Self::Split
        } else {
            Self::Interleaved
        }
    }

    /// Registers a twiddle occupies.
    pub(crate) const fn registers(self) -> usize {
        match self {
            Self::Interleaved => 1,
            Self::Split => 2,
        }
    }
}

/// The chain level of radix `radix` over the twiddles `w`, `samples`
/// complexes a register, in the layout their count selects.
fn chain_level<T: Copy + core::ops::Neg<Output = T>>(
    radix: usize,
    samples: usize,
    w: &[Complex<T>],
    zero: T,
) -> ChainLevel<T> {
    let layout = TwiddleLayout::for_twiddles::<T>(w.len());
    let lanes = match layout {
        TwiddleLayout::Interleaved => interleaved(w),
        TwiddleLayout::Split => split_signed(samples, w),
    };
    ChainLevel {
        radix,
        layout,
        rows: AlignedLanes::new(&lanes, zero),
    }
}

/// The chain's radices, outermost first, for `blocks` base blocks: a chain
/// of eights over the base with at most one radix 4 outside them
/// (RustFFT's `8xn` chain with its closing `4xn`), or none where the block
/// count is not of that form (the sink routes at two, three and four).
pub(crate) fn chain_radices(blocks: usize) -> Option<Vec<usize>> {
    let mut radices = Vec::new();
    let mut rest = blocks;
    while rest % 8 == 0 {
        radices.push(8);
        rest /= 8;
    }
    if radices.is_empty() || !(rest == 1 || rest == 4) {
        return None;
    }
    if rest == 4 {
        radices.push(4);
    }
    radices.reverse();
    Some(radices)
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
            chain: Box::new([]),
        }
    }

    /// The tables for the radix-3 step over three `base`-blocks with
    /// `samples` complex samples a register: `W_{3 base}^j` and
    /// `W_{3 base}^{2 j}` for `j < base`, computed here since the
    /// stage-major cache serves powers of two only.
    pub(crate) fn build_radix3(samples: usize, base: usize) -> Self {
        let zero = T::from_precise(0.0);
        let dir = -1.0_f64;
        let n = 3 * base;
        let w = |j: usize| -> Complex<T> {
            let (s, c) = (dir * core::f64::consts::TAU * j as f64 / n as f64).sin_cos();
            Complex::new(T::from_precise(c), T::from_precise(s))
        };
        let first: Vec<Complex<T>> = (0..base).map(w).collect();
        let second: Vec<Complex<T>> = (0..base).map(|j| w(2 * j)).collect();
        // The same twiddles chunk-major for the radix-3 pass ahead of the
        // blocks: per register chunk, `W^{k}` then `W^{2 k}`.
        debug_assert_eq!(base % samples, 0);
        let rows: Vec<Complex<T>> = (0..base / samples)
            .flat_map(|c| (1..3).flat_map(move |j| (0..samples).map(move |s| (j, samples * c + s))))
            .map(|(j, k)| w(j * k))
            .collect();
        Self {
            inner: AlignedLanes::new(&dup_split(samples, &first), zero),
            second: AlignedLanes::new(&dup_split(samples, &second), zero),
            outer: AlignedLanes::empty(zero),
            chain: Box::new([chain_level(3, samples, &rows, zero)]),
        }
    }

    /// The tables for the column-first chain over `base`-blocks with
    /// `samples` complex samples a register: one level per radix of
    /// `radices` (outermost first, the innermost over the base itself),
    /// each chunk-major from the stage-major table `twiddles` of the
    /// route's length.
    pub(crate) fn build_column_first(
        samples: usize,
        twiddles: &[Complex<T>],
        base: usize,
        radices: &[usize],
    ) -> Self {
        let zero = T::from_precise(0.0);
        let n = base * radices.iter().product::<usize>();
        let mut block = n;
        let chain: Vec<ChainLevel<T>> = radices
            .iter()
            .map(|&radix| {
                block /= radix;
                chain_level(
                    radix,
                    samples,
                    &chunk_major_rows(samples, twiddles, radix, block),
                    zero,
                )
            })
            .collect();
        Self {
            inner: AlignedLanes::empty(zero),
            second: AlignedLanes::empty(zero),
            outer: AlignedLanes::empty(zero),
            chain: chain.into_boxed_slice(),
        }
    }

    /// No tables: the route is one block.
    pub(crate) fn empty() -> Self {
        let zero = T::from_precise(0.0);
        Self {
            chain: Box::new([]),
            inner: AlignedLanes::empty(zero),
            second: AlignedLanes::empty(zero),
            outer: AlignedLanes::empty(zero),
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

    /// The column-first chain's levels, outermost first; empty for the
    /// sink routes.
    pub(crate) fn chain(&self) -> &[ChainLevel<T>] {
        &self.chain
    }
}

/// `W_n^{j k}` for `j` in `1..radix`, `k < block`, `n = radix block`,
/// chunk-major for `samples` complexes a register — for each register
/// chunk the `radix - 1` twiddle registers `j = 1..radix` in turn — from
/// the stage-major table `twiddles` of a length at least `n`: its level
/// for `n` holds `W_n^m` for `m < n / 2` at `n / 2 - 1`, and the upper
/// half of the circle is that level negated, so the pass multiplies by
/// exactly the cache's values.
fn chunk_major_rows<T: MixedRadixScalar<Complex = Complex<T>>>(
    samples: usize,
    twiddles: &[Complex<T>],
    radix: usize,
    block: usize,
) -> Vec<Complex<T>> {
    let n = radix * block;
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
    debug_assert_eq!(block % samples, 0);
    (0..block / samples)
        .flat_map(|c| (1..radix).flat_map(move |j| (0..samples).map(move |s| (j, samples * c + s))))
        .map(|(j, k)| power(j * k))
        .collect()
}

/// `w` as interleaved lanes: the outer level's table, kept compact so the
/// 1024 route's working set stays inside L1; the final sink duplicates
/// each twiddle in registers.
fn interleaved<T: Copy>(w: &[Complex<T>]) -> Vec<T> {
    w.iter().flat_map(|c| [c.re, c.im]).collect()
}

/// `w` relaid per register chunk of `samples` complex samples as two
/// registers: the real parts duplicated, then the imaginary parts as
/// `(-im, im)`, so the pass multiplies an interleaved `y` by `w` as
/// `y * re + swap(y) * signed_im` with one swap and no twiddle shuffle.
fn split_signed<T: Copy + core::ops::Neg<Output = T>>(samples: usize, w: &[Complex<T>]) -> Vec<T> {
    debug_assert_eq!(w.len() % samples, 0);
    let mut lanes = Vec::with_capacity(4 * w.len());
    for chunk in w.chunks_exact(samples) {
        for c in chunk {
            lanes.extend([c.re; 2]);
        }
        for c in chunk {
            lanes.extend([-c.im, c.im]);
        }
    }
    lanes
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

/// `v` times dup-split twiddle chunk `chunk` of `tw`, a forward twiddle,
/// conjugated on the inverse: the sinks keep the forward tables only.
#[expect(
    clippy::inline_always,
    reason = "the base kernel invokes this once per SIMD chunk"
)]
#[inline(always)]
fn twiddled<T, A, const TW_LANES: usize, const INVERSE: bool>(
    simd: &Simd<T, A>,
    tw: &[T; TW_LANES],
    v: ComplexReg<T, A>,
    chunk: usize,
) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    cmul_chunk::<_, _, _, _, _, INVERSE>(&simd.view(tw), v, 2 * chunk)
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
}

impl<
        T: LaneScalar + MixedRadixScalar,
        const LANES: usize,
        const TW_LANES: usize,
        const INVERSE: bool,
    > StoreSink<T> for FinalRadix3Sink<'_, T, LANES, TW_LANES, INVERSE>
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
        let t1 = twiddled::<T, A, TW_LANES, INVERSE>(
            simd,
            self.first_tw,
            input(simd, self.sub1, chunk),
            chunk,
        );
        let t2 = twiddled::<T, A, TW_LANES, INVERSE>(simd, self.second_tw, reg, chunk);
        let [out0, out1, out2] = radix3::<T, A, INVERSE>([sub0, t1, t2], &Thirds::new(*simd));
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
        let (even_low, even_high) = sub0.butterfly(twiddled::<T, A, TW_LANES, INVERSE>(
            simd,
            self.inner_tw,
            sub2,
            chunk,
        ));
        let sub1 = input(simd, self.sub1, chunk);
        let (odd_low, odd_high) = sub1.butterfly(twiddled::<T, A, TW_LANES, INVERSE>(
            simd,
            self.inner_tw,
            reg,
            chunk,
        ));
        let outer = input(simd, self.outer_tw, chunk);
        let odd_high = if INVERSE {
            odd_high.mul_i()
        } else {
            odd_high.mul_neg_i()
        };
        let (out0, out2) = even_low.butterfly(cmul::<T, A, INVERSE>(odd_low, outer));
        let (out1, out3) = even_high.butterfly(cmul::<T, A, INVERSE>(odd_high, outer));
        put(simd, out0.into_interleaved(), out, chunk);
        put(simd, out1.into_interleaved(), out, block + chunk);
        put(simd, out2.into_interleaved(), out, 2 * block + chunk);
        put(simd, out3.into_interleaved(), out, 3 * block + chunk);
    }
}

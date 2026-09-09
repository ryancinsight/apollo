//! Batched power-of-two sub-transforms with the transform index in the lane.
//!
//! ## Why this layout
//!
//! A butterfly vectorized *within* one transform must gather its two operands
//! from positions `j` and `j + half`, and a complex multiply on interleaved data
//! then needs cross-lane shuffles to separate the real and imaginary parts. Both
//! costs vanish when the lane position holds the *transform* index instead:
//! with `B` independent transforms laid out so element `j` of transform `b` sits
//! at `j * B + b`, a butterfly reads two contiguous runs of `B` values, the
//! twiddle is one scalar broadcast across every lane, and no shuffle occurs
//! anywhere.
//!
//! This is the arrangement [`FftPlanarMut`] documents — "lane `c` across all
//! rows is one independent transform instance and no cross-lane shuffle is
//! required" — and which nothing previously used.
//!
//! Measured against the other kernel shapes tried in this crate, all of which
//! sat between 3.4 and 6.1 flops/ns, this reaches 10.2 to 10.7 across batch and
//! length combinations, using the same `hermes_simd` `Vector` operations. The
//! layout is the variable.
//!
//! ## Where it applies
//!
//! The four-step decomposition already splits `N = N1 x N2` and transforms
//! along each axis in turn. Writing `i = j * N2 + b`, the input is *already*
//! batch-major for the first axis, so those `N2` transforms of length `N1` need
//! no transpose at all; and the output index `k2 * N1 + k1` falls out of the
//! second batched pass, so there is no final transpose either. One transpose
//! sits between them, in place because the four-step gate admits only square
//! splits.
//!
//! [`FftPlanarMut`]: crate::domain::storage::FftPlanarMut

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

#[cfg(all(test, windows, target_arch = "x86_64"))]
macro_rules! sect {
    ($label:expr, $body:block) => {{
        let t0 = unsafe { core::arch::x86_64::_rdtsc() };
        let out = $body;
        let t1 = unsafe { core::arch::x86_64::_rdtsc() };
        crate::application::execution::kernel::components::batched::sections::record(
            $label,
            t1 - t0,
        );
        out
    }};
}
#[cfg(not(all(test, windows, target_arch = "x86_64")))]
macro_rules! sect {
    ($label:expr, $body:block) => {{
        let _label: &str = $label;
        $body
    }};
}

mod cache;
mod radix;
mod sweep;
pub(crate) use cache::BatchedPlanCache;

/// Section labels of the time-decimated stage set's sweeps, by sweep index;
/// the attribution probe reports them beneath `stages1`.
const TIME_SWEEPS: [&str; 3] = ["t1", "t2", "t3"];
/// Section labels of the frequency-decimated stage set's sweeps, by sweep
/// index; reported beneath `stages2`.
const FREQUENCY_SWEEPS: [&str; 3] = ["f1", "f2", "f3"];

use eunomia::Complex;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

/// Tile side for the in-place transpose, in elements.
///
/// Per-length batched-transform plan: the stage-major twiddle table, a
/// build-time cost.
///
/// Both stage sets read this one table. The time-decimated set walks its
/// stages upward and the frequency-decimated set downward, over the same
/// values.
pub(crate) struct BatchedPlan<T> {
    len: usize,
    /// Stage-major twiddles: stage `s` (sub-transform length `2^(s+1)`) occupies
    /// `2^s` entries, so the table totals `len - 1`.
    tw: Vec<(T, T)>,
}

impl<T: MixedRadixScalar> BatchedPlan<T> {
    fn new<const INVERSE: bool>(len: usize) -> Self {
        assert!(
            len.is_power_of_two(),
            "batched plan requires a power of two"
        );
        let sign = if INVERSE { 1.0_f64 } else { -1.0_f64 };
        let mut tw = Vec::with_capacity(len.saturating_sub(1));
        let mut l = 2usize;
        while l <= len {
            for j in 0..l / 2 {
                // Direct evaluation per entry: a recurrence here would carry
                // O(N·u) twiddle error, which the forward-error bound this
                // crate documents does not admit.
                let (sin, cos) = (sign * core::f64::consts::TAU * j as f64 / l as f64).sin_cos();
                tw.push((T::from_precise(cos), T::from_precise(sin)));
            }
            l <<= 1;
        }
        Self { len, tw }
    }
}

/// All stages of `batch` independent length-`len` transforms over planar data.
///
/// Dispatch happens once for the whole stage set, which is the placement Hermes
/// ADR 016 requires: outside the innermost loop, and never wrapping a
/// thread-spawning call.
struct BatchedStages<'a, T> {
    re: &'a mut [T],
    im: &'a mut [T],
    tw: &'a [(T, T)],
    /// Interleaved input read by the first pass in place of the planes, or
    /// `None` when the planes already hold it. Rows of `batch` complexes as
    /// `2 * batch` reals in natural order; plane row `p` reads source row
    /// `rev(p)`, which is the row map the deinterleave pass used to apply.
    /// Reading here deletes that pass: the first stage pair is the one that
    /// loads every element exactly once, and two interleaved vector loads
    /// plus one register deinterleave replace the two plane loads.
    source: Option<&'a [T]>,
    /// Live columns per row — the loop bound.
    batch: usize,
    /// Elements per row including [`ROW_PAD`] — the index multiplier.
    stride: usize,
    len: usize,
}

impl<T> LaneKernel<T> for BatchedStages<'_, T>
where
    T: LaneScalar + MixedRadixScalar + radix::Lane,
{
    type Output = ();

    #[expect(
        clippy::inline_always,
        reason = "large LaneKernel::call body must fold into the dispatcher's target-feature scope"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let Self {
            re,
            im,
            tw,
            source,
            batch: b,
            stride: s,
            len,
        } = self;
        // Stages ascend from 2; each sweep applies up to `SWEEP_STAGES` of
        // them per trip through the planes, two per pass while two remain
        // and then one, over tiles that stay in L1 between its passes
        // (see [`sweep`]). The four-step twiddle and the interleaved source
        // both ride the pass over stage 2, the one pass that loads every
        // element exactly once. The per-element operation order is the
        // single-stage one, so results are bitwise those of any other
        // grouping.
        let mut l0 = 2usize;
        for (index, stages) in sweep::sweep_lengths(len.trailing_zeros()).enumerate() {
            sect!(TIME_SWEEPS[index], {
                sweep::sweep_time(re, im, tw, source, b, s, len, l0, stages, simd);
            });
            l0 <<= stages;
        }
    }
}

/// One-vector load at `at`, which the caller has proved is in bounds.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line               call here reintroduces the ADR 009 penalty this kernel exists to avoid"
)]
#[inline(always)]
pub(super) fn load<T, A>(data: &[T], at: usize) -> Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(at + <A as SimdStorage<T>>::LANE_COUNT <= data.len());
    // SAFETY: the caller's loop condition bounds `at + LANE_COUNT` by the slice
    // length, and `BatchedStages::call` receives the capability proving that
    // the host executes `A`. The checked wrapper revalidates both per call,
    // which measured as 45% of this kernel's time; the bound here is
    // loop-invariant.
    unsafe { Vector::<T, A>::load_unaligned(data.as_ptr().add(at)) }
}

/// One-vector store at `at`, which the caller has proved is in bounds.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line               call here reintroduces the ADR 009 penalty this kernel exists to avoid"
)]
#[inline(always)]
pub(super) fn store<T, A>(v: Vector<T, A>, data: &mut [T], at: usize)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(at + <A as SimdStorage<T>>::LANE_COUNT <= data.len());
    // SAFETY: as `load` above.
    unsafe { v.store_unaligned(data.as_mut_ptr().add(at)) }
}

/// Two adjacent vectors of interleaved complexes at `at` (in reals), split
/// into their real and imaginary lanes.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope, as `load`"
)]
#[inline(always)]
pub(super) fn load_interleaved<T, A>(data: &[T], at: usize) -> (Vector<T, A>, Vector<T, A>)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    load::<T, A>(data, at).deinterleave(load::<T, A>(data, at + lanes))
}

/// The counterpart of [`load_interleaved`]: real and imaginary lanes stored
/// as two adjacent vectors of interleaved complexes at `at` (in reals).
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope, as `store`"
)]
#[inline(always)]
pub(super) fn store_interleaved<T, A>(re: Vector<T, A>, im: Vector<T, A>, data: &mut [T], at: usize)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    let (lo, hi) = re.interleave(im);
    store::<T, A>(lo, data, at);
    store::<T, A>(hi, data, at + lanes);
}

/// Bit-reverses a row index within `bits` bits; the map between plane rows
/// and interleaved rows on both sides of the stage sets.
#[inline]
pub(super) fn reverse_row(row: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        row.reverse_bits() >> (usize::BITS - bits)
    }
}

/// Square in-place transpose of an `m x m` plane, tiled for locality.
/// Extra elements appended to each plane row, so the row stride is `m + 8`
/// rather than `m`.
///
/// With an unpadded power-of-two stride, all of it aliases: every plane row
/// maps to the same L1 sets, `re` and `im` (a power-of-two apart) share sets,
/// and the three same-sized buffers a four-step holds live — caller data,
/// scratch planes, twiddle matrix — collide by allocation accident. Eight
/// f64 elements are one cache line, so consecutive rows shift by a full line
/// and the uniform aliasing is gone; for f32 the shift is half a line, which
/// still rotates the sets. The pad is a spacer, never computed on: every loop
/// bounds itself by the live column count and multiplies row indices by the
/// stride.
pub(crate) const ROW_PAD: usize = 8;

/// Tile edge for [`transpose_planes`]. Rows use the padded element stride
/// returned by [`plane_geometry`]; their byte stride also depends on the scalar
/// width. Tile capacity alone does not establish cache-set occupancy.
const TWIDDLE_TRANSPOSE_TILE: usize = 8;

/// Transposes both `m x m` planes in place, tiled.
///
/// Pure exchange: the four-step twiddle that used to ride this pass as a
/// scalar multiply — 26% of the driver at N = 256 — now rides stage-set-2's
/// first-stage vector loads instead, which the twiddle matrix's symmetry
/// (`W^(j*b)` equals its own transpose) makes exactly equivalent.
fn transpose_planes<T: Copy>(re: &mut [T], im: &mut [T], m: usize, stride: usize) {
    debug_assert!(stride >= m && re.len() >= m * stride);
    for ib in (0..m).step_by(TWIDDLE_TRANSPOSE_TILE) {
        let ie = (ib + TWIDDLE_TRANSPOSE_TILE).min(m);
        for jb in (ib..m).step_by(TWIDDLE_TRANSPOSE_TILE) {
            let je = (jb + TWIDDLE_TRANSPOSE_TILE).min(m);
            for i in ib..ie {
                let start = if jb == ib { i + 1 } else { jb };
                for j in start.max(jb)..je {
                    re.swap(i * stride + j, j * stride + i);
                    im.swap(i * stride + j, j * stride + i);
                }
            }
        }
    }
}

/// Plans keyed by `(length, inverse)`: the two directions carry conjugate
/// twiddles and cannot share an entry.
/// Planar, row-permuted four-step twiddle planes.
///
/// The interleaved `W_n^(j*b)` matrix cannot feed a vector multiply against
/// planar data, which is why the transpose's fused multiply was scalar — the
/// measured 26% of the driver at N = 256. These planes fix the layout
/// disagreement once, at build: split into real and imaginary planes, with
/// rows in bit-reversed order so stage-set-2's loads (which run after the row
/// permutation) index them directly. The matrix is symmetric (`W^(j*b)` is
/// its own transpose), which is what lets the multiply move to the other side
/// of the transpose at all.
pub(crate) struct FourStepPlanes<T> {
    pub(crate) re: Box<[T]>,
    pub(crate) im: Box<[T]>,
}

impl<T: MixedRadixScalar<Complex = Complex<T>>> FourStepPlanes<T> {
    fn new<const INVERSE: bool>(n: usize, m: usize) -> Self
    where
        Complex<T>: crate::application::execution::kernel::twiddle_table::TwiddleOutput,
    {
        // Collect the uncached entry stream directly into its final planes,
        // so no full interleaved matrix overlaps their allocation. The threaded four-step's sizes cache
        // exactly one too — the interleaved matrix its fused
        // transpose-multiply reads with better locality — and the two size
        // ranges are disjoint by the routing thresholds. An explicit
        // vector rewrite of the driver's boundary loops was measured against
        // this same baseline and declined: the compiler already vectorizes
        // those canonical interleave patterns, and the added dispatch
        // round-trips made every size 2 to 7% slower.
        let interleaved =
            crate::application::execution::kernel::mixed_radix::caches::build_four_step_twiddles::<
                Complex<T>,
                INVERSE,
            >(n, m, m);
        // Rows in natural order, because the stage set that folds these
        // consumes natural order: it is decimated in frequency. The planes
        // were row-permuted while that set was decimated in time and its
        // input arrived bit-reversed.
        let (re, im): (Vec<_>, Vec<_>) = interleaved.map(|value| (value.re, value.im)).unzip();
        Self {
            re: re.into_boxed_slice(),
            im: im.into_boxed_slice(),
        }
    }
}

/// Runs the stage set of `batch` transforms of length `plan.len` over planar
/// `re`/`im`, whose rows the caller has already bit-reversed — which the
/// driver's deinterleave does for free by writing each row to its reversed
/// position.
fn run_batched<T>(
    re: &mut [T],
    im: &mut [T],
    plan: &BatchedPlan<T>,
    source: Option<&[T]>,
    batch: usize,
    stride: usize,
) where
    T: LaneScalar + MixedRadixScalar + radix::Lane,
{
    hermes_simd::vectorize(BatchedStages {
        re,
        im,
        tw: &plan.tw,
        source,
        batch,
        stride,
        len: plan.len,
    });
}

/// Runs the same stage set decimated in frequency: rows arrive in natural
/// order and leave bit-reversed, which is the pairing that lets the sink
/// absorb the permutation the time-decimated set needed a whole pass to
/// repair (see [`dif`]).
fn run_batched_dif<T>(
    re: &mut [T],
    im: &mut [T],
    plan: &BatchedPlan<T>,
    fold: Option<(&[T], &[T])>,
    sink: Option<&mut [T]>,
    batch: usize,
    stride: usize,
) where
    T: LaneScalar + MixedRadixScalar + radix::Lane,
{
    hermes_simd::vectorize(dif::BatchedStagesDif {
        re,
        im,
        tw: &plan.tw,
        fold,
        sink,
        batch,
        stride,
        len: plan.len,
    });
}

/// Scratch length, in complex elements, that [`four_step_batched`] requires
/// for a transform of length `n`.
///
/// The single definition of the padded-plane requirement, so callers and the
/// driver cannot disagree about it.
pub(crate) fn scratch_len(n: usize) -> usize {
    let m = 1usize << (n.trailing_zeros() / 2);
    m * (m + ROW_PAD)
}

/// Largest length the planar route serves; longer even powers fall to the
/// generic four-step, whose rows thread through Moirai.
///
/// The bound used to be the generic route's threading threshold (65536), on
/// the premise that threaded rows beat a sequential SIMD pass from there.
/// Measured on the pinned performance core against that premise (ADR 0053),
/// the generic route at 65536 cost 2.7 to 4.5 times RustFFT while this route
/// one length below sat at 1.25 times; at 65536 this route measured 208 to
/// 228 µs against the generic route's 466 to 767 across four runs, and at
/// 262144 it halved `f32` while leaving `f64` level. The next even power,
/// 1048576, was measured only under host contention and stays on the
/// generic route until a quiet replicated census decides it. The value binds
/// to one host's cache hierarchy and Moirai's dispatch cost; re-measure
/// before moving it in either direction.
pub(crate) const PLANAR_MAX_LEN: usize = 1 << 18;

/// Whether [`four_step_batched`] covers a transform of length `n`.
///
/// The single definition of the planar route's domain: an even power of two
/// — the square split the driver is written for — up to [`PLANAR_MAX_LEN`].
pub(crate) fn planar_applies(n: usize) -> bool {
    n.is_power_of_two() && n.trailing_zeros() % 2 == 0 && n >= 4 && n <= PLANAR_MAX_LEN
}

/// Whether [`four_step_split_batched`] covers a transform of length `n`.
///
/// An odd power of two has no square split, so it decimates once; the route
/// applies when both halves are then planar. The lower bound is the one the
/// unfused decimation already carried — below it the plan hands these
/// lengths to the base kernels and codelets, which never reach here, so
/// widening the domain would change only what tests exercise.
pub(crate) fn planar_split_applies(n: usize) -> bool {
    n.is_power_of_two() && n.trailing_zeros() % 2 == 1 && n >= 512 && planar_applies(n / 2)
}

/// Scratch, in complex elements, that [`four_step_split_batched`] requires.
///
/// Both half-planes are live at once, which is what lets the combine read
/// them together; against the unfused route — a full `n`-element decimation
/// buffer plus one half-plane nested inside it — this is the smaller peak.
pub(crate) fn split_scratch_len(n: usize) -> usize {
    2 * scratch_len(n / 2)
}

#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) mod sections;

/// Plane geometry for a length-`n` planar transform: the square edge `m` and
/// the padded row stride. Element `j` of a transform lands at plane index
/// `(j / m) * stride + j % m`.
///
/// # Panics
///
/// Panics if `n` is not an even power of two of at least four.
fn plane_geometry(n: usize) -> (usize, usize) {
    let k = n.trailing_zeros();
    assert!(
        n.is_power_of_two() && k % 2 == 0 && n >= 4,
        "requires an even power of two of at least 4"
    );
    let m = 1usize << (k / 2);
    (m, m + ROW_PAD)
}

/// Splits a plane buffer into its real and imaginary halves.
///
/// One complex scratch element is two reals, so `m * stride` complexes hold
/// the two padded planes. The pad breaks the power-of-two row stride that
/// makes every plane row alias to one L1 set (see [`ROW_PAD`]).
///
/// # Panics
///
/// Panics if `scratch` is shorter than `plane`.
fn split_plane<T>(scratch: &mut [Complex<T>], plane: usize) -> (&mut [T], &mut [T])
where
    T: eunomia::layout::Pod,
    Complex<T>: eunomia::layout::Pod,
{
    assert!(
        scratch.len() >= plane,
        "scratch must hold two padded planes"
    );
    let flat: &mut [T] = eunomia::layout::cast_slice_mut(&mut scratch[..plane]);
    flat.split_at_mut(plane)
}

/// Writes both halves of a radix-2 decimation into their planes in one pass.
///
/// The alternative is [`deinterleave_rows`] twice over a strided view, which
/// reads every cache line of `src` once per half. Taking the adjacent pair
/// together reads each line once for both, which is why this exists rather
/// than a step parameter on the sequential form
/// (`gap_audit.md#odd-power-fusion`).
fn deinterleave_decimated_rows<T: Copy>(
    src: &[Complex<T>],
    even: (&mut [T], &mut [T]),
    odd: (&mut [T], &mut [T]),
    m: usize,
    stride: usize,
) {
    let (e_re, e_im) = even;
    let (o_re, o_im) = odd;
    let row_bits = m.trailing_zeros();
    for (row, chunk) in src.chunks_exact(2 * m).enumerate().take(m) {
        let dest = row.reverse_bits() >> (usize::BITS - row_bits);
        let base = dest * stride;
        for b in 0..m {
            let e = chunk[2 * b];
            let o = chunk[2 * b + 1];
            e_re[base + b] = e.re;
            e_im[base + b] = e.im;
            o_re[base + b] = o.re;
            o_im[base + b] = o.im;
        }
    }
}

/// Runs both four-step stage sets over the planes.
///
/// With `seams` the interleaved buffer is the input of the first stage set
/// and the output of the second: the first pass reads it in bit-reversed row
/// order and the last pass writes it back the same way, so neither a
/// deinterleave nor a reinterleave pass exists. Without it the planes hold
/// the input in bit-reversed rows on entry and the output on exit, which is
/// what the odd-power split needs, since its input is decimated and its
/// output combined by their own sinks.
fn planar_stages<T, const INVERSE: bool>(
    re: &mut [T],
    im: &mut [T],
    n: usize,
    m: usize,
    stride: usize,
    seams: Option<&mut [Complex<T>]>,
) where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let seams = seams.map(|data| -> &mut [T] { eunomia::layout::cast_slice_mut(data) });
    // 1. The `m` transforms of length `m` along the first axis; the input is
    //    already batch-major for this direction, so no transpose is needed.
    let plan = T::cached_plan::<INVERSE>(m);
    sect!("stages1", {
        run_batched(re, im, plan.as_ref(), seams.as_deref(), m, stride)
    });

    // 2. Transpose so the second axis becomes batch-major. Pure exchange:
    //    the four-step twiddle now rides stage-set-2's first loads below.
    sect!("transpose", {
        let handled = if T::BOUNDARY_LANES == 8 {
            hermes_simd::vectorize_lanes::<8, T, _>(boundary::TransposePlanes {
                re: &mut *re,
                im: &mut *im,
                m,
                stride,
            })
            .unwrap_or(false)
        } else {
            false
        } || hermes_simd::vectorize_lanes::<4, T, _>(boundary::TransposePlanes {
            re,
            im,
            m,
            stride,
        })
        .unwrap_or(false);
        if !handled {
            transpose_planes(re, im, m, stride);
        }
    });

    // 3. The `m` transforms along the second axis, with the four-step twiddle
    //    W_N^{b·k1} folded into the first stage's loads — the matrix is
    //    symmetric, so applying it after the transpose is identical, and its
    //    planar planes are built once in the shared cache.
    //
    //    This set is decimated in frequency, and that is what deletes the
    //    repair pass that used to sit here. The transpose leaves natural row
    //    order, which is what DIF consumes; it leaves its output
    //    bit-reversed, which the sink absorbs by reading `rev(row)` — the
    //    mirror of the deinterleave's free permutation on the source side.
    //    The result still lands at `k2 * m + k1` once the sink has read it.
    let planes = T::cached_four_step_planes::<INVERSE>(n, m);
    sect!("stages2", {
        run_batched_dif(
            re,
            im,
            plan.as_ref(),
            Some((&planes.re, &planes.im)),
            seams,
            m,
            stride,
        )
    });
}

/// In-place four-step FFT over the padded planar layout.
///
/// # Panics
///
/// Panics if `data.len()` is not an even power of two of at least four, or
/// if `scratch` is shorter than [`scratch_len`].
pub(crate) fn four_step_batched<T, const INVERSE: bool>(
    data: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
) where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let n = data.len();
    let (m, stride) = plane_geometry(n);
    let plane = scratch_len(n);
    let (re, im) = split_plane(scratch, plane);
    // The interleaved buffer is read by the first stage pass and written by
    // the last, so the route is exactly two stage sets and one transpose.
    planar_stages::<T, INVERSE>(re, im, n, m, stride, Some(data));
}

pub(crate) fn four_step_split_batched<T, const INVERSE: bool>(
    data: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
) where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let n = data.len();
    assert!(
        planar_split_applies(n),
        "requires a planar odd power of two"
    );
    let half = n / 2;
    let plane = scratch_len(half);
    assert!(
        scratch.len() >= 2 * plane,
        "scratch must hold both half-planes"
    );
    // The stage-major table ends with the length-`n` stage, whose `n / 2`
    // entries are `W_N^j` in order; earlier stages occupy `n / 2 - 1` slots.
    let twiddles = if INVERSE {
        T::cached_twiddle_inv(n)
    } else {
        T::cached_twiddle_fwd(n)
    };
    let combine = &twiddles[half - 1..n - 1];

    let (m, stride) = plane_geometry(half);
    let (even, odd) = scratch.split_at_mut(plane);
    sect!("deint", {
        let (e_re, e_im) = split_plane(even, plane);
        let (o_re, o_im) = split_plane(odd, plane);
        deinterleave_decimated_rows(data, (e_re, e_im), (o_re, o_im), m, stride);
    });
    {
        let (re, im) = split_plane(even, plane);
        planar_stages::<T, INVERSE>(re, im, half, m, stride, None);
    }
    {
        let (re, im) = split_plane(odd, plane);
        planar_stages::<T, INVERSE>(re, im, half, m, stride, None);
    }
    combine_planar_halves(data, even, odd, m, stride, combine);
}

/// Combines two planar half-transforms into `data` in one pass.
///
/// `even` and `odd` hold the transforms of the even- and odd-indexed
/// subsequences in the padded plane layout defined by [`plane_geometry`].
/// This writes `X[j] = E[j] + W_N^j O[j]` and `X[j + N/2] = E[j] - W_N^j
/// O[j]`, so the butterfly rides the pass that would have interleaved each
/// half back on its own — the pass, and the half-sized buffers it would
/// have written to, are what the fusion removes.
///
/// # Panics
///
/// Panics if `data` is not twice the half-transform length, or if
/// `twiddles` is shorter than that half.
pub(crate) fn combine_planar_halves<T>(
    data: &mut [Complex<T>],
    even: &[Complex<T>],
    odd: &[Complex<T>],
    m: usize,
    stride: usize,
    twiddles: &[Complex<T>],
) where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let half = m * m;
    assert_eq!(data.len(), 2 * half, "combine spans both halves");
    assert!(twiddles.len() >= half, "one rotation per output pair");
    let plane = m * stride;
    let (e_re, e_im) = eunomia::layout::cast_slice::<_, T>(&even[..plane]).split_at(plane);
    let (o_re, o_im) = eunomia::layout::cast_slice::<_, T>(&odd[..plane]).split_at(plane);
    let (low, high) = data.split_at_mut(half);

    // The stage set left bit-reversed rows, and the permutation rides the
    // write side: plane row `p` holds output row `rev(p)`, bit reversal
    // being an involution. Reading the planes in order keeps four streams
    // sequential and leaves two scattered, where carrying it on the read
    // side scattered four (gap_audit.md#sink-permutation).
    let bits = m.trailing_zeros();
    sect!("combine", {
        let twiddle_lanes = eunomia::layout::cast_slice(twiddles);
        let handled =
            if T::BOUNDARY_LANES == 8 {
                hermes_simd::vectorize_hardware_lanes::<8, T, _>(boundary::CombinePlanarHalves {
                    even_re: e_re,
                    even_im: e_im,
                    odd_re: o_re,
                    odd_im: o_im,
                    twiddles: twiddle_lanes,
                    low: eunomia::layout::cast_slice_mut(&mut *low),
                    high: eunomia::layout::cast_slice_mut(&mut *high),
                    m,
                    stride,
                })
                .unwrap_or(false)
            } else {
                false
            } || hermes_simd::vectorize_hardware_lanes::<4, T, _>(boundary::CombinePlanarHalves {
                even_re: e_re,
                even_im: e_im,
                odd_re: o_re,
                odd_im: o_im,
                twiddles: twiddle_lanes,
                low: eunomia::layout::cast_slice_mut(&mut *low),
                high: eunomia::layout::cast_slice_mut(&mut *high),
                m,
                stride,
            })
            .unwrap_or(false);

        if !handled {
            for row in 0..m {
                let base = row * stride;
                let dst = (row.reverse_bits() >> (usize::BITS - bits)) * m;
                for b in 0..m {
                    let j = dst + b;
                    let e = Complex::new(e_re[base + b], e_im[base + b]);
                    let o = Complex::new(o_re[base + b], o_im[base + b]);
                    let rotated = o * twiddles[j];
                    low[j] = e + rotated;
                    high[j] = e - rotated;
                }
            }
        }
    });
}

pub(crate) mod boundary;
mod dif;

// Test-gated deliberately: the interleaved in-place kernel is correct and
// measured — and slower than the planar sibling by 16 to 37% pinned on an
// E-core at every covered size, because the shuffle cost of interleaved
// butterflies outweighs the planar seams it removes. It stays as the
// independent-implementation differential oracle for this module; the
// pinned probe that declined it is committed beside it.
#[cfg(test)]
pub(crate) mod interleaved;

// Windows-gated: pins threads through Win32 to control the hybrid scheduler.
#[cfg(all(test, windows))]
mod pinned_ladder;

// Additionally x86-gated: the pass totals come from `_rdtsc`.
#[cfg(all(test, windows, target_arch = "x86_64"))]
mod pinned_sections;

#[cfg(test)]
mod tests;

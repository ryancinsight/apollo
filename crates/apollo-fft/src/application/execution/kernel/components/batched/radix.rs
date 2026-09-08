//! Butterfly arithmetic and the row-set driver shared by both stage sets.
//!
//! A stage set is a sequence of passes; a pass applies one or two consecutive
//! stages to every row set of the planes. The per-element operation order of
//! a stage is fixed by the transform, so grouping stages into passes changes
//! only how many times each element is loaded and stored: a radix-4 pass
//! (two stages) streams the planes once where two radix-2 passes would
//! stream them twice. Results are bitwise those of the single-stage form
//! whatever the grouping; the grouping stops at two because eight planar
//! rows exceed the AVX2 register file (ADR 0055).
//!
//! The arithmetic is written once over [`Lane`], which both the SIMD vector
//! and the scalar element implement, so the vector loop and its scalar
//! remainder run the same butterfly. The vector form fuses each complex
//! multiply's products (`w.re * x.re - w.im * x.im` as one fused
//! multiply-add against the exactly negated second product); the scalar form
//! keeps the two-operation product the remainder always used. Both are the
//! forms the stage sets carried before the grouping was generalised, so the
//! outputs of every existing test are unchanged bit for bit.

use core::ops::{Add, Mul, Sub};

use hermes_simd::{LaneScalar, SimdArch, SimdKernel, SimdStorage, Vector};

use super::{load, load_interleaved, reverse_row, store, store_interleaved};

/// One lane of butterfly arithmetic: a SIMD vector or a scalar element.
pub(crate) trait Lane:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    /// `self * b + c`.
    fn fma(self, b: Self, c: Self) -> Self;
    /// `self * b - c`.
    fn fms(self, b: Self, c: Self) -> Self;
}

impl<T, A> Lane for Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }

    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn fms(self, b: Self, c: Self) -> Self {
        self.mul_add(b, -c)
    }
}

impl Lane for f64 {
    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        self * b + c
    }

    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn fms(self, b: Self, c: Self) -> Self {
        self * b - c
    }
}

impl Lane for f32 {
    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        self * b + c
    }

    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn fms(self, b: Self, c: Self) -> Self {
        self * b - c
    }
}

/// A complex value or a lane of complex values as its `(re, im)` parts.
pub(super) type Pair<L> = (L, L);

/// `x * w` for `(re, im)` pairs.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
)]
#[inline(always)]
pub(super) fn cmul<L: Lane>((xr, xi): (L, L), (wr, wi): (L, L)) -> (L, L) {
    (wr.fms(xr, wi * xi), wr.fma(xi, wi * xr))
}

/// Time-decimated butterfly: the twiddle multiplies the odd operand first.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
)]
#[inline(always)]
fn dit<L: Lane>(a: (L, L), b: (L, L), w: (L, L)) -> ((L, L), (L, L)) {
    let t = cmul(b, w);
    ((a.0 + t.0, a.1 + t.1), (a.0 - t.0, a.1 - t.1))
}

/// Frequency-decimated butterfly: the sum keeps its place and the
/// difference carries the twiddle.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
)]
#[inline(always)]
fn dif<L: Lane>(a: (L, L), b: (L, L), w: (L, L)) -> ((L, L), (L, L)) {
    let d = (a.0 - b.0, a.1 - b.1);
    ((a.0 + b.0, a.1 + b.1), cmul(d, w))
}

/// One pass over `N` rows with `NT` twiddles: the stages it fuses and the
/// order it applies them in.
///
/// Rows arrive in the order the stage set's driver enumerates them and leave
/// in the same order; twiddles arrive in the order each implementor
/// documents.
pub(super) trait Radix<const N: usize, const NT: usize> {
    /// Applies the fused stages to one row set.
    fn apply<L: Lane>(x: [(L, L); N], t: &[(L, L); NT]) -> [(L, L); N];
}

/// One time-decimated stage `l`: rows `(r, r + l/2)`, twiddle `W_l^j`.
pub(super) struct Dit2;

/// Stages `l` and `2l` time-decimated: rows at offsets `0, l/2, l, 3l/2`;
/// twiddles `W_l^j`, `W_2l^j`, `W_2l^(j + l/2)`.
pub(super) struct Dit4;

/// One frequency-decimated stage `l`: rows `(r, r + l/2)`, twiddle `W_l^j`.
pub(super) struct Dif2;

/// Stages `l` and `l/2` frequency-decimated: rows at offsets `i * l/4` for
/// `i < 4`; twiddles `W_l^j`, `W_l^(j + l/4)`, `W_(l/2)^j`.
pub(super) struct Dif4;

impl Radix<2, 1> for Dit2 {
    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn apply<L: Lane>([a, b]: [(L, L); 2], t: &[(L, L); 1]) -> [(L, L); 2] {
        let (a, b) = dit(a, b, t[0]);
        [a, b]
    }
}

impl Radix<4, 3> for Dit4 {
    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn apply<L: Lane>([a, b, c, d]: [(L, L); 4], t: &[(L, L); 3]) -> [(L, L); 4] {
        let (a, b) = dit(a, b, t[0]);
        let (c, d) = dit(c, d, t[0]);
        let (a, c) = dit(a, c, t[1]);
        let (b, d) = dit(b, d, t[2]);
        [a, b, c, d]
    }
}

impl Radix<2, 1> for Dif2 {
    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn apply<L: Lane>([a, b]: [(L, L); 2], t: &[(L, L); 1]) -> [(L, L); 2] {
        let (a, b) = dif(a, b, t[0]);
        [a, b]
    }
}

impl Radix<4, 3> for Dif4 {
    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn apply<L: Lane>([a, b, c, d]: [(L, L); 4], t: &[(L, L); 3]) -> [(L, L); 4] {
        let (a, c) = dif(a, c, t[0]);
        let (b, d) = dif(b, d, t[1]);
        let (a, b) = dif(a, b, t[2]);
        let (c, d) = dif(c, d, t[2]);
        [a, b, c, d]
    }
}

/// Where a pass reads its rows and where it writes them.
///
/// The interleaved source is the caller's buffer on the first pass of the
/// time-decimated set and the interleaved sink the caller's buffer on the
/// last pass of the frequency-decimated set; every other pass reads and
/// writes the planes. `fold` multiplies the four-step twiddle planes into
/// the loaded rows on whichever pass carries it.
pub(super) struct Seams<'a, 'b, T> {
    /// Four-step twiddle planes, row-major with row stride `batch`.
    pub(super) fold: Option<(&'a [T], &'a [T])>,
    /// Interleaved input rows of `batch` complexes; plane row `p` reads row
    /// `rev(p)`.
    pub(super) source: Option<&'a [T]>,
    /// Interleaved output rows; plane row `p` writes row `rev(p)`.
    pub(super) sink: Option<&'b mut [T]>,
}

/// Runs radix `R` over one row set across the whole batch: the vector loop
/// over `LANE_COUNT` columns at a time and the scalar remainder.
///
/// `rows` are plane row indices in the order `R` expects; `tw` holds the
/// scalar twiddles for the remainder and `twv` their lane splats for the
/// vector loop. The caller has bounded every row index by the plane extent
/// and the batch by the row length, which is what the unchecked loads and
/// stores rely on.
#[expect(
    clippy::inline_always,
    reason = "the driver must fold into the dispatcher's target-feature scope with the kernel that calls it"
)]
#[inline(always)]
pub(super) fn butterfly_rows<T, A, R, const N: usize, const NT: usize>(
    re: &mut [T],
    im: &mut [T],
    rows: [usize; N],
    stride: usize,
    batch: usize,
    row_bits: u32,
    tw: &[Pair<T>; NT],
    twv: &[Pair<Vector<T, A>>; NT],
    seams: Seams<'_, '_, T>,
) where
    T: LaneScalar + Lane,
    A: SimdArch + SimdKernel<T>,
    R: Radix<N, NT>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    let plane: [usize; N] = rows.map(|row| row * stride);
    let fold_at: [usize; N] = rows.map(|row| row * batch);
    let interleaved: [usize; N] = rows.map(|row| reverse_row(row, row_bits) * batch * 2);
    let Seams {
        fold,
        source,
        mut sink,
    } = seams;

    let mut k = 0;
    while k + lanes <= batch {
        let mut x: [(Vector<T, A>, Vector<T, A>); N] = core::array::from_fn(|i| match source {
            Some(src) => load_interleaved::<T, A>(src, interleaved[i] + 2 * k),
            None => (
                load::<T, A>(re, plane[i] + k),
                load::<T, A>(im, plane[i] + k),
            ),
        });
        if let Some((pr, pi)) = fold {
            for (value, at) in x.iter_mut().zip(fold_at) {
                let w = (load::<T, A>(pr, at + k), load::<T, A>(pi, at + k));
                *value = cmul(*value, w);
            }
        }
        let y = R::apply(x, twv);
        match sink.as_deref_mut() {
            Some(out) => {
                for (value, at) in y.into_iter().zip(interleaved) {
                    store_interleaved::<T, A>(value.0, value.1, out, at + 2 * k);
                }
            }
            None => {
                for (value, at) in y.into_iter().zip(plane) {
                    store::<T, A>(value.0, re, at + k);
                    store::<T, A>(value.1, im, at + k);
                }
            }
        }
        k += lanes;
    }

    // Scalar remainder when the batch is not a lane multiple.
    for k in k..batch {
        let mut x: [(T, T); N] = core::array::from_fn(|i| match source {
            Some(src) => (src[interleaved[i] + 2 * k], src[interleaved[i] + 2 * k + 1]),
            None => (re[plane[i] + k], im[plane[i] + k]),
        });
        if let Some((pr, pi)) = fold {
            for (value, at) in x.iter_mut().zip(fold_at) {
                *value = cmul(*value, (pr[at + k], pi[at + k]));
            }
        }
        let y = R::apply(x, tw);
        match sink.as_deref_mut() {
            Some(out) => {
                for (value, at) in y.into_iter().zip(interleaved) {
                    out[at + 2 * k] = value.0;
                    out[at + 2 * k + 1] = value.1;
                }
            }
            None => {
                for (value, at) in y.into_iter().zip(plane) {
                    re[at + k] = value.0;
                    im[at + k] = value.1;
                }
            }
        }
    }
}

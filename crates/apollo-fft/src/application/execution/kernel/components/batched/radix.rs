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

/// The rows of one row set: `first + i * step` for `i < N`.
///
/// Every pass in both stage sets addresses equally spaced rows, and stating
/// the spacing rather than listing the rows lets the pass keep one base and
/// one step per plane instead of a table of offsets that would otherwise
/// occupy, and then overflow, the general registers.
#[derive(Clone, Copy)]
pub(super) struct Rows {
    /// Plane row of the first element of the set.
    pub(super) first: usize,
    /// Plane rows between consecutive elements of the set.
    pub(super) step: usize,
}

/// The columns of one row set a pass covers: batch indices `start..end`.
///
/// A sweep ([`super::sweep`]) runs its passes over one column block of a
/// tile at a time so the block stays in L1 between them; the block is the
/// whole batch only when the tile fits.
#[derive(Clone, Copy)]
pub(super) struct Columns {
    pub(super) start: usize,
    pub(super) end: usize,
}

/// A staged sink block addressed by tile-local row.
///
/// The sink writes one tile block at a time into a contiguous buffer (see
/// [`super::sweep`]): tile row
/// `r` starts `r * pitch` reals in, and batch column `k` sits
/// `2 * (k - first_column)` reals further. The row set's first row is tile
/// row `first`, and its rows are [`Rows::step`] tile rows apart, the same
/// step as in the planes.
#[derive(Clone, Copy)]
pub(super) struct Staging {
    pub(super) first: usize,
    pub(super) pitch: usize,
    pub(super) first_column: usize,
}

impl Staging {
    /// The staging of a pass without a seam: never addressed.
    const NONE: Self = Self {
        first: 0,
        pitch: 0,
        first_column: 0,
    };
}

/// Where a pass reads its rows and where it writes them.
///
/// Exactly the combinations the two stage sets produce: the time-decimated
/// set reads the caller's interleaved rows on its first pass and the planes
/// otherwise; the frequency-decimated set folds the four-step twiddle on its
/// first pass, writes the staged sink block on its last, and both when the
/// two coincide. Each variant selects a monomorphized pass whose row loop
/// carries no seam it does not use, which is what keeps the pass's
/// addressing in registers.
pub(super) enum Seams<'a, 'b, T> {
    /// Planes in, planes out.
    Planes,
    /// Interleaved rows in, `rows` bits wide; plane row `p` reads row `rev(p)`.
    Source(&'a [T], u32),
    /// Planes in with the four-step twiddle planes multiplied into the loads.
    Fold((&'a [T], &'a [T])),
    /// Planes in, staged interleaved rows out.
    Sink(&'b mut [T], Staging),
    /// Folded loads and staged stores in one pass.
    FoldSink((&'a [T], &'a [T]), &'b mut [T], Staging),
}

impl<'a, 'b, T> Seams<'a, 'b, T> {
    /// The frequency-decimated set's seams for one pass.
    pub(super) fn frequency(
        fold: Option<(&'a [T], &'a [T])>,
        sink: Option<(&'b mut [T], Staging)>,
    ) -> Self {
        match (fold, sink) {
            (Some(fold), Some((sink, staging))) => Self::FoldSink(fold, sink, staging),
            (Some(fold), None) => Self::Fold(fold),
            (None, Some((sink, staging))) => Self::Sink(sink, staging),
            (None, None) => Self::Planes,
        }
    }

    /// The time-decimated set's seams for one pass.
    pub(super) fn time(source: Option<(&'a [T], u32)>) -> Self {
        source.map_or(Self::Planes, |(source, row_bits)| {
            Self::Source(source, row_bits)
        })
    }
}

/// Runs radix `R` over one row set across `cols`: the vector loop over
/// `LANE_COUNT` columns at a time and the scalar remainder.
///
/// `tw` holds the scalar twiddles for the remainder and `twv` their lane
/// splats for the vector loop. The caller has bounded every row of `rows` by
/// the plane extent, `cols` by the batch, the source rows by the caller's
/// buffer and the staged rows by the staging buffer, which is what the
/// unchecked loads and stores rely on.
#[expect(
    clippy::inline_always,
    reason = "the driver must fold into the dispatcher's target-feature scope with the kernel that calls it"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the driver is the inner function of both stage sets; its arguments are the pass's operands"
)]
#[inline(always)]
pub(super) fn butterfly_rows<T, A, R, const N: usize, const NT: usize>(
    re: &mut [T],
    im: &mut [T],
    rows: Rows,
    stride: usize,
    batch: usize,
    cols: Columns,
    tw: &[Pair<T>; NT],
    twv: &[Pair<Vector<T, A>>; NT],
    seams: Seams<'_, '_, T>,
) where
    T: LaneScalar + Lane,
    A: SimdArch + SimdKernel<T>,
    R: Radix<N, NT>,
{
    let none: (&[T], &[T]) = (&[], &[]);
    match seams {
        Seams::Planes => pass::<T, A, R, N, NT, false, false, false>(
            re,
            im,
            rows,
            stride,
            batch,
            cols,
            tw,
            twv,
            none,
            &[],
            0,
            &mut [],
            Staging::NONE,
        ),
        Seams::Source(source, row_bits) => pass::<T, A, R, N, NT, true, false, false>(
            re,
            im,
            rows,
            stride,
            batch,
            cols,
            tw,
            twv,
            none,
            source,
            row_bits,
            &mut [],
            Staging::NONE,
        ),
        Seams::Fold(fold) => pass::<T, A, R, N, NT, false, true, false>(
            re,
            im,
            rows,
            stride,
            batch,
            cols,
            tw,
            twv,
            fold,
            &[],
            0,
            &mut [],
            Staging::NONE,
        ),
        Seams::Sink(sink, staging) => pass::<T, A, R, N, NT, false, false, true>(
            re,
            im,
            rows,
            stride,
            batch,
            cols,
            tw,
            twv,
            none,
            &[],
            0,
            sink,
            staging,
        ),
        Seams::FoldSink(fold, sink, staging) => pass::<T, A, R, N, NT, false, true, true>(
            re,
            im,
            rows,
            stride,
            batch,
            cols,
            tw,
            twv,
            fold,
            &[],
            0,
            sink,
            staging,
        ),
    }
}

/// One monomorphized pass: the seams it carries are compile-time, so its
/// row loop tests nothing and computes only the offsets it uses.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope, as the driver"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the pass is the inner function of one driver; its arguments are the driver's, unbundled so each monomorphization keeps only the seams it carries in registers"
)]
#[inline(always)]
fn pass<
    T,
    A,
    R,
    const N: usize,
    const NT: usize,
    const SOURCE: bool,
    const FOLD: bool,
    const SINK: bool,
>(
    re: &mut [T],
    im: &mut [T],
    rows: Rows,
    stride: usize,
    batch: usize,
    cols: Columns,
    tw: &[Pair<T>; NT],
    twv: &[Pair<Vector<T, A>>; NT],
    fold: (&[T], &[T]),
    source: &[T],
    source_bits: u32,
    sink: &mut [T],
    staging: Staging,
) where
    T: LaneScalar + Lane,
    A: SimdArch + SimdKernel<T>,
    R: Radix<N, NT>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    let plane_first = rows.first * stride;
    let plane_step = rows.step * stride;
    let fold_first = rows.first * batch;
    let fold_step = rows.step * batch;
    // Staged sink rows are tile-local and affine like the plane rows; the
    // bit-reversed source row map is not, so the source keeps a table. Both
    // are dead, and so unmaterialized, in the passes without that seam.
    let staged_first = staging.first * staging.pitch;
    let staged_step = rows.step * staging.pitch;
    let source_rows: [usize; N] =
        core::array::from_fn(|i| reverse_row(rows.first + i * rows.step, source_bits) * batch * 2);

    let mut k = cols.start;
    while k + lanes <= cols.end {
        let mut x: [Pair<Vector<T, A>>; N] = core::array::from_fn(|i| {
            if SOURCE {
                load_interleaved::<T, A>(source, source_rows[i] + 2 * k)
            } else {
                let at = plane_first + i * plane_step + k;
                (load::<T, A>(re, at), load::<T, A>(im, at))
            }
        });
        if FOLD {
            for (i, value) in x.iter_mut().enumerate() {
                let at = fold_first + i * fold_step + k;
                let w = (load::<T, A>(fold.0, at), load::<T, A>(fold.1, at));
                *value = cmul(*value, w);
            }
        }
        let y = R::apply(x, twv);
        if SINK {
            for (i, value) in y.into_iter().enumerate() {
                let at = staged_first + i * staged_step + 2 * (k - staging.first_column);
                store_interleaved::<T, A>(value.0, value.1, sink, at);
            }
        } else {
            for (i, value) in y.into_iter().enumerate() {
                let at = plane_first + i * plane_step + k;
                store::<T, A>(value.0, re, at);
                store::<T, A>(value.1, im, at);
            }
        }
        k += lanes;
    }

    // Scalar remainder when the block is not a lane multiple, which is only
    // a batch narrower than one register; the plane column order is then
    // the identity, so column `k` is interleaved sample `k`.
    for k in k..cols.end {
        let mut x: [Pair<T>; N] = core::array::from_fn(|i| {
            if SOURCE {
                let at = source_rows[i] + 2 * k;
                (source[at], source[at + 1])
            } else {
                let at = plane_first + i * plane_step + k;
                (re[at], im[at])
            }
        });
        if FOLD {
            for (i, value) in x.iter_mut().enumerate() {
                let at = fold_first + i * fold_step + k;
                *value = cmul(*value, (fold.0[at], fold.1[at]));
            }
        }
        let y = R::apply(x, tw);
        if SINK {
            for (i, value) in y.into_iter().enumerate() {
                let at = staged_first + i * staged_step + 2 * (k - staging.first_column);
                sink[at] = value.0;
                sink[at + 1] = value.1;
            }
        } else {
            for (i, value) in y.into_iter().enumerate() {
                let at = plane_first + i * plane_step + k;
                re[at] = value.0;
                im[at] = value.1;
            }
        }
    }
}

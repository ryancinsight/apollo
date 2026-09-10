//! The butterflies and the radices both stage sets fuse.
//!
//! A stage set is a sequence of passes; a pass applies one or two consecutive
//! stages to every row set of the planes. The per-element operation order of
//! a stage is fixed by the transform, so grouping stages into passes changes
//! only how many times each element is loaded and stored: a radix-4 pass
//! (two stages) streams the planes once where two radix-2 passes would
//! stream them twice. Results are bitwise those of the single-stage form
//! whatever the grouping; the grouping stops at two because eight planar
//! rows exceed the AVX2 register file (ADR 0055).

use super::lane::Lane;

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

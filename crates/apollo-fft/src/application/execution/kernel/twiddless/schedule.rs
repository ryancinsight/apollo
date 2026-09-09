//! The shared arithmetic body and the three data-movement schedules over it.
//!
//! `compress` and `butterfly` apply the same level to a node, out of place and
//! in place; `leaf` is the published base case; `gather` is lines 12 and 14 of
//! Algorithm 1. Each schedule composes exactly these, so per element the
//! floating-point operation sequence is identical and outputs agree bitwise.

use eunomia::Complex;

use super::{TwiddlessPlan, TwiddlessScalar};

mod private {
    pub trait Sealed {}
    impl Sealed for super::CompressedHalves {}
    impl Sealed for super::ButterflyRecursive {}
    impl Sealed for super::ButterflyIterative {}
}

/// Data-movement strategy over the shared arithmetic body.
///
/// Sealed: the three implementations are the experiment's arms and a new one
/// is a new measured claim.
pub trait Schedule: private::Sealed {
    /// Forward transform of `input` into natural-order `output`; `scratch`
    /// holds one plan length. Length preconditions are checked by
    /// [`TwiddlessPlan::forward`].
    fn forward<F: TwiddlessScalar>(
        plan: &TwiddlessPlan<F>,
        input: &[Complex<F>],
        output: &mut [Complex<F>],
        scratch: &mut [Complex<F>],
    );
}

/// Algorithm 1 as published: each level writes both compressed halves of a
/// node into the other buffer and recurses depth first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompressedHalves;

/// The same depth-first traversal with the butterfly applied in place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ButterflyRecursive;

/// The classic breadth-first stage loop of in-place butterflies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ButterflyIterative;

/// Lines 9 and 10 for one pair: the sum and the modulated difference.
#[inline]
fn pair<F: TwiddlessScalar>(
    a: Complex<F>,
    b: Complex<F>,
    w: Complex<F>,
) -> (Complex<F>, Complex<F>) {
    (a + b, (a - b) * w)
}

/// One level out of place: `[x̂e | x̂o]` of `node` written into `halves`.
///
/// `stride = 2^ℓ` selects `W_N^{n·stride} = W_M^n` for the node length
/// `M = N / 2^ℓ`; the stepped table yields `M ≥ M/2` entries, so the zip is
/// bounded by the halves.
fn compress<F: TwiddlessScalar>(
    node: &[Complex<F>],
    halves: &mut [Complex<F>],
    table: &[Complex<F>],
    stride: usize,
) {
    let half = node.len() / 2;
    let (lo, hi) = node.split_at(half);
    let (even, odd) = halves.split_at_mut(half);
    let twiddles = table.iter().step_by(stride);
    for ((((a, b), e), o), w) in lo
        .iter()
        .zip(hi)
        .zip(even.iter_mut())
        .zip(odd.iter_mut())
        .zip(twiddles)
    {
        let (sum, difference) = pair(*a, *b, *w);
        *e = sum;
        *o = difference;
    }
}

/// The same level in place: the Gentleman–Sande butterfly over `node`.
fn butterfly<F: TwiddlessScalar>(node: &mut [Complex<F>], table: &[Complex<F>], stride: usize) {
    let half = node.len() / 2;
    let (lo, hi) = node.split_at_mut(half);
    let twiddles = table.iter().step_by(stride);
    for ((a, b), w) in lo.iter_mut().zip(hi.iter_mut()).zip(twiddles) {
        let (sum, difference) = pair(*a, *b, *w);
        *a = sum;
        *b = difference;
    }
}

/// `z · (−i)`, exact.
#[inline]
fn times_negative_i<F: TwiddlessScalar>(z: Complex<F>) -> Complex<F> {
    Complex::new(z.im, F::ZERO - z.re)
}

/// The published base case `F(x)` on one leaf block, in place.
///
/// Lengths 1, 2 and 4 have unit twiddles `{1, −1, ±i}` that the definition
/// folds exactly; lengths 3 and 5 evaluate the definition against the table,
/// `W_c^{jm} = W_N^{(jm mod c)·step}` with `step = N / c`.
fn leaf<F: TwiddlessScalar>(block: &mut [Complex<F>], table: &[Complex<F>], step: usize) {
    match block {
        [] | [_] => {}
        [a, b] => {
            let (sum, difference) = (*a + *b, *a - *b);
            *a = sum;
            *b = difference;
        }
        [x0, x1, x2, x3] => {
            let (s02, d02) = (*x0 + *x2, *x0 - *x2);
            let (s13, d13) = (*x1 + *x3, *x1 - *x3);
            let rotated = times_negative_i(d13);
            *x0 = s02 + s13;
            *x1 = d02 + rotated;
            *x2 = s02 - s13;
            *x3 = d02 - rotated;
        }
        _ => {
            let c = block.len();
            debug_assert!(
                c == 3 || c == 5,
                "leaf lengths are bounded by the base case"
            );
            let mut samples = [Complex::<F>::default(); super::BASE_CASE_MAX];
            samples[..c].copy_from_slice(block);
            for (j, bin) in block.iter_mut().enumerate() {
                let mut acc = Complex::<F>::default();
                for (m, sample) in samples[..c].iter().enumerate() {
                    acc += *sample * table[((j * m) % c) * step];
                }
                *bin = acc;
            }
        }
    }
}

/// Lines 12 and 14: leaf block `b` holds output residue `rev_k(b)`, so
/// `X[rev_k(b) + j·2^k] = block_b[j]`.
fn gather<F: TwiddlessScalar>(
    blocks: &[Complex<F>],
    output: &mut [Complex<F>],
    levels: u32,
    leaf_len: usize,
) {
    let chunks = blocks.chunks_exact(leaf_len);
    debug_assert!(chunks.remainder().is_empty(), "N = leaf · 2^levels");
    for (index, block) in chunks.enumerate() {
        let residue = if levels == 0 {
            0
        } else {
            index.reverse_bits() >> (usize::BITS - levels)
        };
        for (j, value) in block.iter().enumerate() {
            output[residue + (j << levels)] = *value;
        }
    }
}

/// Depth-first descent of one node under the published schedule: `cur` holds
/// the node, `other` receives its compressed halves, roles swap per level.
fn descend_out_of_place<F: TwiddlessScalar>(
    cur: &mut [Complex<F>],
    other: &mut [Complex<F>],
    table: &[Complex<F>],
    stride: usize,
    leaf_len: usize,
    step: usize,
) {
    if cur.len() == leaf_len {
        leaf(cur, table, step);
        return;
    }
    compress(cur, other, table, stride);
    let half = cur.len() / 2;
    let (cur_lo, cur_hi) = cur.split_at_mut(half);
    let (other_lo, other_hi) = other.split_at_mut(half);
    descend_out_of_place(other_lo, cur_lo, table, stride * 2, leaf_len, step);
    descend_out_of_place(other_hi, cur_hi, table, stride * 2, leaf_len, step);
}

/// Depth-first descent of one node with in-place butterflies.
fn descend_in_place<F: TwiddlessScalar>(
    node: &mut [Complex<F>],
    table: &[Complex<F>],
    stride: usize,
    leaf_len: usize,
    step: usize,
) {
    if node.len() == leaf_len {
        leaf(node, table, step);
        return;
    }
    butterfly(node, table, stride);
    let half = node.len() / 2;
    let (lo, hi) = node.split_at_mut(half);
    descend_in_place(lo, table, stride * 2, leaf_len, step);
    descend_in_place(hi, table, stride * 2, leaf_len, step);
}

/// The base case reached without any level: `F(x)` straight into `output`.
fn leaf_only<F: TwiddlessScalar>(
    plan: &TwiddlessPlan<F>,
    input: &[Complex<F>],
    output: &mut [Complex<F>],
) {
    output.copy_from_slice(input);
    leaf(output, plan.table(), plan.transform_len() / plan.leaf());
}

impl Schedule for CompressedHalves {
    fn forward<F: TwiddlessScalar>(
        plan: &TwiddlessPlan<F>,
        input: &[Complex<F>],
        output: &mut [Complex<F>],
        scratch: &mut [Complex<F>],
    ) {
        if plan.levels() == 0 {
            leaf_only(plan, input, output);
            return;
        }
        let table = plan.table();
        let step = plan.transform_len() / plan.leaf();
        let half = plan.transform_len() / 2;
        // Level ℓ writes buffer P_{ℓ+1} with P alternating, so the leaves land
        // in P_k; choose the level-0 target so that P_k is `scratch`.
        let (first, second): (&mut [Complex<F>], &mut [Complex<F>]) = if plan.levels() % 2 == 1 {
            (&mut *scratch, &mut *output)
        } else {
            (&mut *output, &mut *scratch)
        };
        compress(input, first, table, 1);
        let (first_lo, first_hi) = first.split_at_mut(half);
        let (second_lo, second_hi) = second.split_at_mut(half);
        descend_out_of_place(first_lo, second_lo, table, 2, plan.leaf(), step);
        descend_out_of_place(first_hi, second_hi, table, 2, plan.leaf(), step);
        gather(scratch, output, plan.levels(), plan.leaf());
    }
}

impl Schedule for ButterflyRecursive {
    fn forward<F: TwiddlessScalar>(
        plan: &TwiddlessPlan<F>,
        input: &[Complex<F>],
        output: &mut [Complex<F>],
        scratch: &mut [Complex<F>],
    ) {
        if plan.levels() == 0 {
            leaf_only(plan, input, output);
            return;
        }
        let table = plan.table();
        let step = plan.transform_len() / plan.leaf();
        compress(input, scratch, table, 1);
        let (lo, hi) = scratch.split_at_mut(plan.transform_len() / 2);
        descend_in_place(lo, table, 2, plan.leaf(), step);
        descend_in_place(hi, table, 2, plan.leaf(), step);
        gather(scratch, output, plan.levels(), plan.leaf());
    }
}

impl Schedule for ButterflyIterative {
    fn forward<F: TwiddlessScalar>(
        plan: &TwiddlessPlan<F>,
        input: &[Complex<F>],
        output: &mut [Complex<F>],
        scratch: &mut [Complex<F>],
    ) {
        if plan.levels() == 0 {
            leaf_only(plan, input, output);
            return;
        }
        let table = plan.table();
        let step = plan.transform_len() / plan.leaf();
        compress(input, scratch, table, 1);
        let mut stride = 2;
        let mut node_len = plan.transform_len() / 2;
        while node_len > plan.leaf() {
            for node in scratch.chunks_exact_mut(node_len) {
                butterfly(node, table, stride);
            }
            stride *= 2;
            node_len /= 2;
        }
        for block in scratch.chunks_exact_mut(plan.leaf()) {
            leaf(block, table, step);
        }
        gather(scratch, output, plan.levels(), plan.leaf());
    }
}

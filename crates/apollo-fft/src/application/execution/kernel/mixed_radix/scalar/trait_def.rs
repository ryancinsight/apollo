use super::super::traits::ShortWinogradScalar;
use super::transpose::transpose_tiled_scalar;
use crate::application::execution::kernel::components::radix_composite::CompositeCache;
use crate::application::execution::kernel::pot::{PoTStrategy, SizedPoT};
use std::sync::Arc;

pub(crate) mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for f32 {}
}

/// Bluestein cache key: (m, inverse_flag, generator_inverse).
pub(crate) type BluesteinKey = (usize, bool, usize);

pub(crate) type BluesteinEntry<C> = Arc<[C]>;

pub trait BluesteinStore {
    type Cpx: Copy + Send + Sync + 'static;
    fn tl_get(key: BluesteinKey) -> Option<BluesteinEntry<Self::Cpx>>;
    fn tl_insert(key: BluesteinKey, val: BluesteinEntry<Self::Cpx>);
    fn global(
    ) -> &'static parking_lot::RwLock<rustc_hash::FxHashMap<BluesteinKey, BluesteinEntry<Self::Cpx>>>;
}

pub trait MixedRadixScalar:
    private::Sealed
    + Sized
    + Copy
    + 'static
    + ShortWinogradScalar
    + CompositeCache
    + BluesteinStore<Cpx = Self::Complex>
{
    /// Minimum Rader convolution length `N - 1` for the half-cyclic CRT split.
    ///
    /// `usize::MAX` disables automatic production routing while retaining the
    /// strategy for forced benchmark and equivalence tests.
    const HALF_CYCLIC_RADER_THRESHOLD: usize;

    /// Prime lengths whose Rader convolution uses the half-cyclic backend before
    /// the generic threshold. The slice is scalar-specific and resolved through
    /// monomorphization, avoiding runtime type inspection in the Rader selector.
    const HALF_CYCLIC_RADER_PRIMES: &'static [usize];

    /// Precision-specific composite stage order for N=200.
    ///
    /// The stage order is part of the scalar policy because f64 and f32 favor
    /// different cache/register tradeoffs in the measured fused Stockham path.
    const COMPOSITE_RADICES_200: &'static [usize];

    /// Whether N=63 is forced to the composite route for this scalar.
    const FORCE_COMPOSITE_63: bool;

    /// Whether N=72 is forced to the composite route for this scalar.
    const FORCE_COMPOSITE_72: bool;

    /// Whether runtime Rader should prefer Bluestein for f32-sized mid primes.
    const PREFER_BLUESTEIN_MID_RADER: bool;

    /// Whether Bluestein pads to a power-of-two length instead of the next
    /// 7-smooth length. This is a scalar policy because f32 and f64 favor
    /// different Stockham/composite tradeoffs.
    const BLUESTEIN_PAD_POWER_OF_TWO: bool;

    /// Whether Bluestein phase construction uses native f32 trigonometry before
    /// entering the scalar complex constructor.
    const BLUESTEIN_NATIVE_PHASE_TRIG: bool;

    type Complex: Copy
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = Self::Complex>
        + std::ops::Mul<Output = Self::Complex>;

    fn complex(re: f64, im: f64) -> Self::Complex;

    fn cached_twiddle_fwd(n: usize) -> Arc<[Self::Complex]>;
    fn cached_twiddle_inv(n: usize) -> Arc<[Self::Complex]>;

    fn with_twiddle_fwd<R>(n: usize, f: impl FnOnce(&[Self::Complex]) -> R) -> R;
    fn with_twiddle_inv<R>(n: usize, f: impl FnOnce(&[Self::Complex]) -> R) -> R;

    fn cached_rader_spectrum<const INVERSE: bool>(
        n: usize,
        generator_inverse: usize,
    ) -> Arc<[Self::Complex]>;

    /// Return precomputed negacyclic Rader convolution spectra for prime length `n`.
    ///
    /// Returns `(cyclic_spectrum, negacyclic_spectrum)` each of length `(n-1)/2`.
    fn cached_rader_negacyclic_spectra<const INVERSE: bool>(
        n: usize,
        generator_inverse: usize,
    ) -> (Arc<[Self::Complex]>, Arc<[Self::Complex]>);

    /// Return precomputed twist twiddles `e^{i*pi*j/m}` for negacyclic convolution.
    fn cached_rader_neg_twiddles(m: usize) -> Arc<[Self::Complex]>;

    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Self::Complex]) -> R) -> R;
    fn with_pfa_scratch<R>(n: usize, f: impl FnOnce(&mut [Self::Complex]) -> R) -> R;
    fn with_rader_padded_scratch<R>(n: usize, f: impl FnOnce(&mut [Self::Complex]) -> R) -> R;
    fn with_bluestein_scratch<R>(n: usize, f: impl FnOnce(&mut [Self::Complex]) -> R) -> R;

    fn cached_four_step_twiddles<const INVERSE: bool>(
        n: usize,
        n1: usize,
        n2: usize,
    ) -> Arc<[Self::Complex]>;

    fn pointwise_mul(a: &mut [Self::Complex], b: &[Self::Complex]);

    fn pointwise_mul_conj(a: &mut [Self::Complex], b: &[Self::Complex]);

    fn stockham_forward(
        data: &mut [Self::Complex],
        scratch: &mut [Self::Complex],
        twiddles: &[Self::Complex],
    );

    #[inline]
    fn stockham_forward_normalized(
        data: &mut [Self::Complex],
        scratch: &mut [Self::Complex],
        twiddles: &[Self::Complex],
        n: usize,
    ) {
        Self::stockham_forward(data, scratch, twiddles);
        Self::normalize(data, n);
    }

    /// Sized stockham forward (const LOG2) to push monomorphization from plan PoT
    /// sized executors through pot_inplace_sized down to the kernel (full const LOG2
    /// to transform_sized / with_strategy / lenXXX, zero runtime log2 in hot path).
    fn stockham_forward_sized<const LOG2: u32>(
        data: &mut [Self::Complex],
        scratch: &mut [Self::Complex],
        twiddles: &[Self::Complex],
    ) {
        // Default fallback (runtime log2); f32/f64 impls override with const LOG2 for mono.
        let n = 1usize << LOG2;
        Self::stockham_forward(data, scratch, twiddles);
        if false {
            Self::normalize(data, n);
        }
    }

    #[inline]
    fn stockham_forward_normalized_sized<const LOG2: u32>(
        data: &mut [Self::Complex],
        scratch: &mut [Self::Complex],
        twiddles: &[Self::Complex],
    ) {
        Self::stockham_forward_sized::<LOG2>(data, scratch, twiddles);
        Self::normalize(data, 1usize << LOG2);
    }

    fn short_winograd<const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [Self::Complex],
    ) -> bool;

    /// In-place small transforms (2, 3, 4, 5, 6, 7, 8, 9, 16, 32).
    ///
    /// # Safety
    /// Caller must guarantee `data.len()` matches the expected size.
    unsafe fn small_pot_inplace<const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [Self::Complex],
    ) -> bool;

    /// Sized small power-of-two transforms.
    ///
    /// # Safety
    /// Caller must guarantee `data.len()` matches N.
    unsafe fn small_pot_inplace_sized<const N: usize, const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [Self::Complex],
    );

    fn composite_forward(data: &mut [Self::Complex], radices: &[usize]);
    fn composite_forward_with_pointwise(
        data: &mut [Self::Complex],
        radices: &[usize],
        pointwise_spectrum: &[Self::Complex],
    );
    fn composite_inverse_unnorm(data: &mut [Self::Complex], radices: &[usize]);
    fn composite_inverse(data: &mut [Self::Complex], radices: &[usize]);

    fn normalize(data: &mut [Self::Complex], n: usize);

    fn transpose_matrix(src: &[Self::Complex], dst: &mut [Self::Complex], n1: usize, n2: usize) {
        transpose_tiled_scalar(src, dst, n1, n2);
    }

    /// Optimized power-of-two in-place transform.
    fn pot_inplace<const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [Self::Complex],
        twiddles: &[Self::Complex],
    );

    /// Optimized power-of-two in-place transform, monomorphized by const LOG2 + PoTStrategy ZST.
    ///
    /// This elevates the architecture: the SizedPoT< S, LOG2 > from FftPlan1D (SSOT for PoT
    /// selection and strategy) is threaded directly to the implementation (zero-cost,
    /// enables full const propagation to stockham transform_sized / with_strategy / lenXXX
    /// bodies, and monomorph per (strategy, LOG2) for future PoT schedule markers).
    /// Callers with known compile-time LOG2 (plan exec_sized arms for hot sizes 128+) must
    /// use this; generic/unknown PoT sizes continue to use pot_inplace (runtime log2 inside).
    ///
    /// Default delegates to pot_inplace (preserves compat for old call sites; updated call
    /// sites in same diff pass the ZST for mono benefit).
    fn pot_inplace_sized<
        const INVERSE: bool,
        const NORMALIZE: bool,
        S: PoTStrategy,
        const LOG2: u32,
    >(
        data: &mut [Self::Complex],
        twiddles: &[Self::Complex],
        _s: SizedPoT<S, LOG2>,
    ) {
        Self::pot_inplace::<INVERSE, NORMALIZE>(data, twiddles);
    }

    /// Precomputed twiddles for small power-of-two sizes.
    fn small_pot_twiddles<const INVERSE: bool>(n: usize) -> &'static [Self::Complex];

    fn use_generated_codelet_plan(n: usize) -> bool;
}

use crate::application::execution::kernel::winograd::{apply_twiddle_impl, WinogradScalar};
use num_complex::Complex;
use std::cell::RefCell;
use std::sync::Arc;

#[derive(Clone)]
pub struct CompositeTwiddleEntry<C> {
    pub radices: Arc<[usize]>,
    pub twiddles: Arc<[C]>,
    pub offsets: Arc<[usize]>,
}

pub trait CompositeCache: WinogradScalar {
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex<Self>]) -> R) -> R;
    fn cached_twiddles(inverse: bool, radices: &[usize]) -> (Arc<[Complex<Self>]>, Arc<[usize]>);
}

thread_local! {
    static TL_TWIDDLES_FWD_64: RefCell<Vec<CompositeTwiddleEntry<num_complex::Complex64>>> = const { RefCell::new(Vec::new()) };
    static TL_TWIDDLES_INV_64: RefCell<Vec<CompositeTwiddleEntry<num_complex::Complex64>>> = const { RefCell::new(Vec::new()) };

    static TL_TWIDDLES_FWD_32: RefCell<Vec<CompositeTwiddleEntry<num_complex::Complex32>>> = const { RefCell::new(Vec::new()) };
    static TL_TWIDDLES_INV_32: RefCell<Vec<CompositeTwiddleEntry<num_complex::Complex32>>> = const { RefCell::new(Vec::new()) };

    static TL_COMPOSITE_SCRATCH_64: RefCell<Vec<Vec<num_complex::Complex64>>> =
        const { RefCell::new(Vec::new()) };
    static TL_COMPOSITE_SCRATCH_32: RefCell<Vec<Vec<num_complex::Complex32>>> =
        const { RefCell::new(Vec::new()) };
}

fn build_composite_twiddles<F: WinogradScalar>(
    inverse: bool,
    radices: &[usize],
) -> (Vec<Complex<F>>, Vec<usize>) {
    let sign: f64 = if inverse { 1.0 } else { -1.0 };
    let total_twiddles: usize = radices
        .iter()
        .scan(1usize, |p, &r| {
            let out = *p;
            *p *= r;
            Some(out)
        })
        .sum();
    let mut all_twiddles = Vec::with_capacity(total_twiddles);
    unsafe { all_twiddles.set_len(total_twiddles) };
    let mut stage_offsets = Vec::with_capacity(radices.len());
    unsafe { stage_offsets.set_len(radices.len()) };

    let one = Complex::new(F::cast_f64(1.0), F::cast_f64(0.0));
    let mut prev_len = 1usize;
    let mut tw_idx = 0;
    let mut offset_idx = 0;
    for &r in radices {
        let stage_len = prev_len * r;
        unsafe { *stage_offsets.get_unchecked_mut(offset_idx) = tw_idx };
        offset_idx += 1;
        let base_angle = sign * std::f64::consts::TAU / stage_len as f64;
        let w_base = Complex::new(F::cast_f64(base_angle.cos()), F::cast_f64(base_angle.sin()));
        let mut tw = one;
        for _ in 0..prev_len {
            unsafe { *all_twiddles.get_unchecked_mut(tw_idx) = tw };
            tw_idx += 1;
            tw = apply_twiddle_impl(tw, w_base);
        }
        prev_len = stage_len;
    }
    debug_assert_eq!(tw_idx, total_twiddles);
    debug_assert_eq!(offset_idx, radices.len());
    (all_twiddles, stage_offsets)
}

impl CompositeCache for f64 {
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex<Self>]) -> R) -> R {
        let mut scratch =
            TL_COMPOSITE_SCRATCH_64.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
        if scratch.len() < n {
            let cur = scratch.len();
            scratch.reserve(n.saturating_sub(cur));
            unsafe { scratch.set_len(n) };
        }
        let res = f(&mut scratch[..n]);
        TL_COMPOSITE_SCRATCH_64.with(|pool| pool.borrow_mut().push(scratch));
        res
    }

    #[inline]
    fn cached_twiddles(inverse: bool, radices: &[usize]) -> (Arc<[Complex<Self>]>, Arc<[usize]>) {
        let tl = if inverse {
            &TL_TWIDDLES_INV_64
        } else {
            &TL_TWIDDLES_FWD_64
        };
        if let Some(cached) = tl.with(|cache| {
            cache
                .borrow()
                .iter()
                .find(|entry| entry.radices.as_ref() == radices)
                .map(|entry| (Arc::clone(&entry.twiddles), Arc::clone(&entry.offsets)))
        }) {
            return cached;
        }
        let (tw, offsets) = build_composite_twiddles::<f64>(inverse, radices);
        let tw = Arc::from(tw.into_boxed_slice());
        let offsets = Arc::from(offsets.into_boxed_slice());
        tl.with(|c| {
            c.borrow_mut().push(CompositeTwiddleEntry {
                radices: Arc::from(radices),
                twiddles: Arc::clone(&tw),
                offsets: Arc::clone(&offsets),
            });
        });
        (tw, offsets)
    }
}

impl CompositeCache for f32 {
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex<Self>]) -> R) -> R {
        let mut scratch =
            TL_COMPOSITE_SCRATCH_32.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
        if scratch.len() < n {
            let cur = scratch.len();
            scratch.reserve(n.saturating_sub(cur));
            unsafe { scratch.set_len(n) };
        }
        let res = f(&mut scratch[..n]);
        TL_COMPOSITE_SCRATCH_32.with(|pool| pool.borrow_mut().push(scratch));
        res
    }

    #[inline]
    fn cached_twiddles(inverse: bool, radices: &[usize]) -> (Arc<[Complex<Self>]>, Arc<[usize]>) {
        let tl = if inverse {
            &TL_TWIDDLES_INV_32
        } else {
            &TL_TWIDDLES_FWD_32
        };
        if let Some(cached) = tl.with(|cache| {
            cache
                .borrow()
                .iter()
                .find(|entry| entry.radices.as_ref() == radices)
                .map(|entry| (Arc::clone(&entry.twiddles), Arc::clone(&entry.offsets)))
        }) {
            return cached;
        }
        let (tw, offsets) = build_composite_twiddles::<f32>(inverse, radices);
        let tw = Arc::from(tw.into_boxed_slice());
        let offsets = Arc::from(offsets.into_boxed_slice());
        tl.with(|c| {
            c.borrow_mut().push(CompositeTwiddleEntry {
                radices: Arc::from(radices),
                twiddles: Arc::clone(&tw),
                offsets: Arc::clone(&offsets),
            });
        });
        (tw, offsets)
    }
}

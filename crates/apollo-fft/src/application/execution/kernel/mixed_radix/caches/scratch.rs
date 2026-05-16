use num_complex::{Complex32, Complex64};
use std::cell::RefCell;

thread_local! {
    pub(super) static TL_STOCKHAM_SCRATCH_64: RefCell<Vec<Complex64>> =
        const { RefCell::new(Vec::new()) };
    pub(super) static TL_STOCKHAM_SCRATCH_32: RefCell<Vec<Complex32>> =
        const { RefCell::new(Vec::new()) };
    pub(super) static TL_PFA_SCRATCH_64: RefCell<Vec<Vec<Complex64>>> =
        const { RefCell::new(Vec::new()) };
    pub(super) static TL_PFA_SCRATCH_32: RefCell<Vec<Vec<Complex32>>> =
        const { RefCell::new(Vec::new()) };
    pub(super) static TL_RADER_PADDED_SCRATCH_64: RefCell<Vec<Vec<Complex64>>> =
        const { RefCell::new(Vec::new()) };
    pub(super) static TL_RADER_PADDED_SCRATCH_32: RefCell<Vec<Vec<Complex32>>> =
        const { RefCell::new(Vec::new()) };
}

#[inline]
pub(crate) fn get_aligned_slice_mut<T>(vec: &mut Vec<T>, n: usize, align_bytes: usize) -> &mut [T] {
    let size = std::mem::size_of::<T>();
    let align_elements = align_bytes / size;
    let required_len = n + align_elements;
    if vec.len() < required_len {
        let cur = vec.len();
        vec.reserve(required_len.saturating_sub(cur));
        unsafe { vec.set_len(required_len) };
    }
    let ptr = vec.as_mut_ptr() as usize;
    let offset = ptr.wrapping_neg() & (align_bytes - 1);
    let start_idx = offset / size;
    &mut vec[start_idx..start_idx + n]
}

#[inline]
pub(crate) fn with_stockham_scratch_64<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
    TL_STOCKHAM_SCRATCH_64.with(|cell| match cell.try_borrow_mut() {
        Ok(mut scratch) => {
            let aligned = get_aligned_slice_mut(&mut scratch, n, 64);
            f(aligned)
        }
        Err(_) => {
            let mut local: Vec<Complex64> = {
                let mut v = Vec::with_capacity(n + 7);
                unsafe { v.set_len(n + 7) };
                v
            };
            let aligned = get_aligned_slice_mut(&mut local, n, 64);
            f(aligned)
        }
    })
}

#[inline]
pub(crate) fn with_stockham_scratch_32<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
    TL_STOCKHAM_SCRATCH_32.with(|cell| match cell.try_borrow_mut() {
        Ok(mut scratch) => {
            let aligned = get_aligned_slice_mut(&mut scratch, n, 64);
            f(aligned)
        }
        Err(_) => {
            let mut local: Vec<Complex32> = {
                let mut v = Vec::with_capacity(n + 15);
                unsafe { v.set_len(n + 15) };
                v
            };
            let aligned = get_aligned_slice_mut(&mut local, n, 64);
            f(aligned)
        }
    })
}

#[inline]
pub(crate) fn with_pfa_scratch_64<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
    let mut scratch = TL_PFA_SCRATCH_64.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
    let res = f(get_aligned_slice_mut(&mut scratch, n, 64));
    TL_PFA_SCRATCH_64.with(|pool| pool.borrow_mut().push(scratch));
    res
}

#[inline]
pub(crate) fn with_pfa_scratch_32<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
    let mut scratch = TL_PFA_SCRATCH_32.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
    let res = f(get_aligned_slice_mut(&mut scratch, n, 64));
    TL_PFA_SCRATCH_32.with(|pool| pool.borrow_mut().push(scratch));
    res
}

#[inline]
pub(crate) fn with_rader_padded_scratch_64<R>(
    n: usize,
    f: impl FnOnce(&mut [Complex64]) -> R,
) -> R {
    let mut scratch =
        TL_RADER_PADDED_SCRATCH_64.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
    let res = f(get_aligned_slice_mut(&mut scratch, n, 64));
    TL_RADER_PADDED_SCRATCH_64.with(|pool| {
        let mut p = pool.borrow_mut();
        if p.len() < 2 {
            p.push(scratch);
        }
    });
    res
}

#[inline]
pub(crate) fn with_rader_padded_scratch_32<R>(
    n: usize,
    f: impl FnOnce(&mut [Complex32]) -> R,
) -> R {
    let mut scratch =
        TL_RADER_PADDED_SCRATCH_32.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
    let res = f(get_aligned_slice_mut(&mut scratch, n, 64));
    TL_RADER_PADDED_SCRATCH_32.with(|pool| {
        let mut p = pool.borrow_mut();
        if p.len() < 2 {
            p.push(scratch);
        }
    });
    res
}

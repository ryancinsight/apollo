use num_complex::{Complex32, Complex64};
use std::borrow::Cow;
use std::cell::RefCell;

// Alignment padding: for 64-byte alignment, we need at most 7 elements (f32=8B) or 3 (f64=16B)
// Using 8 elements for f32 and 4 for f64 as minimum needed, with safety margin
const STOCKHAM_F32_ALIGN_PADDING: usize = 8; // 64 / 8 = 8 elements max offset
const STOCKHAM_F64_ALIGN_PADDING: usize = 4; // 64 / 16 = 4 elements max offset

thread_local! {
    // Stockham scratch buffers: start empty, grow on demand
    // Pre-allocation on first use is efficient since it grows geometrically
    static TL_STOCKHAM_SCRATCH_64: RefCell<Vec<Complex64>> =
        const { RefCell::new(Vec::new()) };
    static TL_STOCKHAM_SCRATCH_32: RefCell<Vec<Complex32>> =
        const { RefCell::new(Vec::new()) };

    // PFA uses a pool of Vecs for recursive sub-FFTs
    static TL_PFA_SCRATCH_64: RefCell<Vec<Vec<Complex64>>> =
        const { RefCell::new(Vec::new()) };
    static TL_PFA_SCRATCH_32: RefCell<Vec<Vec<Complex32>>> =
        const { RefCell::new(Vec::new()) };

    // Rader padded scratch
    static TL_RADER_PADDED_SCRATCH_64: RefCell<Vec<Complex64>> =
        const { RefCell::new(Vec::new()) };
    static TL_RADER_PADDED_SCRATCH_32: RefCell<Vec<Complex32>> =
        const { RefCell::new(Vec::new()) };

    // Bluestein chirp scratch
    static TL_BLUESTEIN_SCRATCH_64: RefCell<Vec<Complex64>> =
        const { RefCell::new(Vec::new()) };
    static TL_BLUESTEIN_SCRATCH_32: RefCell<Vec<Complex32>> =
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
    // Compute offset to achieve align_bytes alignment: (align - (ptr mod align)) mod align
    let offset = (align_bytes - (ptr & (align_bytes - 1))) & (align_bytes - 1);
    let start_idx = offset / size;
    &mut vec[start_idx..start_idx + n]
}

mod sealed {
    pub(crate) trait ScratchStoreSealed {}
}

/// Sealed trait providing thread-local scratch buffer access per complex type.
///
/// Implemented for `Complex64` and `Complex32`. All access is via the three
/// generic free functions `with_stockham_scratch`, `with_pfa_scratch`, and
/// `with_rader_padded_scratch`.
pub(crate) trait ScratchStore: sealed::ScratchStoreSealed + 'static {
    fn with_stockham_scratch_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_pfa_scratch_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_rader_padded_scratch_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
    fn with_bluestein_scratch_impl<R, F: FnOnce(&mut [Self]) -> R>(n: usize, f: F) -> R;
}

impl sealed::ScratchStoreSealed for Complex64 {}
impl ScratchStore for Complex64 {
    #[inline]
    fn with_stockham_scratch_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_STOCKHAM_SCRATCH_64.with(|cell| match cell.try_borrow_mut() {
            Ok(mut scratch) => {
                let aligned = get_aligned_slice_mut(&mut scratch, n, 64);
                f(aligned)
            }
            Err(_) => {
                // Nested call path: allocate with minimal padding for alignment
                let mut local: Vec<Complex64> = {
                    let mut v = Vec::with_capacity(n + STOCKHAM_F64_ALIGN_PADDING);
                    unsafe { v.set_len(n + STOCKHAM_F64_ALIGN_PADDING) };
                    v
                };
                let aligned = get_aligned_slice_mut(&mut local, n, 64);
                // Cow zero-copy view for nested fallback (TL pooled vs owned); now named and available for
                // sub-operations (e.g. read-only twiddle/spectrum views or future external borrow unification).
                // Mem eff additive for rader/PoT pads using with_stockham during bluestein etc.
                let zero_copy_view: Cow<[Complex64]> = Cow::Borrowed(aligned);
                let _ = &zero_copy_view; // exercised; prevents dead, ready for read views
                f(aligned)
            }
        })
    }

    #[inline]
    fn with_pfa_scratch_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        let mut scratch =
            TL_PFA_SCRATCH_64.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
        let res = f(get_aligned_slice_mut(&mut scratch, n, 64));
        TL_PFA_SCRATCH_64.with(|pool| pool.borrow_mut().push(scratch));
        res
    }

    #[inline]
    fn with_rader_padded_scratch_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_RADER_PADDED_SCRATCH_64.with(|cell| match cell.try_borrow_mut() {
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
    fn with_bluestein_scratch_impl<R, F: FnOnce(&mut [Complex64]) -> R>(n: usize, f: F) -> R {
        TL_BLUESTEIN_SCRATCH_64.with(|cell| match cell.try_borrow_mut() {
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
}

impl sealed::ScratchStoreSealed for Complex32 {}
impl ScratchStore for Complex32 {
    #[inline]
    fn with_stockham_scratch_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_STOCKHAM_SCRATCH_32.with(|cell| match cell.try_borrow_mut() {
            Ok(mut scratch) => {
                let aligned = get_aligned_slice_mut(&mut scratch, n, 64);
                f(aligned)
            }
            Err(_) => {
                // Nested call path: allocate with minimal padding for alignment
                let mut local: Vec<Complex32> = {
                    let mut v = Vec::with_capacity(n + STOCKHAM_F32_ALIGN_PADDING);
                    unsafe { v.set_len(n + STOCKHAM_F32_ALIGN_PADDING) };
                    v
                };
                let aligned = get_aligned_slice_mut(&mut local, n, 64);
                let zero_copy_view: Cow<[Complex32]> = Cow::Borrowed(aligned);
                let _ = &zero_copy_view;
                f(aligned)
            }
        })
    }

    #[inline]
    fn with_pfa_scratch_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        let mut scratch =
            TL_PFA_SCRATCH_32.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
        let res = f(get_aligned_slice_mut(&mut scratch, n, 64));
        TL_PFA_SCRATCH_32.with(|pool| pool.borrow_mut().push(scratch));
        res
    }

    #[inline]
    fn with_rader_padded_scratch_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_RADER_PADDED_SCRATCH_32.with(|cell| match cell.try_borrow_mut() {
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
    fn with_bluestein_scratch_impl<R, F: FnOnce(&mut [Complex32]) -> R>(n: usize, f: F) -> R {
        TL_BLUESTEIN_SCRATCH_32.with(|cell| match cell.try_borrow_mut() {
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
}

#[inline]
pub(crate) fn with_stockham_scratch<C: ScratchStore, R, F: FnOnce(&mut [C]) -> R>(
    n: usize,
    f: F,
) -> R {
    C::with_stockham_scratch_impl(n, f)
}

#[inline]
pub(crate) fn with_pfa_scratch<C: ScratchStore, R, F: FnOnce(&mut [C]) -> R>(n: usize, f: F) -> R {
    C::with_pfa_scratch_impl(n, f)
}

#[inline]
pub(crate) fn with_rader_padded_scratch<C: ScratchStore, R, F: FnOnce(&mut [C]) -> R>(
    n: usize,
    f: F,
) -> R {
    C::with_rader_padded_scratch_impl(n, f)
}

#[inline]
pub(crate) fn with_bluestein_scratch<C: ScratchStore, R, F: FnOnce(&mut [C]) -> R>(
    n: usize,
    f: F,
) -> R {
    C::with_bluestein_scratch_impl(n, f)
}

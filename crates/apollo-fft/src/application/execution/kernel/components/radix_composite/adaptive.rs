use num_complex::Complex;

use super::arity::dispatch_single_radix;
use super::cache::CompositeCache;
use crate::application::execution::kernel::mixed_radix::traits::ShortWinogradScalar;

struct ComposeArena {
    buf: Vec<u8>,
    top: usize,
}

impl ComposeArena {
    const fn new() -> Self {
        Self {
            buf: Vec::new(),
            top: 0,
        }
    }

    #[inline]
    unsafe fn reserve<F>(&mut self, count: usize) {
        let elem_size = core::mem::size_of::<Complex<F>>();
        let align = core::mem::align_of::<Complex<F>>();
        let byte_size = count * elem_size;
        let aligned = (self.top + align - 1) & !(align - 1);
        let needed = aligned + byte_size;
        if self.buf.len() < needed {
            assert_eq!(self.top, 0, "arena realloc with live inner pointers");
            self.buf.resize(needed.next_power_of_two(), 0u8);
        }
    }

    #[inline]
    unsafe fn alloc<F>(&mut self, count: usize) -> (*mut Complex<F>, usize) {
        let elem_size = core::mem::size_of::<Complex<F>>();
        let align = core::mem::align_of::<Complex<F>>();
        let byte_size = count * elem_size;
        let aligned = (self.top + align - 1) & !(align - 1);
        let needed = aligned + byte_size;
        if self.buf.len() < needed {
            assert_eq!(self.top, 0, "arena realloc with live inner pointers");
            let cap = (byte_size * 2).next_power_of_two().max(needed);
            self.buf.resize(cap, 0u8);
        }
        let ptr = self.buf.as_mut_ptr().add(aligned).cast::<Complex<F>>();
        let saved = self.top;
        self.top = needed;
        (ptr, saved)
    }

    #[inline]
    fn dealloc(&mut self, saved: usize) {
        self.top = saved;
    }
}

thread_local! {
    static COMPOSE_ARENA: core::cell::UnsafeCell<ComposeArena> =
        core::cell::UnsafeCell::new(ComposeArena::new());
}

struct ArenaGuard(usize);

impl Drop for ArenaGuard {
    #[inline]
    fn drop(&mut self) {
        COMPOSE_ARENA.with(|cell| unsafe { (*cell.get()).dealloc(self.0) });
    }
}

fn composite_adaptive_scratch_size_elems(radices: &[usize], prev_len: usize) -> usize {
    if radices.len() <= 1 {
        return 0;
    }
    let outer_r = radices[radices.len() - 1];
    let inner_r_total: usize = radices[..radices.len() - 1].iter().product();
    outer_r * inner_r_total * prev_len
        + composite_adaptive_scratch_size_elems(&radices[..radices.len() - 1], prev_len)
}

fn composite_fused_adaptive_inner<F: CompositeCache + ShortWinogradScalar, const INVERSE: bool>(
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    scratch: &mut [Complex<F>],
    prev_len: usize,
    b_out: usize,
    groups_out: usize,
    radices: &[usize],
    twiddles: &[&[Complex<F>]],
    pointwise_spectrum: Option<&[Complex<F>]>,
) {
    let n_stages = radices.len();
    debug_assert!(n_stages >= 1);
    if n_stages == 1 {
        dispatch_single_radix::<F, INVERSE>(
            src,
            dst,
            prev_len,
            b_out,
            groups_out,
            radices[0],
            twiddles[0],
            pointwise_spectrum,
        );
        return;
    }

    let outer_r = radices[n_stages - 1];
    let inner_radices = &radices[..n_stages - 1];
    let inner_r_total: usize = inner_radices.iter().product();
    let inner_out_len = inner_r_total * prev_len;
    let total_mid = outer_r * inner_out_len;
    let inner_groups_out = outer_r * groups_out;

    let (mid, rest) = scratch.split_at_mut(total_mid);

    for (b_inner, mid_chunk) in mid.chunks_exact_mut(inner_out_len).enumerate() {
        let b_inner_global = b_out + b_inner * groups_out;
        composite_fused_adaptive_inner::<F, INVERSE>(
            src,
            mid_chunk,
            rest,
            prev_len,
            b_inner_global,
            inner_groups_out,
            inner_radices,
            &twiddles[..n_stages - 1],
            None,
        );
    }

    dispatch_single_radix::<F, INVERSE>(
        mid,
        dst,
        inner_out_len,
        0,
        1,
        outer_r,
        twiddles[n_stages - 1],
        pointwise_spectrum,
    );
}

pub(super) fn composite_fused_adaptive<
    F: CompositeCache + ShortWinogradScalar,
    const INVERSE: bool,
>(
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    prev_len: usize,
    b_out: usize,
    groups_out: usize,
    radices: &[usize],
    twiddles: &[&[Complex<F>]],
    pointwise: Option<&[Complex<F>]>,
) {
    debug_assert_eq!(radices.len(), twiddles.len());
    if radices.is_empty() {
        return;
    }
    if radices.len() == 1 {
        dispatch_single_radix::<F, INVERSE>(
            src,
            dst,
            prev_len,
            b_out,
            groups_out,
            radices[0],
            twiddles[0],
            pointwise,
        );
        return;
    }

    let scratch_needed = composite_adaptive_scratch_size_elems(radices, prev_len);
    let (scratch_ptr, saved_top) = COMPOSE_ARENA.with(|cell| unsafe {
        let arena = &mut *cell.get();
        arena.reserve::<F>(scratch_needed);
        arena.alloc::<F>(scratch_needed)
    });
    let _guard = ArenaGuard(saved_top);
    let scratch: &mut [Complex<F>] =
        unsafe { core::slice::from_raw_parts_mut(scratch_ptr, scratch_needed) };

    composite_fused_adaptive_inner::<F, INVERSE>(
        src, dst, scratch, prev_len, b_out, groups_out, radices, twiddles, pointwise,
    );
}

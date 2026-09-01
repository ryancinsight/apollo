use eunomia::Complex;

use super::arity::dispatch_single_radix;
use super::cache::CompositeCache;
use crate::application::execution::kernel::components::winograd::ShortWinogradScalar;

/// Thread-local bump arena for the fused composite's per-group scratch.
///
/// # Soundness invariant
///
/// `alloc` hands out a raw pointer into `buf`'s heap allocation, and the
/// caller turns it into a `&mut [Complex<F>]` with `from_raw_parts_mut`
/// while the arena's own `&mut` borrow (taken inside the `thread_local`
/// closure) is already gone. That slice stays valid because nothing touches
/// `buf` again while it lives: `reserve` and `alloc` both complete before
/// the slice exists, `dealloc` (run by the [`ArenaGuard`]) only rewinds
/// `top`, and a *nested* composite — the only way a second `alloc` could
/// grow `buf` and move the allocation under a live slice — is asserted
/// against with `top == 0` at every resize. The assert is the trip-wire, not
/// the design.
///
/// # Owned-allocation fallback
///
/// If nested composites ever become legal, replace the growable `Vec<u8>`
/// with storage whose allocations never move: either return an owned
/// `Box<[Complex<F>]>` per group (one allocation per call, no aliasing to
/// reason about) or a chunked arena that appends new chunks and never
/// reallocates old ones. Until then the pattern is covered by the unit tests
/// below, run natively and under miri.
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

    /// Byte offset of the next `count`-lane block and the bump position after
    /// it. Alignment is taken on the buffer's *address*, not its offset: a
    /// `Vec<u8>` only promises byte alignment, so an offset-aligned block can
    /// still hand out a misaligned `&mut [Complex<F>]` (miri rejects exactly
    /// that; real allocators happen to return wider-aligned blocks).
    #[inline]
    fn block_layout<F>(&self, count: usize) -> (usize, usize) {
        let align = core::mem::align_of::<Complex<F>>();
        let base = self.buf.as_ptr().addr();
        let aligned = ((base + self.top + align - 1) & !(align - 1)) - base;
        (
            aligned,
            aligned + count * core::mem::size_of::<Complex<F>>(),
        )
    }

    /// Grows the buffer for a `count`-lane block, plus one alignment of slack
    /// so the address-aligned block still fits wherever the new allocation
    /// lands. Growth is only legal with no live inner pointers.
    #[inline]
    fn grow_for<F>(&mut self, count: usize) {
        let (_, needed) = self.block_layout::<F>(count);
        if self.buf.len() < needed {
            assert_eq!(self.top, 0, "arena realloc with live inner pointers");
            let slack = needed + core::mem::align_of::<Complex<F>>();
            let cap = (count * core::mem::size_of::<Complex<F>>() * 2)
                .next_power_of_two()
                .max(slack);
            self.buf.resize(cap, 0u8);
        }
    }

    #[inline]
    unsafe fn reserve<F>(&mut self, count: usize) {
        self.grow_for::<F>(count);
    }

    #[inline]
    unsafe fn alloc<F>(&mut self, count: usize) -> (*mut Complex<F>, usize) {
        self.grow_for::<F>(count);
        let (aligned, needed) = self.block_layout::<F>(count);
        debug_assert!(needed <= self.buf.len());
        // SAFETY: `grow_for` sized the buffer for `needed` bytes past the
        // address-aligned offset, so the pointer stays inside the allocation.
        let ptr = unsafe { self.buf.as_mut_ptr().add(aligned) }.cast::<Complex<F>>();
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
    #[expect(
        clippy::missing_const_for_thread_local,
        reason = "lazy initialization preserves the composite-radix benchmark path"
    )]
    static COMPOSE_ARENA: core::cell::UnsafeCell<ComposeArena> =
        core::cell::UnsafeCell::new(ComposeArena::new());
}

struct ArenaGuard(usize);

impl Drop for ArenaGuard {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `dealloc` only rewinds the bump pointer; the slice handed
        // out for this reservation is dead by the time the guard drops.
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
    // SAFETY: the thread-local cell is borrowed only inside this closure,
    // which does not re-enter the arena; see the `ComposeArena` invariant.
    let (scratch_ptr, saved_top) = COMPOSE_ARENA.with(|cell| unsafe {
        let arena = &mut *cell.get();
        arena.reserve::<F>(scratch_needed);
        arena.alloc::<F>(scratch_needed)
    });
    let _guard = ArenaGuard(saved_top);
    // SAFETY: `alloc` returned an aligned pointer to `scratch_needed`
    // initialized (zeroed) lanes inside the arena's allocation, and the
    // `ComposeArena` invariant guarantees that allocation neither moves nor
    // is handed out again before `_guard` rewinds it after this slice's last
    // use; the arena's `&mut` borrow ended with the closure above.
    let scratch: &mut [Complex<F>] =
        unsafe { core::slice::from_raw_parts_mut(scratch_ptr, scratch_needed) };

    composite_fused_adaptive_inner::<F, INVERSE>(
        src, dst, scratch, prev_len, b_out, groups_out, radices, twiddles, pointwise,
    );
}

#[cfg(test)]
mod arena_tests {
    use super::{ArenaGuard, ComposeArena, COMPOSE_ARENA};
    use eunomia::Complex;

    /// The `from_raw_parts_mut` pattern exactly as the composite uses it:
    /// reserve, alloc, build the slice after the arena borrow ends, write
    /// and read through it, and let the guard rewind. Under miri this is the
    /// aliasing check the production path has no other witness for.
    #[test]
    fn slice_over_the_arena_round_trips_and_rewinds() {
        let count = 37usize;
        // SAFETY: the thread-local borrow is confined to this closure and
        // the arena is not re-entered inside it (the production pattern).
        let (ptr, saved) = COMPOSE_ARENA.with(|cell| unsafe {
            let arena = &mut *cell.get();
            arena.reserve::<f64>(count);
            arena.alloc::<f64>(count)
        });
        // SAFETY: a shared view of the bump pointer only; no slice over the
        // arena's buffer exists yet.
        let top_after_alloc = COMPOSE_ARENA.with(|cell| unsafe { (*cell.get()).top });
        assert!(top_after_alloc >= count * core::mem::size_of::<Complex<f64>>());
        {
            let _guard = ArenaGuard(saved);
            // SAFETY: `ptr` addresses `count` zeroed lanes reserved above and
            // the arena is not touched until the guard rewinds after the
            // slice's last use — the `ComposeArena` invariant.
            let scratch: &mut [Complex<f64>] =
                unsafe { core::slice::from_raw_parts_mut(ptr, count) };
            for (index, lane) in scratch.iter_mut().enumerate() {
                *lane = Complex::new(index as f64, -(index as f64));
            }
            let sum: f64 = scratch.iter().map(|value| value.re - value.im).sum();
            assert_eq!(sum, (count * (count - 1)) as f64);
        }
        // SAFETY: as above; the slice and guard are gone.
        let top_after_drop = COMPOSE_ARENA.with(|cell| unsafe { (*cell.get()).top });
        assert_eq!(
            top_after_drop, saved,
            "the guard rewinds to its reservation"
        );
    }

    /// Two sequential reservations reuse the same region: the rewind makes
    /// the second allocation land where the first did.
    #[test]
    fn sequential_reservations_reuse_the_region() {
        let first = {
            // SAFETY: closure-confined arena borrow, no re-entry.
            let (ptr, saved) = COMPOSE_ARENA.with(|cell| unsafe {
                let arena = &mut *cell.get();
                arena.reserve::<f32>(16);
                arena.alloc::<f32>(16)
            });
            let _guard = ArenaGuard(saved);
            ptr as usize
        };
        let second = {
            // SAFETY: as for `first`; the previous guard already rewound.
            let (ptr, saved) = COMPOSE_ARENA.with(|cell| unsafe {
                let arena = &mut *cell.get();
                arena.reserve::<f32>(16);
                arena.alloc::<f32>(16)
            });
            let _guard = ArenaGuard(saved);
            ptr as usize
        };
        assert_eq!(first, second);
    }

    /// The trip-wire: growing the arena while an inner reservation is live
    /// would move the allocation under that reservation's slice, so `alloc`
    /// refuses it. Exercised on a private arena so the thread-local stays
    /// clean for the other tests.
    #[test]
    #[should_panic(expected = "arena realloc with live inner pointers")]
    fn growing_under_a_live_reservation_is_refused() {
        let mut arena = ComposeArena::new();
        // SAFETY: a private arena with no outstanding pointers; the second
        // reservation is expected to trip the realloc assert, not to be used.
        unsafe {
            arena.reserve::<f64>(8);
            let _inner = arena.alloc::<f64>(8);
            // A reservation far past the current capacity while `top != 0`.
            arena.reserve::<f64>(1 << 20);
        }
    }
}

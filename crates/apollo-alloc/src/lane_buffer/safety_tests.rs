use std::cell::Cell;
use std::mem::align_of;
use std::num::NonZeroU8;

use bytemuck::Zeroable;

use super::LaneBuffer;

#[repr(align(8192))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Overaligned(u8);

// SAFETY: every u8 representation is valid; alignment adds no value invariant.
unsafe impl Zeroable for Overaligned {}

#[test]
fn backing_guarantees_element_alignment() {
    for len in [0, 1, 3] {
        let buffer = LaneBuffer::<Overaligned>::zeros(len, 0);
        assert_eq!(buffer.allocation.layout.align(), align_of::<Overaligned>());
        assert_eq!(buffer.addr() % align_of::<Overaligned>(), 0);
        assert_eq!(buffer.as_slice(), vec![Overaligned(0); len]);
    }
    let source = [Overaligned(7), Overaligned(13)];
    let buffer = LaneBuffer::from_slice(&source, 0);
    assert_eq!(buffer.addr() % align_of::<Overaligned>(), 0);
    assert_eq!(buffer.as_slice(), source);
}

#[test]
fn copies_values_without_zero_validity() {
    let source = [
        NonZeroU8::new(7).expect("nonzero"),
        NonZeroU8::new(13).expect("nonzero"),
    ];
    let mut buffer = LaneBuffer::from_slice(&source, 512);
    assert_eq!(buffer.as_slice(), source);
    buffer.as_mut_slice()[0] = source[1];
    assert_eq!(buffer.as_slice(), &[source[1], source[1]]);
    assert_eq!(source[0].get(), 7);

    let references = [&source[0], &source[1]];
    let buffer = LaneBuffer::from_slice(&references, 1024);
    assert!(std::ptr::eq(buffer.as_slice()[0], &raw const source[0]));
    assert!(std::ptr::eq(buffer.as_slice()[1], &raw const source[1]));

    let buffer = LaneBuffer::<NonZeroU8>::from_slice(&[], 3072);
    assert_eq!(buffer.as_slice().len(), 0);
    assert_eq!(buffer.addr() % 4096, 3072);
}

thread_local! {
    static DROPS: Cell<usize> = const { Cell::new(0) };
}

struct Tracked(u8);
// SAFETY: zero is a valid u8; the destructor accepts every stored value.
unsafe impl Zeroable for Tracked {}
impl Drop for Tracked {
    fn drop(&mut self) {
        DROPS.set(DROPS.get() + 1);
        assert_ne!(self.0, 1, "element destructor test");
    }
}

#[test]
fn owned_elements_are_dropped() {
    DROPS.set(0);
    let mut buffer = LaneBuffer::<Tracked>::zeros(3, 512);
    assert_eq!(
        buffer.as_slice().iter().map(|x| x.0).collect::<Vec<_>>(),
        [0; 3]
    );
    buffer.as_mut_slice()[1].0 = 2;
    drop(buffer);
    assert_eq!(DROPS.get(), 3);
}

#[test]
fn destructor_unwinding_drops_the_tail_and_frees_storage() {
    DROPS.set(0);
    let mut buffer = LaneBuffer::<Tracked>::zeros(3, 512);
    buffer.as_mut_slice()[0].0 = 1;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(buffer))).is_err());
    assert_eq!(DROPS.get(), 3);
    // Miri's leak check also verifies that backing storage was freed.
}

struct ZeroSized;
// SAFETY: a unit struct has one valid representation and no stored bytes.
unsafe impl Zeroable for ZeroSized {}
impl Drop for ZeroSized {
    fn drop(&mut self) {
        DROPS.set(DROPS.get() + 1);
    }
}

#[test]
fn zero_sized_elements_are_initialized_and_dropped() {
    DROPS.set(0);
    let mut buffer = LaneBuffer::<ZeroSized>::zeros(5, 3072);
    assert_eq!(buffer.as_mut_slice().len(), 5);
    assert_eq!(buffer.addr() % 4096, 3072);
    drop(buffer);
    assert_eq!(DROPS.get(), 5);
}

#[test]
fn rejects_invalid_lanes_and_sizes_before_allocating() {
    for lane in [1, 4095, 4096, usize::MAX] {
        assert!(std::panic::catch_unwind(|| LaneBuffer::<u64>::zeros(1, lane)).is_err());
        assert!(std::panic::catch_unwind(|| LaneBuffer::from_slice(&[7u64], lane)).is_err());
    }
    assert!(std::panic::catch_unwind(|| LaneBuffer::<Overaligned>::zeros(0, 512)).is_err());
    for len in [usize::MAX, usize::MAX / 8, usize::try_from(isize::MAX).expect("invariant: isize::MAX fits usize") / 8] {
        assert!(std::panic::catch_unwind(|| LaneBuffer::<u64>::zeros(len, 0)).is_err());
    }
}

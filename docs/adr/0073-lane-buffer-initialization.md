# ADR 0073: Lane buffer initialization and ownership

Status: Accepted

## Contract

`LaneBuffer<T>` exposes exactly `len` initialized, exclusively owned elements.
Its slice address equals the requested lane modulo 4096 and is aligned for `T`.
The backing layout is nonempty, bounds the complete slice, and is reused for
deallocation. Every owned element is dropped before backing storage is freed,
including zero-sized elements and unwinding from an element destructor.

## Decision

Require the existing `bytemuck::Zeroable` safety contract only on `zeros`.
`Default` does not mean all-zero bytes are valid and cannot justify this API.
`from_slice` retains `T: Copy` and copies directly into raw storage before any
typed reference is formed; it therefore accepts nonzero integers and references.
Use `max(4096, align_of::<T>())` for the allocation, with the lane still required
to be a multiple of `align_of::<T>()`. Over-aligned types can use lane zero.
Keep backing storage in a separate owner so element-drop unwinding frees it.

Rejecting every over-aligned type would unnecessarily narrow the contract.
Requiring `Zeroable` on the buffer or copy constructor would likewise reject
valid copied values. A local zero-validity trait would duplicate bytemuck.

## Migration and verification

The unpublished pre-1.0 `apollo-alloc` advances from 0.16.0 to 0.17.0 because
`zeros` gains a bound. Use a verified `Zeroable` implementation or `from_slice`
with initialized values. Enable num-complex's bytemuck support for existing FFT
benchmark scratch types; the timed closures and lane choices stay identical.
Verify compile-fail zero constructors, valid nonzero/reference copies, layout
alignment, lane bounds, overflow, empty/ZST storage, mutation, destruction,
destructor unwinding, Miri, and the repository CI gates.

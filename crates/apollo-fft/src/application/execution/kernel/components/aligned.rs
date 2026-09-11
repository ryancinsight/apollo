//! The cache-line-aligned wrapper hot tables and staging buffers sit in.

/// `A` placed on a 64-byte boundary. The allocator places a `Box` at 16
/// bytes and the compiler a stack array at its scalar's alignment, so
/// whether a 32-byte vector load from a table or the staging buffer
/// splits a cache line is the heap's or the frame's luck: the pinned probe
/// read the same kernel 8 to 30% apart through two allocations of one
/// table. The wrapper makes the boundary a type fact.
#[repr(C, align(64))]
pub(crate) struct CacheLineAligned<A>(pub(crate) A);

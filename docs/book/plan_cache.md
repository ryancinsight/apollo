# Plans, tables, and workspace

An FFT plan records how a transform of a particular shape executes. Reusing
it avoids repeating route selection and allows immutable coefficient tables
to remain shared. Workspace has a different lifetime: it holds intermediate
values during execution and must remain disjoint from every active input.

## Reusing a plan

`FftPlan1D::<T>::new(Shape1D)` constructs a reusable plan. The corresponding
2-D and 3-D constructors accept `Shape2D` and `Shape3D`. These shapes validate
nonzero dimensions before execution. Forward and inverse methods share the
plan; direction is not a separate plan-cache key.

The array API resolves cached plans through `PlanCacheProvider` on the storage
scalar. Its `get_1d_plan`, `get_2d_plan`, and `get_3d_plan` methods return `Arc`
handles. Explicitly constructing a plan does not insert that plan into these
caches.

Each scalar and dimension has two caches, both bounded (ADR 0068). A
process-wide table holds at most 64 shapes. A thread's own ring holds the
four it used most recently, so a repeated or alternating shape is served
without taking the table's lock and without allocating. The ring and the
table share one slot per plan, so a use the ring serves still counts as a
use for eviction.

Past 64 shapes the table evicts the least recently used. Recency is measured
at the granularity of misses: the table's tick advances once per plan built,
which is what keeps a ring hit free of any read-modify-write, so plans used
since the last build carry the same stamp and tie. The victim among tied
plans is one whose slot no ring holds -- between ticks that is the only
evidence of use the table has -- and position order decides only when every
tied slot is equally held.

Retained plan memory is therefore bounded: at most 64 plans per scalar and
dimension, plus four in each live thread's ring.

`apollo_fft::clear_plan_caches()` releases what the caches alone hold. It
empties every shared table and the calling thread's rings; each other
thread's ring empties at that thread's next lookup through it. A plan a
caller still holds stays alive through its `Arc` and is not re-cached.

`StaticFftPlan1D<T, N>`, `StaticFftPlan2D<T, NX, NY>`, and
`StaticFftPlan3D<T, NX, NY, NZ>` encode shape in const generics. Their values are
zero-sized. Their kernels still use coefficient tables and temporary storage;
a zero-sized plan does not imply a zero-memory transform.

## Immutable coefficient tables

FourStep's planar kernels share process-wide stage tables and twiddle planes.
A table's key includes length and direction. Each worker keeps only the last
length's handle for each direction, in fixed inline storage. A first visit to
a worker can therefore acquire an existing table without allocating a local
map. Alternating lengths replace handles and consult the shared table cache;
the globally owned coefficients remain valid throughout execution.

## Temporary values

Mnemosyne supplies Apollo's reusable caller-thread transpose buffers. Leto
performs layout movement and Moirai joins disjoint lane tasks before those
buffers can be reused.

For generic FourStep lanes longer than 1024, an axis pass uses its inactive
transpose companion as lane workspace. If a lane needs more scratch elements
than its length, one task groups enough adjacent lanes to supply that scratch
and reuses it sequentially. The last incomplete group runs after the parallel
join. This keeps the existing full-volume storage bound. A degenerate volume
that cannot supply one workspace uses the existing local execution route.

Moirai workers release Apollo FFT scratch when idle. The enclosing caller's
buffer survives across submissions, so worker quiescence does not discard the
lane workspace. Strided views use a separate rank-specific staging role to
keep their logical values disjoint from transpose scratch. Smaller sized
kernels retain their own scratch and dispatch policies.

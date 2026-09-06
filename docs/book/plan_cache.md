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
handles. The process caches retain plans by shape for each scalar implementation;
thread-local handles avoid repeated shared lookups. Explicitly constructing a
plan does not insert that plan into these caches.

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

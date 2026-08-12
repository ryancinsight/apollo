# Plan Cache

Apollo caches FFT execution plans to avoid repeating the expensive
mixed-radix factorization on every call.

## Runtime-Cached Plans

`FftPlan1D`, `FftPlan2D`, `FftPlan3D` are created once per unique
transform size and cached globally:

```rust,ignore
use apollo_fft::FftPlan1D;

// Plan is computed and cached on first use; subsequent calls hit the cache
let plan = FftPlan1D::get_or_create(4096)?;
let out = plan.forward(&input)?;
```

The cache key is `(size, scalar_type, direction)`.

## `PlanCacheProvider` Trait

Backends that maintain their own plan cache implement `PlanCacheProvider`:

```rust,ignore
pub trait PlanCacheProvider {
    fn get_plan_1d(&self, size: usize) -> Option<Arc<FftPlan1D>>;
    fn insert_plan_1d(&self, size: usize, plan: Arc<FftPlan1D>);
}
```

The wgpu backend's `GpuTransformPlanner` caches compiled WGSL shader
pipelines by `(shape, precision_profile, direction)`, avoiding
shader recompilation across training iterations.

## `Shape1D / 2D / 3D`

Validated non-zero shape structs prevent zero-size FFT plans from reaching
the execution path. All plan constructors require a `Shape*D`:

```rust,ignore
let shape = Shape1D::new(4096)?;  // errors if size == 0
let plan = FftPlan1D::new(shape)?;
```

## Static vs. Dynamic Plans

| Plan type | Cache hit | Compile-time size | Use case |
|-----------|-----------|-------------------|----------|
| `StaticFftPlan1D<N>` | never needed | yes | Fixed-size inner loops |
| `FftPlan1D` (cached) | yes | no | Variable-size; amortized |
| `FftPlan1D` (fresh) | no | no | One-shot calls |

# Apollo Leto Interop

`apollo-leto-interop` owns Apollo's shared boundary between borrowed Leto views
and dense Mnemosyne-backed outputs. Contiguous views stay borrowed; strided
views materialize once in logical row-major order. Slice constructors copy into
Mnemosyne storage. Vector constructors move elements into separately allocated
Mnemosyne storage and release the source allocation.

```rust
use apollo_leto_interop::try_dense_from_slice;

let output = try_dense_from_slice([2, 2], &[1_u32, 2, 3, 4])
    .expect("shape cardinality matches the values");
assert_eq!(output.as_slice(), Some(&[1, 2, 3, 4][..]));
```

See the [API documentation](https://docs.rs/apollo-leto-interop) and the
[Apollo repository](https://github.com/ryancinsight/apollo).

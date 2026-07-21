//! Unified macros for the two-level (thread-local + global) cache pattern used
//! across the complex-type caches in this module.
//!
//! Every complex-type cache follows the same structure:
//! 1. Two `static LazyLock<RwLock<FxHashMap<K, V>>>` globals (one per precision)
//! 2. Two thread-local stores per precision: a sparse `FxHashMap` and, where
//!    the key domain is bounded, a heap-backed flat table
//! 3. A sealed marker trait + a Store trait with `tl_get`/`tl_insert`/`global`
//! 4. Identical `impl` blocks for `Complex64` and `Complex32`
//! 5. A `cached_*` function with TL-then-global-then-build logic
//!
//! `declare_cache_store!` handles steps 3–4 (the sealed trait, Store trait,
//! and both impl blocks). Each cache file keeps its own statics (step 1–2)
//! and its own cached function (step 5), which may use the companion
//! `cached_fetch_arc!` macro for the common `Arc<[C]>` + closure pattern.
//! Flat tables use `const`-initialized fixed-size TLS arrays, so first access
//! has no runtime initializer stack frame while hot lookup retains direct TLS
//! storage and compile-time lengths for bounds-check elimination.
//!
//! Uses `FxHashMap` (from rustc_hash) for faster hashing of integer keys.

/// Generates: the sealed module with marker trait, the `Store` trait with
/// `tl_get`/`tl_insert`/`global` methods, and both `Complex64`/`Complex32`
/// impl blocks.
///
/// # Parameters
///
/// * `sealed_mod` / `sealed_trait` — module and trait names for the sealed marker.
/// * `store_trait` — name of the public `Store` trait.
/// * `extra_bounds` — additional supertrait bounds as token trees
///   (e.g. `[Clone, 'static]` since `'static` is a lifetime, not a path).
/// * `key` — the key type (e.g. `usize`, `(usize, usize)`).
/// * `val64` / `val32` — concrete value types for each precision.
/// * `val_self` — the value type using `Self` (for the trait definition).
/// * `tl_get` / `tl_insert` / `global` — method names.
/// * `global_ret_self` — return type of `global()` using `Self`.
/// * `tl64` / `tl32` — thread-local static names.
/// * `global64` / `global32` — global static names.
#[macro_export]
macro_rules! declare_cache_store {
    (
        sealed_mod: $sealed_mod:ident,
        sealed_trait: $sealed_trait:ident,
        store_trait: $store_trait:ident,
        extra_bounds: [$($bound:tt),* $(,)?],
        key: $key_ty:ty,
        val_precise: $val_precise:ty,
        val_reduced: $val_reduced:ty,
        val_self: $val_self:ty,
        tl_get: $tl_get:ident,
        tl_insert: $tl_insert:ident,
        global: $global:ident,
        global_ret_self: $global_ret_self:ty,
        tl_precise: $tl_precise:ident,
        tl_reduced: $tl_reduced:ident,
        global_precise: $global_precise:ident,
        global_reduced: $global_reduced:ident,
    ) => {
        mod $sealed_mod {
            pub(crate) trait $sealed_trait {}
        }

        pub(crate) trait $store_trait: $($bound +)* $sealed_mod::$sealed_trait {
            fn $tl_get(key: $key_ty) -> Option<$val_self>;
            fn $tl_insert(key: $key_ty, v: $val_self);
            fn $global() -> &'static $global_ret_self;
        }

        impl $sealed_mod::$sealed_trait for eunomia::Complex64 {}
        impl $store_trait for eunomia::Complex64 {
            #[inline]
            fn $tl_get(key: $key_ty) -> Option<$val_precise> {
                $tl_precise.with(|c| c.borrow().get(&key).cloned())
            }
            #[inline]
            fn $tl_insert(key: $key_ty, v: $val_precise) {
                $tl_precise.with(|c| {
                    c.borrow_mut().insert(key, v);
                });
            }
            #[inline]
            fn $global() -> &'static $global_ret_self {
                &$global_precise
            }
        }

        impl $sealed_mod::$sealed_trait for eunomia::Complex32 {}
        impl $store_trait for eunomia::Complex32 {
            #[inline]
            fn $tl_get(key: $key_ty) -> Option<$val_reduced> {
                $tl_reduced.with(|c| c.borrow().get(&key).cloned())
            }
            #[inline]
            fn $tl_insert(key: $key_ty, v: $val_reduced) {
                $tl_reduced.with(|c| {
                    c.borrow_mut().insert(key, v);
                });
            }
            #[inline]
            fn $global() -> &'static $global_ret_self {
                &$global_reduced
            }
        }
    };

    (
        sealed_mod: $sealed_mod:ident,
        sealed_trait: $sealed_trait:ident,
        store_trait: $store_trait:ident,
        extra_bounds: [$($bound:tt),* $(,)?],
        key: $key_ty:ty,
        val_precise: $val_precise:ty,
        val_reduced: $val_reduced:ty,
        val_self: $val_self:ty,
        tl_get: $tl_get:ident,
        tl_insert: $tl_insert:ident,
        global: $global:ident,
        global_ret_self: $global_ret_self:ty,
        tl_precise: $tl_precise:ident,
        tl_reduced: $tl_reduced:ident,
        global_precise: $global_precise:ident,
        global_reduced: $global_reduced:ident,
        tl_precise_flat: $tl_precise_flat:ident,
        tl_reduced_flat: $tl_reduced_flat:ident,
        flat_check: $flat_check:expr,
        flat_idx: $flat_idx:expr,
    ) => {
        mod $sealed_mod {
            pub(crate) trait $sealed_trait {}
        }

        pub(crate) trait $store_trait: $($bound +)* $sealed_mod::$sealed_trait {
            fn $tl_get(key: $key_ty) -> Option<$val_self>;
            fn $tl_insert(key: $key_ty, v: $val_self);
            fn $global() -> &'static $global_ret_self;
        }

        impl $sealed_mod::$sealed_trait for eunomia::Complex64 {}
        impl $store_trait for eunomia::Complex64 {
            #[inline]
            fn $tl_get(key: $key_ty) -> Option<$val_precise> {
                if ($flat_check)(key) {
                    let idx = ($flat_idx)(key);
                    $tl_precise_flat.with(|c| c.borrow()[idx].clone())
                } else {
                    $tl_precise.with(|c| c.borrow().get(&key).cloned())
                }
            }
            #[inline]
            fn $tl_insert(key: $key_ty, v: $val_precise) {
                if ($flat_check)(key) {
                    let idx = ($flat_idx)(key);
                    $tl_precise_flat.with(|c| c.borrow_mut()[idx] = Some(v));
                } else {
                    $tl_precise.with(|c| {
                        c.borrow_mut().insert(key, v);
                    });
                }
            }
            #[inline]
            fn $global() -> &'static $global_ret_self {
                &$global_precise
            }
        }

        impl $sealed_mod::$sealed_trait for eunomia::Complex32 {}
        impl $store_trait for eunomia::Complex32 {
            #[inline]
            fn $tl_get(key: $key_ty) -> Option<$val_reduced> {
                if ($flat_check)(key) {
                    let idx = ($flat_idx)(key);
                    $tl_reduced_flat.with(|c| c.borrow()[idx].clone())
                } else {
                    $tl_reduced.with(|c| c.borrow().get(&key).cloned())
                }
            }
            #[inline]
            fn $tl_insert(key: $key_ty, v: $val_reduced) {
                if ($flat_check)(key) {
                    let idx = ($flat_idx)(key);
                    $tl_reduced_flat.with(|c| c.borrow_mut()[idx] = Some(v));
                } else {
                    $tl_reduced.with(|c| {
                        c.borrow_mut().insert(key, v);
                    });
                }
            }
            #[inline]
            fn $global() -> &'static $global_ret_self {
                &$global_reduced
            }
        }
    };
}

/// Generates a `cached_*` function for the common pattern where:
/// - The stored value is `Arc<[C]>`
/// - A build closure `impl FnOnce(K) -> Vec<C>` is passed by the caller
///
/// # Parameters
///
/// * `fn_vis` / `fn_name` — visibility and name of the generated function.
/// * `store_trait` — the `Store` trait bound.
/// * `key` — the key type.
/// * `tl_get` / `tl_insert` / `global` — method names on the Store trait.
#[macro_export]
macro_rules! cached_fetch_arc {
    (
        fn $fn_vis:vis $fn_name:ident<$store_trait:ident>(
            $key_pat:ident : $key_ty:ty,
            build_fn: $build_generic:ty,
        ) -> Arc<[F]>
        using tl_get = $tl_get:ident, tl_insert = $tl_insert:ident, global = $global:ident,
    ) => {
        #[inline]
        $fn_vis fn $fn_name<F: $store_trait>(
            $key_pat: $key_ty,
            build_fn: impl FnOnce($key_ty) -> Vec<F>,
        ) -> std::sync::Arc<[F]> {
            if let Some(v) = F::$tl_get($key_pat) {
                return v;
            }
            let v = {
                let maybe = F::$global().read().get(&$key_pat).cloned();
                if let Some(v) = maybe {
                    v
                } else {
                    let new_v: std::sync::Arc<[F]> = std::sync::Arc::from(build_fn($key_pat));
                    F::$global()
                        .write()
                        .entry($key_pat)
                        .or_insert_with(|| std::sync::Arc::clone(&new_v))
                        .clone()
                }
            };
            F::$tl_insert($key_pat, std::sync::Arc::clone(&v));
            v
        }
    };
}

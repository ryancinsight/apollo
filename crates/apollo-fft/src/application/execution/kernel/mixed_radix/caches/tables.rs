//! The two table shapes every cache family here is built from.

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::LazyLock;

/// A process-wide cache table behind a read-write lock, built on first use.
pub(super) type SharedTable<K, V> = LazyLock<RwLock<FxHashMap<K, V>>>;

/// A thread-local cache table.
pub(super) type LocalTable<K, V> = RefCell<FxHashMap<K, V>>;

/// The empty shared table, for a `static` initializer.
pub(super) const fn shared_table<K, V>() -> SharedTable<K, V> {
    LazyLock::new(|| RwLock::new(FxHashMap::default()))
}

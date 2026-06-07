//! Branded permission cells for Apollo execution workspaces.
//!
//! The crate separates mutable permission from storage. A [`GhostCell`] stores
//! data; a [`GhostToken`] carries the unique permission needed to obtain
//! mutable access to all cells in one brand. This removes runtime borrow checks
//! from execution-scope ownership graphs while preserving Rust's aliasing rule.
//!
//! [`LocalGhostCell`] is the intentionally narrower primitive for thread-local
//! statics. It is unsafe to enter because static storage can be re-entered; the
//! caller must prove that no live reference from the same cell exists.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::UnsafeCell;
use std::marker::PhantomData;

type Invariant<'brand> = PhantomData<fn(&'brand ()) -> &'brand ()>;

/// Unique permission for one GhostCell brand.
///
/// A token is created only by [`GhostToken::scope`]. The higher-ranked closure
/// prevents the brand lifetime from escaping, so cells and references created
/// inside the scope cannot be returned with their brand attached.
pub struct GhostToken<'brand> {
    brand: Invariant<'brand>,
}

impl GhostToken<'static> {
    /// Runs `f` with a fresh brand token.
    #[inline]
    pub fn scope<R>(f: impl for<'brand> FnOnce(&mut GhostToken<'brand>) -> R) -> R {
        let mut token = GhostToken { brand: PhantomData };
        f(&mut token)
    }
}

/// Storage governed by a [`GhostToken`] brand.
///
/// ## Invariant
///
/// `borrow_mut` requires `&mut GhostToken<'brand>`. Rust permits only one live
/// mutable borrow of that token, so no second mutable or immutable borrow
/// through the same brand can coexist with the returned `&mut T`.
pub struct GhostCell<'brand, T> {
    value: UnsafeCell<T>,
    brand: Invariant<'brand>,
}

impl<'brand, T> GhostCell<'brand, T> {
    /// Creates a branded cell.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
            brand: PhantomData,
        }
    }

    /// Consumes the cell and returns the stored value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.value.into_inner()
    }

    /// Returns mutable access through an exclusive borrow of the cell itself.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.value.get_mut()
    }

    /// Borrows the stored value immutably under the matching brand.
    #[inline]
    pub fn borrow<'a>(&'a self, _token: &'a GhostToken<'brand>) -> &'a T {
        // SAFETY: immutable access is shared and the matching token prevents a
        // simultaneous mutable borrow from existing through this API.
        unsafe { &*self.value.get() }
    }

    /// Borrows the stored value mutably under the matching brand.
    #[inline]
    pub fn borrow_mut<'a>(&'a self, _token: &'a mut GhostToken<'brand>) -> &'a mut T {
        // SAFETY: the returned mutable reference is tied to the unique mutable
        // borrow of the brand token. A second borrow through the same brand
        // cannot be created while this reference is live.
        unsafe { &mut *self.value.get() }
    }
}

// SAFETY: shared access to a GhostCell can produce shared `&T`, so `T` must be
// Sync. Mutable access still requires unique token ownership.
unsafe impl<'brand, T: Sync> Sync for GhostCell<'brand, T> {}
unsafe impl<'brand, T: Send> Send for GhostCell<'brand, T> {}

/// Permission token for a [`LocalGhostCell`] root.
///
/// This token is constructed only inside `LocalGhostCell::with_token`.
pub struct LocalGhostToken<'brand> {
    brand: Invariant<'brand>,
}

/// Zero-overhead permission cell for thread-local roots.
///
/// Unlike [`GhostCell`], this type has no brand parameter because thread-local
/// statics must have a concrete type. Entering the root is unsafe; after entry,
/// mutation still requires the local token.
pub struct LocalGhostCell<T> {
    value: UnsafeCell<T>,
}

impl<T> LocalGhostCell<T> {
    /// Creates a local permission cell.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
        }
    }

    /// Enters a local permission root.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that:
    /// - this cell is not re-entered while any reference derived from a
    ///   previous call remains live;
    /// - the supplied token is used only with the `branded` cell argument;
    /// - the closure does not call back into the same local root.
    ///
    /// These conditions hold for non-reentrant thread-local execution roots
    /// that borrow their state for one transform and do not expose the token.
    #[inline]
    pub unsafe fn with_token<R>(
        &self,
        f: impl for<'brand> FnOnce(&mut LocalGhostToken<'brand>, &'brand LocalGhostCell<T>) -> R,
    ) -> R {
        let mut token = LocalGhostToken { brand: PhantomData };
        f(&mut token, self)
    }

    /// Borrows the stored value immutably under the local token.
    #[inline]
    pub fn borrow<'a, 'brand>(&'a self, _token: &'a LocalGhostToken<'brand>) -> &'a T {
        // SAFETY: immutable access is shared and mutation requires a mutable
        // local token borrow.
        unsafe { &*self.value.get() }
    }

    /// Borrows the stored value mutably under the local token.
    #[inline]
    pub fn borrow_mut<'a, 'brand>(&'a self, _token: &'a mut LocalGhostToken<'brand>) -> &'a mut T {
        // SAFETY: the unsafe root entry proves non-reentrancy for this local
        // cell, and the mutable token prevents concurrent borrows in the scope.
        unsafe { &mut *self.value.get() }
    }
}

// SAFETY: LocalGhostCell is intended for thread-local roots. If moved as a
// value, sending it requires sending the contained value.
unsafe impl<T: Send> Send for LocalGhostCell<T> {}

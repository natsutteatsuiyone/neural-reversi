//! Fixed-length, over-aligned heap buffer for SIMD and cache-line access.
//!
//! [`AlignedBuffer<T, ALIGN>`] owns a heap allocation whose base address is a
//! multiple of `ALIGN` bytes. Unlike [`Vec`], it has no spare capacity and
//! cannot grow: every buffer in the engine is sized once at load time and
//! then only read or overwritten in place, so the length *is* the capacity.
//! Dropping the growth machinery keeps the type a thin pointer + length pair
//! and lets SIMD code rely on the alignment of `as_ptr()` for aligned loads
//! (`_mm256_load_si256`, `_mm512_load_si512`, …).
//!
//! `ALIGN` must be a power of two that is at least `align_of::<T>()`; both
//! conditions are checked at compile time.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::fmt;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// Heap buffer of `len` `T` values whose start address is `ALIGN`-aligned.
pub struct AlignedBuffer<T, const ALIGN: usize> {
    /// Allocation base. `ALIGN`-aligned; dangling (never dereferenced) when
    /// the buffer holds zero bytes.
    ptr: NonNull<T>,
    /// Number of initialized `T` values. Equals the allocation length.
    len: usize,
}

impl<T, const ALIGN: usize> AlignedBuffer<T, ALIGN> {
    /// Compile-time guard: `ALIGN` is a power of two and covers `T`.
    const VALID_ALIGN: () = {
        assert!(ALIGN.is_power_of_two(), "ALIGN must be a power of two");
        assert!(
            ALIGN >= mem::align_of::<T>(),
            "ALIGN must be at least align_of::<T>()"
        );
    };

    /// Allocates `len` uninitialized, `ALIGN`-aligned slots.
    ///
    /// Returns the base pointer and the [`Layout`] used (needed for
    /// deallocation). For an empty buffer no allocation is performed and a
    /// dangling-but-aligned pointer is returned with a zero-sized layout.
    fn alloc_uninit(len: usize) -> (NonNull<T>, Layout) {
        let () = Self::VALID_ALIGN;

        let size = len
            .checked_mul(mem::size_of::<T>())
            .expect("AlignedBuffer: capacity overflow");
        let layout = Layout::from_size_align(size, ALIGN).expect("AlignedBuffer: invalid layout");

        if size == 0 {
            // Strict-provenance form (not an `int as *mut T` cast) so Miri
            // can still flag real pointer bugs elsewhere.
            let dangling = std::ptr::without_provenance_mut::<T>(ALIGN);
            return (NonNull::new(dangling).unwrap(), layout);
        }

        // SAFETY: `layout` has non-zero size.
        let raw = unsafe { alloc(layout) } as *mut T;
        let ptr = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));
        (ptr, layout)
    }

    /// Layout of the current allocation, for [`dealloc`].
    ///
    /// `len` equals the allocated element count for any successfully
    /// constructed buffer, so this reproduces the allocation layout.
    fn layout(&self) -> Layout {
        let size = self.len * mem::size_of::<T>();
        // Already validated at allocation time.
        Layout::from_size_align(size, ALIGN).unwrap()
    }

    /// Creates a buffer of `len` elements, each a clone of `value`.
    pub fn from_elem(value: T, len: usize) -> Self
    where
        T: Clone,
    {
        let (ptr, _) = Self::alloc_uninit(len);
        let mut fill = Filling::<T, ALIGN> {
            ptr,
            cap: len,
            initialized: 0,
        };
        let base = ptr.as_ptr();
        for i in 0..len {
            // SAFETY: `i < len`; slot is allocated and uninitialized.
            unsafe { base.add(i).write(value.clone()) };
            fill.initialized = i + 1;
        }
        fill.into_buffer()
    }

    /// Creates a buffer from an iterator of known length.
    ///
    /// The iterator must yield exactly `iter.len()` items.
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut it = iter.into_iter();
        let len = it.len();
        let (ptr, _) = Self::alloc_uninit(len);
        let mut fill = Filling::<T, ALIGN> {
            ptr,
            cap: len,
            initialized: 0,
        };
        let base = ptr.as_ptr();
        for i in 0..len {
            let item = it
                .next()
                .expect("AlignedBuffer::from_iter: iterator shorter than ExactSizeIterator::len()");
            // SAFETY: `i < len`; slot is allocated and uninitialized.
            unsafe { base.add(i).write(item) };
            fill.initialized = i + 1;
        }
        fill.into_buffer()
    }

    /// Number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Raw const pointer to the first element. `ALIGN`-aligned.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Slice view over the whole buffer.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Mutable slice view over the whole buffer.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

/// Owns the allocation while it is being filled.
///
/// If a panic unwinds before [`Filling::into_buffer`] is called, its `Drop`
/// drops the `initialized` prefix and frees the full `cap` allocation, so the
/// `dealloc` layout always matches the original `alloc`.
struct Filling<T, const ALIGN: usize> {
    ptr: NonNull<T>,
    /// Allocated element count (the full allocation).
    cap: usize,
    /// Elements written so far.
    initialized: usize,
}

impl<T, const ALIGN: usize> Filling<T, ALIGN> {
    /// Hands the completed allocation to an [`AlignedBuffer`], cancelling the
    /// cleanup guard.
    fn into_buffer(self) -> AlignedBuffer<T, ALIGN> {
        debug_assert_eq!(self.initialized, self.cap);
        let ptr = self.ptr;
        let len = self.cap;
        mem::forget(self);
        AlignedBuffer { ptr, len }
    }
}

impl<T, const ALIGN: usize> Drop for Filling<T, ALIGN> {
    fn drop(&mut self) {
        // SAFETY: the first `initialized` slots hold valid `T` values.
        unsafe {
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.initialized,
            ))
        };
        let size = self.cap * mem::size_of::<T>();
        if size != 0 {
            let layout = Layout::from_size_align(size, ALIGN).unwrap();
            // SAFETY: `ptr`/`layout` come from the matching allocation.
            unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) };
        }
    }
}

impl<T, const ALIGN: usize> Deref for AlignedBuffer<T, ALIGN> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` points to `len` initialized, contiguous `T` values.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T, const ALIGN: usize> DerefMut for AlignedBuffer<T, ALIGN> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: `ptr` points to `len` initialized, contiguous `T` values
        // and `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T, const ALIGN: usize> Drop for AlignedBuffer<T, ALIGN> {
    fn drop(&mut self) {
        let layout = self.layout();
        // SAFETY: the first `self.len` slots are initialized.
        unsafe {
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.len,
            ))
        };
        if layout.size() != 0 {
            // SAFETY: `ptr`/`layout` come from the matching allocation.
            unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) };
        }
    }
}

impl<T: Clone, const ALIGN: usize> Clone for AlignedBuffer<T, ALIGN> {
    fn clone(&self) -> Self {
        Self::from_iter(self.iter().cloned())
    }
}

impl<T: fmt::Debug, const ALIGN: usize> fmt::Debug for AlignedBuffer<T, ALIGN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

// SAFETY: `AlignedBuffer` owns a unique heap allocation; sending/sharing it is
// sound exactly when sending/sharing the contained `T` values is.
unsafe impl<T: Send, const ALIGN: usize> Send for AlignedBuffer<T, ALIGN> {}
unsafe impl<T: Sync, const ALIGN: usize> Sync for AlignedBuffer<T, ALIGN> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::CACHE_LINE_SIZE;

    #[test]
    fn from_elem_len_and_values() {
        let v = AlignedBuffer::<i32, CACHE_LINE_SIZE>::from_elem(7, 100);
        assert_eq!(v.len(), 100);
        assert!(v.iter().all(|&x| x == 7));
    }

    #[test]
    fn base_pointer_is_aligned() {
        let v = AlignedBuffer::<i16, CACHE_LINE_SIZE>::from_elem(0, 257);
        assert_eq!(v.as_ptr() as usize % CACHE_LINE_SIZE, 0);

        let v8 = AlignedBuffer::<i8, CACHE_LINE_SIZE>::from_elem(0, 1);
        assert_eq!(v8.as_ptr() as usize % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn from_iter_matches_source() {
        let v = AlignedBuffer::<usize, CACHE_LINE_SIZE>::from_iter(0..50);
        assert_eq!(v.len(), 50);
        for (i, &x) in v.iter().enumerate() {
            assert_eq!(i, x);
        }
    }

    #[test]
    fn mutation_through_deref() {
        let mut v = AlignedBuffer::<i32, CACHE_LINE_SIZE>::from_elem(0, 8);
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = i as i32;
        }
        assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);
        v[0] = 42;
        assert_eq!(v[0], 42);
    }

    #[test]
    fn clone_is_independent() {
        let mut a = AlignedBuffer::<i32, CACHE_LINE_SIZE>::from_elem(1, 16);
        let b = a.clone();
        a[0] = 999;
        assert_eq!(b[0], 1);
        assert_eq!(b.as_ptr() as usize % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn empty_buffer_is_safe() {
        let v = AlignedBuffer::<i64, CACHE_LINE_SIZE>::from_elem(0, 0);
        assert_eq!(v.len(), 0);
        assert_eq!(v.as_slice(), &[] as &[i64]);
        assert_eq!(v.as_ptr() as usize % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn debug_matches_slice() {
        let v = AlignedBuffer::<i32, CACHE_LINE_SIZE>::from_iter([1, 2, 3]);
        assert_eq!(format!("{v:?}"), "[1, 2, 3]");
    }
}

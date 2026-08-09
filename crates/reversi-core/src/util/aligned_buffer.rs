//! Fixed-length, cache-line-aligned buffer for SIMD access.
//!
//! [`AlignedBuffer<T>`] owns a fixed-length allocation whose base address is a
//! multiple of [`CACHE_LINE_SIZE`] bytes. Unlike [`Vec`], it has no spare
//! capacity and cannot grow: every buffer in the engine is sized once at load
//! time and then only read or overwritten in place, so the length *is* the
//! capacity. SIMD code can rely on the alignment of `as_ptr()` for aligned
//! loads (`_mm256_load_si256`, `_mm512_load_si512`, …).
//!
//! `CACHE_LINE_SIZE` must cover `align_of::<T>()`; checked at compile time.

use std::alloc::{Layout, alloc, alloc_zeroed, dealloc, handle_alloc_error};
use std::fmt;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use crate::constants::CACHE_LINE_SIZE;
use crate::util::large_pages;

/// A fixed-length buffer of `len` `T` values whose start address is cache-line-aligned.
pub struct AlignedBuffer<T> {
    /// Allocation base. `CACHE_LINE_SIZE`-aligned; dangling (never
    /// dereferenced) when the buffer holds zero bytes.
    ptr: NonNull<T>,
    /// Number of initialized `T` values. Equals the allocation length.
    len: usize,
    /// Whether the allocation came from [`large_pages`] and must be released
    /// through it instead of the global allocator.
    large_pages: bool,
}

impl<T> AlignedBuffer<T> {
    /// Compile-time guard: the cache-line alignment covers `T`.
    const VALID_ALIGN: () = assert!(
        CACHE_LINE_SIZE >= mem::align_of::<T>(),
        "CACHE_LINE_SIZE must be at least align_of::<T>()"
    );

    /// Allocates `len` cache-line-aligned slots, preferring large pages.
    ///
    /// Returns the block and whether it came from [`large_pages`], which the
    /// matching [`AlignedBuffer::release`] needs. Large-page blocks are always
    /// zeroed; the global-allocator fallback only when `zeroed` asks for it.
    ///
    /// For an empty buffer no allocation is performed and a
    /// dangling-but-aligned pointer is returned.
    fn alloc_raw(len: usize, zeroed: bool) -> (NonNull<T>, bool) {
        let () = Self::VALID_ALIGN;

        let size = len
            .checked_mul(mem::size_of::<T>())
            .expect("AlignedBuffer: capacity overflow");
        if size == 0 {
            // Strict-provenance form (not an `int as *mut T` cast) so Miri
            // can still flag real pointer bugs elsewhere.
            let dangling = std::ptr::without_provenance_mut::<T>(CACHE_LINE_SIZE);
            return (NonNull::new(dangling).unwrap(), false);
        }

        // Windows aligns the returned address to its large-page minimum.
        if let Some(ptr) = large_pages::alloc_zeroed(size) {
            return (ptr.cast(), true);
        }

        let layout =
            Layout::from_size_align(size, CACHE_LINE_SIZE).expect("AlignedBuffer: invalid layout");

        // SAFETY: `layout` has non-zero size.
        let raw = unsafe {
            if zeroed {
                alloc_zeroed(layout)
            } else {
                alloc(layout)
            }
        } as *mut T;
        (
            NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout)),
            false,
        )
    }

    /// Frees a block from [`AlignedBuffer::alloc_raw`].
    ///
    /// # Safety
    ///
    /// `ptr`, `len` and `from_large_pages` must describe that allocation, and
    /// every element must already have been dropped.
    unsafe fn release(ptr: NonNull<T>, len: usize, from_large_pages: bool) {
        if from_large_pages {
            // SAFETY: the block came from `large_pages::alloc_zeroed`.
            unsafe { large_pages::free(ptr.cast()) };
            return;
        }
        let size = len * mem::size_of::<T>();
        if size != 0 {
            // Already validated at allocation time.
            let layout = Layout::from_size_align(size, CACHE_LINE_SIZE).unwrap();
            // SAFETY: `ptr`/`layout` come from the matching allocation.
            unsafe { dealloc(ptr.as_ptr() as *mut u8, layout) };
        }
    }

    /// Creates a buffer of `len` zeroed elements.
    ///
    /// Unlike [`AlignedBuffer::from_elem`] this never writes the elements, so a
    /// large-page buffer stays as the kernel handed it over.
    ///
    /// # Safety
    ///
    /// An all-zero byte pattern must be a valid value of `T`.
    pub unsafe fn zeroed(len: usize) -> Self {
        let (ptr, large_pages) = Self::alloc_raw(len, true);
        AlignedBuffer {
            ptr,
            len,
            large_pages,
        }
    }

    /// Creates a buffer of `len` elements, each a clone of `value`.
    pub fn from_elem(value: T, len: usize) -> Self
    where
        T: Clone,
    {
        Self::from_iter(std::iter::repeat_n(value, len))
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
        let (ptr, large_pages) = Self::alloc_raw(len, false);
        let mut fill = Filling {
            ptr,
            cap: len,
            initialized: 0,
            large_pages,
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

    /// Returns the number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns a raw const pointer to the first element.
    ///
    /// `CACHE_LINE_SIZE`-aligned.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a slice view over the whole buffer.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Returns a mutable slice view over the whole buffer.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

/// Owner of the allocation while it is being filled.
///
/// If a panic unwinds before [`Filling::into_buffer`] is called, its `Drop`
/// drops the `initialized` prefix and frees the full `cap` allocation, so the
/// `dealloc` layout always matches the original `alloc`.
struct Filling<T> {
    ptr: NonNull<T>,
    /// Allocated element count (the full allocation).
    cap: usize,
    /// Elements written so far.
    initialized: usize,
    /// Whether the allocation came from [`large_pages`].
    large_pages: bool,
}

impl<T> Filling<T> {
    /// Hands the completed allocation to an [`AlignedBuffer`], cancelling the
    /// cleanup guard.
    fn into_buffer(self) -> AlignedBuffer<T> {
        debug_assert_eq!(self.initialized, self.cap);
        let ptr = self.ptr;
        let len = self.cap;
        let large_pages = self.large_pages;
        mem::forget(self);
        AlignedBuffer {
            ptr,
            len,
            large_pages,
        }
    }
}

impl<T> Drop for Filling<T> {
    fn drop(&mut self) {
        // SAFETY: the first `initialized` slots hold valid `T` values, and the
        // allocation is described by `cap`/`large_pages`.
        unsafe {
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.initialized,
            ));
            AlignedBuffer::release(self.ptr, self.cap, self.large_pages);
        };
    }
}

impl<T> Deref for AlignedBuffer<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` points to `len` initialized, contiguous `T` values.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for AlignedBuffer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: `ptr` points to `len` initialized, contiguous `T` values
        // and `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: the first `self.len` slots are initialized, and the
        // allocation is described by `len`/`large_pages`.
        unsafe {
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.len,
            ));
            Self::release(self.ptr, self.len, self.large_pages);
        };
    }
}

impl<T: Clone> Clone for AlignedBuffer<T> {
    fn clone(&self) -> Self {
        Self::from_iter(self.iter().cloned())
    }
}

impl<T: fmt::Debug> fmt::Debug for AlignedBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

// SAFETY: `AlignedBuffer` owns a unique heap allocation; sending/sharing it is
// sound exactly when sending/sharing the contained `T` values is.
unsafe impl<T: Send> Send for AlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for AlignedBuffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_elem_len_and_values() {
        let v = AlignedBuffer::<i32>::from_elem(7, 100);
        assert_eq!(v.len(), 100);
        assert!(v.iter().all(|&x| x == 7));
    }

    #[test]
    fn base_pointer_is_aligned() {
        let v = AlignedBuffer::<i16>::from_elem(0, 257);
        assert_eq!(v.as_ptr() as usize % CACHE_LINE_SIZE, 0);

        let v8 = AlignedBuffer::<i8>::from_elem(0, 1);
        assert_eq!(v8.as_ptr() as usize % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn from_iter_matches_source() {
        let v = AlignedBuffer::<usize>::from_iter(0..50);
        assert_eq!(v.len(), 50);
        for (i, &x) in v.iter().enumerate() {
            assert_eq!(i, x);
        }
    }

    #[test]
    fn mutation_through_deref() {
        let mut v = AlignedBuffer::<i32>::from_elem(0, 8);
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = i as i32;
        }
        assert_eq!(v.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);
        v[0] = 42;
        assert_eq!(v[0], 42);
    }

    #[test]
    fn clone_is_independent() {
        let mut a = AlignedBuffer::<i32>::from_elem(1, 16);
        let b = a.clone();
        a[0] = 999;
        assert_eq!(b[0], 1);
        assert_eq!(b.as_ptr() as usize % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn empty_buffer_is_safe() {
        let v = AlignedBuffer::<i64>::from_elem(0, 0);
        assert_eq!(v.len(), 0);
        assert_eq!(v.as_slice(), &[] as &[i64]);
        assert_eq!(v.as_ptr() as usize % CACHE_LINE_SIZE, 0);
    }
}

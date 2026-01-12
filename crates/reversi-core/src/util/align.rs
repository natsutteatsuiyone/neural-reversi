//! 64-byte aligned wrapper for SIMD and cache optimization.

use std::fmt;
use std::ops::{Deref, DerefMut, Index, IndexMut};

/// Wrapper type ensuring 64-byte alignment for SIMD performance.
#[repr(C, align(64))]
pub struct Align64<T>(pub T);

impl<T> Align64<T> {
    /// Returns a raw pointer to the wrapped value.
    #[allow(dead_code)]
    pub fn as_ptr(&self) -> *const T {
        &self.0 as *const T
    }

    /// Returns a mutable raw pointer to the wrapped value.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        &mut self.0 as *mut T
    }
}

impl<T> Deref for Align64<T> {
    type Target = T;

    /// Dereferences the aligned wrapper to access the inner value.
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Align64<T> {
    /// Mutably dereferences the aligned wrapper to access the inner value.
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Clone> Clone for Align64<T> {
    /// Creates a copy of the aligned wrapper by cloning the inner value.
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Copy> Copy for Align64<T> {}

impl<T, I> Index<I> for Align64<T>
where
    T: Index<I>,
{
    type Output = T::Output;

    /// Provides indexed access to the wrapped value.
    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, I> IndexMut<I> for Align64<T>
where
    T: IndexMut<I>,
{
    /// Provides mutable indexed access to the wrapped value.
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T, const N: usize> Align64<[T; N]> {
    /// Returns a slice view of the aligned array.
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// Returns a mutable slice view of the aligned array.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T: Default> Default for Align64<T> {
    /// Creates an aligned wrapper with the default value of the inner type.
    fn default() -> Self {
        Self(T::default())
    }
}

impl<T: fmt::Debug> fmt::Debug for Align64<T> {
    /// Formats the aligned wrapper by delegating to the inner value's Debug implementation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment() {
        let aligned = Align64([0u8; 32]);
        let ptr = aligned.as_ptr() as usize;
        assert_eq!(ptr % 64, 0, "Align64 should provide 64-byte alignment");
    }

    #[test]
    fn test_deref() {
        let aligned = Align64(42);
        assert_eq!(*aligned, 42);
    }

    #[test]
    fn test_deref_mut() {
        let mut aligned = Align64(42);
        *aligned = 100;
        assert_eq!(*aligned, 100);
    }

    #[test]
    fn test_clone() {
        let original = Align64(vec![1, 2, 3]);
        let cloned = original.clone();
        assert_eq!(*original, *cloned);
    }

    #[test]
    fn test_copy() {
        let original = Align64(42);
        let copied = original;
        assert_eq!(*original, *copied);
    }

    #[test]
    fn test_index() {
        let aligned = Align64([1, 2, 3, 4, 5]);
        assert_eq!(aligned[0], 1);
        assert_eq!(aligned[2], 3);
        assert_eq!(aligned[4], 5);
    }

    #[test]
    fn test_index_mut() {
        let mut aligned = Align64([1, 2, 3, 4, 5]);
        aligned[2] = 10;
        assert_eq!(aligned[2], 10);
    }

    #[test]
    fn test_as_slice() {
        let aligned = Align64([1, 2, 3, 4, 5]);
        let slice = aligned.as_slice();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_as_mut_slice() {
        let mut aligned = Align64([1, 2, 3, 4, 5]);
        let slice = aligned.as_mut_slice();
        slice[2] = 10;
        assert_eq!(aligned[2], 10);
    }

    #[test]
    fn test_default() {
        let aligned: Align64<i32> = Align64::default();
        assert_eq!(*aligned, 0);

        let aligned_vec: Align64<Vec<i32>> = Align64::default();
        assert!(aligned_vec.is_empty());
    }

    #[test]
    fn test_debug() {
        let aligned = Align64(42);
        assert_eq!(format!("{:?}", aligned), "42");

        let aligned_vec = Align64(vec![1, 2, 3]);
        assert_eq!(format!("{:?}", aligned_vec), "[1, 2, 3]");
    }

    #[test]
    fn test_pointers() {
        let mut aligned = Align64([1, 2, 3, 4]);

        // Test const pointer
        let ptr = aligned.as_ptr();
        unsafe {
            assert_eq!((*ptr)[0], 1);
        }

        // Test mut pointer
        let mut_ptr = aligned.as_mut_ptr();
        unsafe {
            (*mut_ptr)[0] = 10;
        }
        assert_eq!(aligned[0], 10);
    }
}

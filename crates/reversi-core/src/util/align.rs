//! 64-byte-aligned wrapper for SIMD and cache-line optimization.

use std::fmt;
use std::ops::{Deref, DerefMut, Index, IndexMut};

/// Wrapper type ensuring 64-byte alignment for SIMD operations and cache line optimization.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Align64<T>(pub T);

impl<T> Align64<T> {
    /// Returns a raw pointer to the wrapped value.
    ///
    /// The returned pointer is guaranteed to be 64-byte aligned.
    #[allow(dead_code)]
    pub fn as_ptr(&self) -> *const T {
        &self.0 as *const T
    }

    /// Returns a mutable raw pointer to the wrapped value.
    ///
    /// The returned pointer is guaranteed to be 64-byte aligned.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        &mut self.0 as *mut T
    }
}

impl<T> Deref for Align64<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Align64<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, I> Index<I> for Align64<T>
where
    T: Index<I>,
{
    type Output = T::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, I> IndexMut<I> for Align64<T>
where
    T: IndexMut<I>,
{
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
    fn default() -> Self {
        Self(T::default())
    }
}

impl<T: fmt::Debug> fmt::Debug for Align64<T> {
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
}

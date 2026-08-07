//! 64-byte-aligned wrapper for SIMD and cache-line optimization.

use std::fmt;
use std::ops::{Deref, DerefMut};

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

use std::fmt;
use std::ops::{Deref, DerefMut, Index, IndexMut};

/// A wrapper type that ensures 64-byte alignment for optimal SIMD performance.
///
/// This type is used throughout the neural network evaluation code to ensure
/// that arrays and buffers are properly aligned for AVX2 instructions and
/// cache line optimization.
#[repr(C, align(64))]
pub struct Align64<T>(pub T);

impl<T> Align64<T> {
    /// Returns a raw pointer to the wrapped value.
    ///
    /// This is commonly used for SIMD operations that require aligned pointers.
    pub fn as_ptr(&self) -> *const T {
        &self.0 as *const T
    }

    /// Returns a mutable raw pointer to the wrapped value.
    ///
    /// This is commonly used for SIMD operations that require aligned pointers.
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

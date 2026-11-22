//! 64-byte-aligned wrapper for SIMD and cache-line optimization.

use std::fmt;
use std::ops::{Deref, DerefMut};

/// Wrapper type ensuring 64-byte alignment for SIMD operations and cache line optimization.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Align64<T>(pub T);

impl<T> Align64<T> {
    /// Returns a mutable raw pointer to the wrapped value.
    ///
    /// The returned pointer is guaranteed to be 64-byte aligned.
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        &mut self.0 as *mut T
    }
}

impl<const N: usize> Align64<[u8; N]> {
    #[inline(always)]
    pub fn sub_array<const OFFSET: usize, const LEN: usize>(&self) -> &Align64<[u8; LEN]> {
        const {
            assert!(OFFSET.is_multiple_of(64));
            assert!(matches!(OFFSET.checked_add(LEN), Some(end) if end <= N));
            assert!(std::mem::size_of::<Align64<[u8; LEN]>>() == LEN);
        }
        // SAFETY: the assertions above keep the aligned, padding-free subarray inside `self`.
        unsafe { &*self.0.as_ptr().add(OFFSET).cast::<Align64<[u8; LEN]>>() }
    }

    #[inline(always)]
    pub fn sub_array_mut<const OFFSET: usize, const LEN: usize>(
        &mut self,
    ) -> &mut Align64<[u8; LEN]> {
        const {
            assert!(OFFSET.is_multiple_of(64));
            assert!(matches!(OFFSET.checked_add(LEN), Some(end) if end <= N));
            assert!(std::mem::size_of::<Align64<[u8; LEN]>>() == LEN);
        }
        // SAFETY: the assertions above keep the aligned, padding-free subarray inside `self`.
        unsafe { &mut *self.0.as_mut_ptr().add(OFFSET).cast::<Align64<[u8; LEN]>>() }
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
    fn debug_delegates_to_inner_value() {
        assert_eq!(format!("{:?}", Align64(42)), "42");
    }

    #[test]
    fn sub_array_preserves_alignment_and_isolates_adjacent_bytes() {
        let mut buf = Align64([0xAAu8; 288]);
        let base = buf.as_ptr().addr();

        {
            let first = buf.sub_array_mut::<0, 128>();
            assert_eq!(first.as_ptr().addr(), base);
            assert_eq!(first.as_ptr().addr() % 64, 0);
            first[0] = 0x11;
            first[127] = 0x22;
        }
        {
            let second = buf.sub_array_mut::<128, 128>();
            assert_eq!(second.as_ptr().addr(), base + 128);
            assert_eq!(second.as_ptr().addr() % 64, 0);
            assert_eq!(second[0], 0xAA);
            second[0] = 0x33;
            second[127] = 0x44;
        }

        let combined = buf.sub_array::<0, 256>();
        assert_eq!(combined.as_ptr().addr(), base);
        assert_eq!(combined.as_ptr().addr() % 64, 0);
        assert_eq!(
            (combined[0], combined[127], combined[128], combined[255]),
            (0x11, 0x22, 0x33, 0x44)
        );
        assert!(buf[256..].iter().all(|&b| b == 0xAA));
    }
}

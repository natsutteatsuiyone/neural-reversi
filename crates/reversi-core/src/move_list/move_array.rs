//! Fixed-capacity inline storage for [`Move`] values.

use std::cmp::Reverse;
use std::fmt;
use std::mem::{MaybeUninit, offset_of};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr;
use std::slice;

use super::{MAX_MOVES, Move};

// `data_ptr_from` casts `*mut MoveArray` straight to `*mut Move`; pin the
// offset-0 layout it relies on at compile time.
const _: () = assert!(offset_of!(MoveArray, data) == 0);

/// Fixed-capacity storage specialized for Reversi legal moves.
///
/// The maximum move count is a small game invariant, so this stores moves inline
/// and tracks the length next to the buffer. `Move` is `Copy`, which lets retain/clone
/// operate by value without drop bookkeeping.
#[repr(C)]
pub(super) struct MoveArray {
    data: [MaybeUninit<Move>; MAX_MOVES],
    pub(super) len: usize,
}

impl MoveArray {
    #[inline]
    pub(super) fn new() -> Self {
        MoveArray {
            data: [MaybeUninit::uninit(); MAX_MOVES],
            len: 0,
        }
    }

    #[inline(always)]
    pub(super) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub(super) fn as_ptr(&self) -> *const Move {
        self.data.as_ptr().cast()
    }

    #[inline(always)]
    pub(super) fn as_mut_ptr(&mut self) -> *mut Move {
        self.data.as_mut_ptr().cast()
    }

    /// Returns a raw pointer to the inline `Move` buffer of a `MoveArray`
    /// reached through a raw pointer to a possibly-uninitialised slot.
    ///
    /// # Safety
    /// `arr` must be a valid, properly-aligned pointer to a `MoveArray`.
    #[inline(always)]
    pub(super) unsafe fn data_ptr_from(arr: *mut MoveArray) -> *mut Move {
        arr as *mut Move
    }

    /// Returns a raw pointer to the `len` field of a `MoveArray` reached
    /// through a raw pointer to a possibly-uninitialised slot.
    ///
    /// # Safety
    /// `arr` must be a valid, properly-aligned pointer to a `MoveArray`.
    #[inline(always)]
    pub(super) unsafe fn len_ptr_from(arr: *mut MoveArray) -> *mut usize {
        unsafe { &raw mut (*arr).len }
    }

    #[inline(always)]
    pub(super) fn as_slice(&self) -> &[Move] {
        // SAFETY: `0..self.len` is always initialized.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    #[inline(always)]
    pub(super) fn as_mut_slice(&mut self) -> &mut [Move] {
        // SAFETY: `0..self.len` is always initialized; `&mut self` ensures unique access.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    #[inline(always)]
    pub(super) unsafe fn get_unchecked(&self, index: usize) -> Move {
        debug_assert!(index < self.len);
        // SAFETY: caller ensures `index < self.len`.
        unsafe { *self.as_ptr().add(index) }
    }

    #[inline(always)]
    pub(super) unsafe fn get_unchecked_ref(&self, index: usize) -> &Move {
        debug_assert!(index < self.len);
        // SAFETY: caller ensures `index < self.len`.
        unsafe { &*self.as_ptr().add(index) }
    }

    #[inline(always)]
    pub(super) unsafe fn swap_unchecked(&mut self, a: usize, b: usize) {
        debug_assert!(a < self.len);
        debug_assert!(b < self.len);
        let ptr = self.as_mut_ptr();
        // SAFETY: caller ensures `a, b < self.len`.
        unsafe { ptr::swap(ptr.add(a), ptr.add(b)) };
    }

    #[inline(always)]
    unsafe fn compare_swap_unchecked(&mut self, a: usize, b: usize) {
        debug_assert!(a < self.len);
        debug_assert!(b < self.len);

        let ptr = self.as_mut_ptr();
        // SAFETY: caller ensures `a, b < self.len`.
        unsafe {
            let a_ptr = ptr.add(a);
            let b_ptr = ptr.add(b);
            if (*a_ptr).value < (*b_ptr).value {
                ptr::swap(a_ptr, b_ptr);
            }
        }
    }

    #[inline]
    pub(super) fn sort_by_value_desc(&mut self) {
        match self.len {
            0 | 1 => {}
            2 => {
                // SAFETY: match arm guarantees `len == 2`.
                unsafe { self.compare_swap_unchecked(0, 1) };
            }
            3 => {
                // SAFETY: match arm guarantees `len == 3`.
                unsafe {
                    self.compare_swap_unchecked(0, 1);
                    self.compare_swap_unchecked(1, 2);
                    self.compare_swap_unchecked(0, 1);
                }
            }
            _ => self
                .as_mut_slice()
                .sort_unstable_by_key(|mv| Reverse(mv.value)),
        }
    }

    #[inline]
    pub(super) fn retain(&mut self, mut keep: impl FnMut(Move) -> bool) {
        let len = self.len;
        let mut write = 0;
        let ptr = self.as_mut_ptr();

        for read in 0..len {
            // SAFETY: `read < len`.
            let mv = unsafe { *ptr.add(read) };
            if keep(mv) {
                if write != read {
                    // SAFETY: `write <= read < len`.
                    unsafe { ptr.add(write).write(mv) };
                }
                write += 1;
            }
        }

        self.len = write;
    }
}

impl Clone for MoveArray {
    #[inline]
    fn clone(&self) -> Self {
        let mut cloned = MoveArray::new();
        let len = self.len();
        if len != 0 {
            // SAFETY: source `0..len` is initialized; `cloned` owns a separate buffer.
            unsafe { ptr::copy_nonoverlapping(self.as_ptr(), cloned.as_mut_ptr(), len) };
        }
        cloned.len = len;
        cloned
    }
}

impl fmt::Debug for MoveArray {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl Default for MoveArray {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for MoveArray {
    type Target = [Move];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for MoveArray {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Index<usize> for MoveArray {
    type Output = Move;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl IndexMut<usize> for MoveArray {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

//! Utility types and functions for low-level operations.
//!
//! Provides memory alignment wrappers, atomic bitsets, spinlocks,
//! and helper functions used throughout the engine's hot paths.

pub mod align;
pub mod bitset;
pub mod spinlock;

/// Returns the high 64 bits of `a * b`.
///
/// Used for fast-range hash-to-index mapping: `mul_hi64(key, n)` maps
/// a 64-bit hash uniformly to `[0, n)` without division.
#[inline(always)]
pub fn mul_hi64(a: u64, b: u64) -> u64 {
    (((a as u128) * (b as u128)) >> 64) as u64
}

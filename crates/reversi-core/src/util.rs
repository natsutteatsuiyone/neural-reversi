//! Utility types and functions for low-level operations.
//!
//! Provides memory alignment wrappers, atomic bitsets, spinlocks,
//! and helper functions used throughout the engine's hot paths.

pub mod align;
pub mod bitset;
pub mod spinlock;

/// Performs unchecked array indexing, bypassing bounds checks.
///
/// Supports multi-dimensional access by chaining indices with commas.
///
/// # Safety
///
/// All indices must be within bounds. Out-of-bounds access causes
/// undefined behavior.
///
/// # Examples
///
/// ```
/// use reversi_core::uget;
///
/// let arr = [10, 20, 30];
/// let val = unsafe { *uget!(arr; 1) };
/// assert_eq!(val, 20);
///
/// let arr_2d = [[1, 2], [3, 4]];
/// let val = unsafe { *uget!(arr_2d; 0, 1) };
/// assert_eq!(val, 2);
/// ```
#[macro_export]
macro_rules! uget {
    ($arr:expr; $i:expr $(,)?) => {{
        #[allow(unused_unsafe)]
        #[allow(clippy::macro_metavars_in_unsafe)]
        unsafe {{ ($arr).get_unchecked($i) }}
    }};
    ($arr:expr; $i:expr, $($rest:expr),+ $(,)?) => {{
        let __p = {{
            #[allow(unused_unsafe)]
            #[allow(clippy::macro_metavars_in_unsafe)]
            unsafe {{ ($arr).get_unchecked($i) }}
        }};
        $crate::uget!(&*__p; $($rest),+)
    }};
}

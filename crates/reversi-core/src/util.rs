pub mod align;
pub mod bitset;
pub mod spinlock;

/// Rounds up a number to the nearest multiple of a base value.
///
/// # Arguments
///
/// * `n` - The number to round up
/// * `base` - The base value to round to
///
/// # Returns
///
/// The smallest multiple of `base` that is greater than or equal to `n`.
pub const fn ceil_to_multiple(n: usize, base: usize) -> usize {
    n.div_ceil(base) * base
}

/// Unsafe array access macro that skips bounds checking.
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

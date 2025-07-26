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

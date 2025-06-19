pub mod bitset;
pub mod spinlock;

pub const fn ceil_to_multiple(n: usize, base: usize) -> usize {
    n.div_ceil(base) * base
}

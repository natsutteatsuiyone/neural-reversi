/// A space-efficient bitset implementation using a 64-bit integer.
#[derive(Clone, Copy, Default)]
pub struct BitSet {
    data: u64,
}

impl BitSet {
    /// Creates a new empty bitset with all bits cleared.
    #[inline(always)]
    pub const fn new() -> Self {
        Self { data: 0 }
    }

    /// Returns the number of bits currently set.
    ///
    /// Returns the count of bits set to 1.
    #[inline(always)]
    pub const fn count(&self) -> u32 {
        self.data.count_ones()
    }

    /// Sets the bit at the specified index to 1.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the bit to set (0-based).
    #[inline(always)]
    pub fn set(&mut self, index: usize) {
        debug_assert!(index < 64);
        self.data |= 1 << index;
    }

    /// Clears the bit at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the bit to clear (0-based).
    #[inline(always)]
    pub fn reset(&mut self, index: usize) {
        debug_assert!(index < 64);
        self.data &= !(1 << index);
    }

    /// Tests whether the bit at the specified index is set.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the bit to test (0-based).
    ///
    /// Returns `true` if the bit is set, `false` otherwise.
    #[inline(always)]
    pub const fn test(&self, index: usize) -> bool {
        (self.data >> index) & 1 != 0
    }

    /// Checks if all bits are clear.
    ///
    /// Returns `true` if no bits are set, `false` otherwise.
    #[inline(always)]
    pub const fn none(&self) -> bool {
        self.data == 0
    }

    /// Clears all bits.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.data = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let bitset = BitSet::new();
        assert_eq!(bitset.count(), 0);
        assert!(bitset.none());
    }

    #[test]
    fn test_set_and_test() {
        let mut bitset = BitSet::new();

        bitset.set(0);
        assert!(bitset.test(0));
        assert_eq!(bitset.count(), 1);

        bitset.set(5);
        assert!(bitset.test(5));
        assert_eq!(bitset.count(), 2);

        bitset.set(63);
        assert!(bitset.test(63));
        assert_eq!(bitset.count(), 3);
    }

    #[test]
    fn test_reset() {
        let mut bitset = BitSet::new();

        bitset.set(10);
        assert!(bitset.test(10));

        bitset.reset(10);
        assert!(!bitset.test(10));
        assert_eq!(bitset.count(), 0);
    }

    #[test]
    fn test_count() {
        let mut bitset = BitSet::new();
        assert_eq!(bitset.count(), 0);

        for i in 0..10 {
            bitset.set(i);
        }
        assert_eq!(bitset.count(), 10);
    }

    #[test]
    fn test_none() {
        let mut bitset = BitSet::new();
        assert!(bitset.none());

        bitset.set(0);
        assert!(!bitset.none());

        bitset.reset(0);
        assert!(bitset.none());
    }

    #[test]
    fn test_clear() {
        let mut bitset = BitSet::new();

        bitset.set(1);
        bitset.set(5);
        bitset.set(20);
        assert_eq!(bitset.count(), 3);

        bitset.clear();
        assert_eq!(bitset.count(), 0);
        assert!(bitset.none());
    }

    #[test]
    fn test_multiple_operations() {
        let mut bitset = BitSet::new();

        bitset.set(0);
        bitset.set(1);
        bitset.set(31);
        bitset.set(32);
        bitset.set(63);

        assert_eq!(bitset.count(), 5);
        assert!(bitset.test(0));
        assert!(bitset.test(1));
        assert!(bitset.test(31));
        assert!(bitset.test(32));
        assert!(bitset.test(63));

        bitset.reset(1);
        bitset.reset(32);

        assert_eq!(bitset.count(), 3);
        assert!(!bitset.test(1));
        assert!(!bitset.test(32));
        assert!(bitset.test(0));
        assert!(bitset.test(31));
        assert!(bitset.test(63));
    }
}

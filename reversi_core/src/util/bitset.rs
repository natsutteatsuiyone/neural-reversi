/// A space-efficient bitset implementation using a 64-bit integer.
pub struct BitSet {
    /// The underlying 64-bit storage for the bitset
    data: u64,
    /// The number of bits currently set (maintained for O(1) count access)
    pub count: usize,
}

impl BitSet {
    /// Creates a new empty bitset with all bits cleared.
    pub fn new() -> BitSet {
        BitSet { data: 0, count: 0 }
    }

    /// Sets the bit at the specified index to 1.
    ///
    /// If the bit is already set, this operation has no effect.
    /// The count is only incremented if the bit was previously unset.
    ///
    /// # Arguments
    ///
    /// * `index` - The bit position to set (0-63)
    #[inline]
    pub fn set(&mut self, index: usize) {
        let mask = 1 << index;
        let old_data = self.data;
        self.data |= mask;
        self.count += (old_data != self.data) as usize;
    }

    /// Clears the bit at the specified index (sets it to 0).
    ///
    /// If the bit is already clear, this operation has no effect.
    /// The count is only decremented if the bit was previously set.
    ///
    /// # Arguments
    ///
    /// * `index` - The bit position to clear (0-63)
    #[inline]
    pub fn reset(&mut self, index: usize) {
        let mask = 1 << index;
        let old_data = self.data;
        self.data &= !mask;
        self.count -= (old_data != self.data) as usize;
    }

    /// Tests whether the bit at the specified index is set.
    ///
    /// # Arguments
    ///
    /// * `index` - The bit position to test (0-63)
    ///
    /// # Returns
    ///
    /// `true` if the bit is set (1), `false` if it's clear (0).
    #[inline]
    pub fn test(&self, index: usize) -> bool {
        (self.data & (1 << index)) != 0
    }

    /// Checks if all bits in the bitset are clear.
    ///
    /// # Returns
    ///
    /// `true` if no bits are set, `false` otherwise.
    #[inline]
    pub fn none(&self) -> bool {
        self.data == 0
    }

    /// Clears all bits in the bitset, resetting it to its initial empty state.
    ///
    /// After calling this method, `count` will be 0 and `none()` will return `true`.
    #[inline]
    pub fn clear(&mut self) {
        self.data = 0;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let bitset = BitSet::new();
        assert_eq!(bitset.count, 0);
        assert!(bitset.none());
        assert_eq!(bitset.data, 0);
    }

    #[test]
    fn test_set_single_bit() {
        let mut bitset = BitSet::new();

        bitset.set(0);
        assert!(bitset.test(0));
        assert_eq!(bitset.count, 1);
        assert!(!bitset.none());

        bitset.set(63);
        assert!(bitset.test(63));
        assert_eq!(bitset.count, 2);

        bitset.set(31);
        assert!(bitset.test(31));
        assert_eq!(bitset.count, 3);
    }

    #[test]
    fn test_set_same_bit_twice() {
        let mut bitset = BitSet::new();

        bitset.set(5);
        assert_eq!(bitset.count, 1);

        // Setting the same bit again should not change count
        bitset.set(5);
        assert_eq!(bitset.count, 1);
        assert!(bitset.test(5));
    }

    #[test]
    fn test_reset_bit() {
        let mut bitset = BitSet::new();

        bitset.set(10);
        bitset.set(20);
        assert_eq!(bitset.count, 2);

        bitset.reset(10);
        assert!(!bitset.test(10));
        assert!(bitset.test(20));
        assert_eq!(bitset.count, 1);

        bitset.reset(20);
        assert!(!bitset.test(20));
        assert_eq!(bitset.count, 0);
        assert!(bitset.none());
    }

    #[test]
    fn test_reset_unset_bit() {
        let mut bitset = BitSet::new();

        bitset.set(5);
        assert_eq!(bitset.count, 1);

        // Resetting an unset bit should not change count
        bitset.reset(6);
        assert_eq!(bitset.count, 1);
        assert!(bitset.test(5));
        assert!(!bitset.test(6));
    }

    #[test]
    fn test_test_method() {
        let mut bitset = BitSet::new();

        // Test all bits are initially false
        for i in 0..64 {
            assert!(!bitset.test(i));
        }

        // Set some bits and test
        bitset.set(0);
        bitset.set(7);
        bitset.set(15);
        bitset.set(31);
        bitset.set(63);

        assert!(bitset.test(0));
        assert!(bitset.test(7));
        assert!(bitset.test(15));
        assert!(bitset.test(31));
        assert!(bitset.test(63));

        // Test unset bits are still false
        assert!(!bitset.test(1));
        assert!(!bitset.test(8));
        assert!(!bitset.test(32));
    }

    #[test]
    fn test_none() {
        let mut bitset = BitSet::new();

        assert!(bitset.none());

        bitset.set(0);
        assert!(!bitset.none());

        bitset.set(63);
        assert!(!bitset.none());

        bitset.reset(0);
        assert!(!bitset.none());

        bitset.reset(63);
        assert!(bitset.none());
    }

    #[test]
    fn test_clear() {
        let mut bitset = BitSet::new();

        // Set multiple bits
        bitset.set(0);
        bitset.set(1);
        bitset.set(2);
        bitset.set(31);
        bitset.set(63);
        assert_eq!(bitset.count, 5);
        assert!(!bitset.none());

        // Clear all
        bitset.clear();
        assert_eq!(bitset.count, 0);
        assert!(bitset.none());

        // Verify all bits are cleared
        for i in 0..64 {
            assert!(!bitset.test(i));
        }
    }

    #[test]
    #[should_panic]
    fn test_set_out_of_bounds() {
        let mut bitset = BitSet::new();
        bitset.set(64); // Should panic on shift overflow
    }

    #[test]
    #[should_panic]
    fn test_reset_out_of_bounds() {
        let mut bitset = BitSet::new();
        bitset.reset(64); // Should panic on shift overflow
    }

    #[test]
    #[should_panic]
    fn test_test_out_of_bounds() {
        let bitset = BitSet::new();
        bitset.test(64); // Should panic on shift overflow
    }
}

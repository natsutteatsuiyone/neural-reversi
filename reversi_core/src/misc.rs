pub struct BitSet {
    data: u64,
    pub count: usize,
}

impl BitSet {
    pub fn new() -> BitSet {
        BitSet { data: 0, count: 0 }
    }

    #[inline]
    pub fn set(&mut self, index: usize) {
        let b = 1 << index;
        if self.data & b == 0 {
            self.data |= b;
            self.count += 1;
        }
    }

    #[inline]
    pub fn reset(&mut self, index: usize) {
        let b = 1 << index;
        if self.data & b != 0 {
            self.data ^= b;
            self.count -= 1;
        }
    }

    #[inline]
    pub fn test(&self, index: usize) -> bool {
        (self.data & (1 << index)) != 0
    }

    #[inline]
    pub fn none(&self) -> bool {
        self.data == 0
    }

    #[inline]
    pub fn clear(&mut self) {
        self.data = 0;
        self.count = 0;
    }
}

pub const fn ceil_to_multiple(n: usize, base: usize) -> usize {
    n.div_ceil(base) * base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset_operations() {
        let mut bs = BitSet::new();
        assert!(bs.none());

        bs.set(0);
        assert!(!bs.none());
        assert!(bs.test(0));
        assert_eq!(bs.count, 1);

        bs.reset(0);
        assert!(bs.none());
        assert!(!bs.test(0));
        assert_eq!(bs.count, 0);
    }
}


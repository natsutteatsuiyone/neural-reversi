//! Search stack for maintaining principal variation state at each ply.

use crate::constants::MAX_PLY;
use crate::square::Square;

/// A record stored for each ply in the search stack.
#[derive(Clone, Copy)]
pub struct StackRecord {
    /// Principal variation line from this ply to the end of search.
    pub pv: [Square; MAX_PLY],
}

/// Manages PV (Principal Variation) tracking across all plies.
pub struct SearchStack {
    stack: [StackRecord; MAX_PLY],
}

impl Default for SearchStack {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchStack {
    /// Creates a new search stack with all PV entries cleared.
    pub fn new() -> Self {
        SearchStack {
            stack: [StackRecord {
                pv: [Square::None; MAX_PLY],
            }; MAX_PLY],
        }
    }

    /// Updates the principal variation at the given ply.
    #[inline]
    pub fn update_pv(&mut self, sq: Square, ply: usize) {
        debug_assert!(ply + 1 < MAX_PLY, "ply out of range: {ply}");
        self.stack[ply].pv[0] = sq;
        if ply == 0 {
            return;
        }
        let mut idx = 0;
        while idx < self.stack[ply + 1].pv.len() && self.stack[ply + 1].pv[idx] != Square::None {
            self.stack[ply].pv[idx + 1] = self.stack[ply + 1].pv[idx];
            idx += 1;
        }
        self.stack[ply].pv[idx + 1] = Square::None;
    }

    /// Clears the principal variation at the given ply.
    ///
    /// Sets only the sentinel at position 0; all PV consumers stop at `Square::None`,
    /// so a full fill is unnecessary.
    #[inline]
    pub fn clear_pv(&mut self, ply: usize) {
        self.stack[ply].pv[0] = Square::None;
    }

    /// Returns the principal variation at the given ply.
    #[inline]
    pub fn get_pv(&self, ply: usize) -> &[Square; MAX_PLY] {
        &self.stack[ply].pv
    }

    /// Sets the principal variation at the given ply.
    #[inline]
    pub fn set_pv(&mut self, ply: usize, pv: &[Square; MAX_PLY]) {
        self.stack[ply].pv.copy_from_slice(pv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_initializes_all_pv_to_none() {
        let stack = SearchStack::new();
        for ply in 0..MAX_PLY {
            assert!(stack.get_pv(ply).iter().all(|&sq| sq == Square::None));
        }
    }

    #[test]
    fn update_pv_at_ply_zero_sets_first_entry_only() {
        let mut stack = SearchStack::new();
        stack.update_pv(Square::D3, 0);

        let pv = stack.get_pv(0);
        assert_eq!(pv[0], Square::D3);
    }

    #[test]
    fn update_pv_propagates_child_pv() {
        let mut stack = SearchStack::new();

        // Simulate a 3-ply PV built bottom-up: ply 3 → ply 2 → ply 1
        // Leaf: ply 3 has no children, set directly
        stack.update_pv(Square::C5, 0);

        // Ply 3: set move at leaf
        stack.stack[3].pv[0] = Square::F6;
        stack.stack[3].pv[1] = Square::None;

        // Ply 2: update_pv should copy child PV from ply 3
        stack.update_pv(Square::E3, 2);

        let pv2 = stack.get_pv(2);
        assert_eq!(pv2[0], Square::E3);
        assert_eq!(pv2[1], Square::F6); // copied from ply 3
        assert_eq!(pv2[2], Square::None); // sentinel

        // Ply 1: update_pv should copy child PV from ply 2
        stack.update_pv(Square::D3, 1);

        let pv1 = stack.get_pv(1);
        assert_eq!(pv1[0], Square::D3);
        assert_eq!(pv1[1], Square::E3); // copied from ply 2
        assert_eq!(pv1[2], Square::F6); // transitively from ply 3
        assert_eq!(pv1[3], Square::None); // sentinel
    }

    #[test]
    fn clear_pv_sets_sentinel() {
        let mut stack = SearchStack::new();
        stack.update_pv(Square::D3, 0);
        assert_eq!(stack.get_pv(0)[0], Square::D3);

        stack.clear_pv(0);
        assert_eq!(stack.get_pv(0)[0], Square::None);
    }

    #[test]
    fn set_pv_and_get_pv_roundtrip() {
        let mut stack = SearchStack::new();
        let mut pv = [Square::None; MAX_PLY];
        pv[0] = Square::C4;
        pv[1] = Square::D3;
        pv[2] = Square::None;

        stack.set_pv(5, &pv);

        let result = stack.get_pv(5);
        assert_eq!(result[0], Square::C4);
        assert_eq!(result[1], Square::D3);
        assert_eq!(result[2], Square::None);
    }

    #[test]
    fn update_pv_does_not_affect_other_plies() {
        let mut stack = SearchStack::new();
        stack.update_pv(Square::D3, 0);

        // Other plies should remain untouched
        for ply in 1..MAX_PLY {
            assert!(stack.get_pv(ply).iter().all(|&sq| sq == Square::None));
        }
    }
}

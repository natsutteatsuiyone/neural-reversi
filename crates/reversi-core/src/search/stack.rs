//! Search stack for maintaining principal variation state at each ply.

use crate::constants::MAX_PLY;
use crate::square::Square;

/// A record stored for each ply in the search stack.
#[derive(Clone, Copy)]
struct StackRecord {
    /// Principal variation from this ply, read up to the first [`None`] or the end.
    pv: [Option<Square>; MAX_PLY],
}

/// PV (Principal Variation) tracking across all plies.
pub struct SearchStack {
    stack: [StackRecord; MAX_PLY],
}

impl SearchStack {
    /// Creates a new search stack with all PV entries cleared.
    pub(crate) fn new() -> Self {
        SearchStack {
            stack: [StackRecord {
                pv: [None; MAX_PLY],
            }; MAX_PLY],
        }
    }

    /// Prepends a square to its child's principal variation, reserving the terminator.
    ///
    /// # Panics
    ///
    /// Panics if `ply` and `ply + 1` do not both identify stack rows.
    #[inline]
    pub fn update_pv(&mut self, sq: Square, ply: usize) {
        let [parent, child, ..] = &mut self.stack[ply..] else {
            panic!("ply out of range: {ply}");
        };
        parent.pv[0] = Some(sq);
        let mut idx = 0;
        while idx < MAX_PLY - 2 && child.pv[idx].is_some() {
            parent.pv[idx + 1] = child.pv[idx];
            idx += 1;
        }
        parent.pv[idx + 1] = None;
    }

    /// Clears the principal variation at the given ply.
    ///
    /// Sets only position 0; all PV consumers stop at the first [`None`],
    /// so a full fill is unnecessary.
    ///
    /// # Panics
    ///
    /// Panics if `ply` is at least [`MAX_PLY`].
    #[inline]
    pub fn clear_pv(&mut self, ply: usize) {
        self.stack[ply].pv[0] = None;
    }

    /// Clears the given ply's PV and the next ply's PV if that row exists.
    ///
    /// PV searches clear these rows up front so early returns cannot leak a stale
    /// continuation from a previous sibling or root move.
    ///
    /// # Panics
    ///
    /// Panics if `ply` is at least [`MAX_PLY`].
    #[inline]
    pub fn prepare_pv(&mut self, ply: usize) {
        self.clear_pv(ply);
        if ply + 1 < MAX_PLY {
            self.clear_pv(ply + 1);
        }
    }

    /// Returns the entire PV buffer at the given ply.
    ///
    /// # Panics
    ///
    /// Panics if `ply` is at least [`MAX_PLY`].
    #[inline]
    pub fn pv(&self, ply: usize) -> &[Option<Square>; MAX_PLY] {
        &self.stack[ply].pv
    }

    /// Copies the entire PV buffer into the given ply.
    ///
    /// # Panics
    ///
    /// Panics if `ply` is at least [`MAX_PLY`].
    #[inline]
    pub fn set_pv(&mut self, ply: usize, pv: &[Option<Square>; MAX_PLY]) {
        self.stack[ply].pv.copy_from_slice(pv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_pv_propagates_child_pv() {
        let mut stack = SearchStack::new();

        // Simulate a 3-ply PV built bottom-up: ply 3 → ply 2 → ply 1
        stack.stack[3].pv[0] = Some(Square::F6);

        // Ply 2: update_pv should copy child PV from ply 3
        stack.update_pv(Square::E3, 2);

        let pv2 = stack.pv(2);
        assert_eq!(pv2[0], Some(Square::E3));
        assert_eq!(pv2[1], Some(Square::F6)); // copied from ply 3

        // Ply 1: update_pv should copy child PV from ply 2
        stack.update_pv(Square::D3, 1);

        let pv1 = stack.pv(1);
        assert_eq!(pv1[0], Some(Square::D3));
        assert_eq!(pv1[1], Some(Square::E3)); // copied from ply 2
        assert_eq!(pv1[2], Some(Square::F6)); // transitively from ply 3
    }

    #[test]
    fn prepare_pv_clears_current_and_child_only() {
        let mut stack = SearchStack::new();
        stack.stack[0].pv[0] = Some(Square::C4);
        stack.stack[1].pv[0] = Some(Square::D3);
        stack.stack[2].pv[0] = Some(Square::E3);
        stack.stack[3].pv[0] = Some(Square::F6);

        stack.prepare_pv(1);

        assert_eq!(stack.pv(0)[0], Some(Square::C4));
        assert_eq!(stack.pv(1)[0], None);
        assert_eq!(stack.pv(2)[0], None);
        assert_eq!(stack.pv(3)[0], Some(Square::F6));
    }

    #[test]
    fn update_pv_does_not_affect_other_plies() {
        let mut stack = SearchStack::new();
        for (ply, record) in stack.stack.iter_mut().enumerate().skip(1) {
            record.pv[0] = Some(if ply % 2 == 0 { Square::E3 } else { Square::F6 });
            record.pv[1] = Some(Square::C4);
        }
        let before = stack.stack;
        stack.update_pv(Square::D3, 0);

        // Other plies should remain untouched
        for (ply, record) in before.iter().enumerate().skip(1) {
            assert_eq!(*stack.pv(ply), record.pv, "ply={ply}");
        }
    }

    #[test]
    fn update_pv_reserves_a_terminator_for_a_full_imported_child() {
        let mut stack = SearchStack::new();
        stack.set_pv(1, &[Some(Square::C3); MAX_PLY]);

        stack.update_pv(Square::D3, 0);

        let parent = stack.pv(0);
        assert_eq!(parent[0], Some(Square::D3));
        assert_eq!(parent[1..MAX_PLY - 1], [Some(Square::C3); MAX_PLY - 2]);
        assert_eq!(parent[MAX_PLY - 1], None);
        assert_eq!(*stack.pv(1), [Some(Square::C3); MAX_PLY]);
    }
}

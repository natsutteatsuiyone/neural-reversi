//! A transposition table shared by games on one line of play.

use std::sync::{Arc, Mutex, Weak};

use super::turn::{SearchTurn, TurnState};
use crate::board::Board;
use crate::transposition_table::TranspositionTable;

/// Transposition table that two games of one engine share while they follow
/// the same moves, such as the games of a synchro match.
///
/// The first game whose board leaves that line searches on a copy from then
/// on. Games join with [`Engine::new_game_sharing`].
///
/// [`Engine::new_game_sharing`]: super::Engine::new_game_sharing
#[derive(Clone, Default)]
pub(crate) struct SharedTable(Arc<Mutex<State>>);

#[derive(Default)]
struct State {
    /// Table of the first game. Weak so the table lives exactly as long as
    /// the games using it.
    tt: Weak<TranspositionTable>,
    /// Turn of the engine whose games share the table.
    owner: Option<Arc<TurnState>>,
    /// Number of games that joined, which numbers the next one.
    members: u32,
    /// Game and the board it left after its latest move.
    last_left: Option<(u32, Board)>,
    /// Whether a game left: with a copy, or by replacing its table.
    diverged: bool,
}

impl SharedTable {
    /// Admits a game of the engine that owns `turn`, unless two joined
    /// already. Returns its number and its table: the shared one while a game
    /// still holds it, else `fresh()`.
    pub(super) fn join(
        &self,
        turn: &Arc<TurnState>,
        fresh: impl FnOnce() -> TranspositionTable,
    ) -> Option<(u32, Arc<TranspositionTable>)> {
        let mut state = self.0.lock().unwrap();
        if state.members == 2 {
            return None;
        }
        let owner = state.owner.get_or_insert_with(|| Arc::clone(turn));
        assert!(
            Arc::ptr_eq(owner, turn),
            "a table is shared by the games of one engine"
        );
        let member = state.members;
        state.members += 1;
        let tt = state.tt.upgrade().unwrap_or_else(|| {
            let tt = Arc::new(fresh());
            if member == 0 {
                state.tt = Arc::downgrade(&tt);
            }
            tt
        });
        Some((member, tt))
    }

    /// Ends the sharing: each game keeps the table it has.
    pub(crate) fn dissolve(&self) {
        self.0.lock().unwrap().diverged = true;
    }

    /// Returns a copy of the table for game `member` once `board`, the
    /// position it moves from, is not the one another game left by its latest
    /// move.
    pub(crate) fn split_off(
        &self,
        turn: &SearchTurn,
        member: u32,
        board: &Board,
    ) -> Option<TranspositionTable> {
        let mut state = self.0.lock().unwrap();
        assert!(
            state
                .owner
                .as_ref()
                .is_some_and(|owner| turn.belongs_to(owner)),
            "the search turn belongs to another engine"
        );
        let diverged = !state.diverged
            && state
                .last_left
                .as_ref()
                .is_some_and(|(by, left)| *by != member && left != board);
        state.diverged |= diverged;
        diverged
            .then(|| state.tt.upgrade())
            .flatten()
            // SAFETY: only searches and resets of the owner's games write the
            // table. Each runs within the owner's turn and has published its
            // writes before giving it up, and the caller holds that turn until
            // after the copy.
            .map(|tt| unsafe { tt.duplicate() })
    }

    /// Records `left`, the position after the latest move of game `member`.
    pub(crate) fn record_move(&self, member: u32, left: Board) {
        let mut state = self.0.lock().unwrap();
        if !state.diverged {
            state.last_left = Some((member, left));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::square::Square;

    #[test]
    fn a_game_splits_off_only_when_its_board_leaves_the_line_of_the_other() {
        let owner = Arc::new(TurnState::default());
        let turn = &owner.wait();
        let shared = SharedTable::default();
        let (first, tt) = shared.join(&owner, || TranspositionTable::new(0)).unwrap();
        let (second, same) = shared.join(&owner, || unreachable!()).unwrap();
        assert!(shared.join(&owner, || unreachable!()).is_none());
        assert!(Arc::ptr_eq(&tt, &same));
        let start = Board::new();
        let [a, b] = [Square::D3, Square::C4].map(|sq| start.make_move(sq));

        assert!(shared.split_off(turn, first, &start).is_none());
        shared.record_move(first, a);
        // Same game again (the opponent passed there): no comparison yet.
        assert!(shared.split_off(turn, first, &b).is_none());
        // The opponent repeated our move in the other game.
        assert!(shared.split_off(turn, second, &a).is_none());
        shared.record_move(second, a.make_move(Square::C3));

        assert!(
            shared
                .split_off(turn, first, &a.make_move(Square::E3))
                .is_some()
        );
        assert!(shared.0.lock().unwrap().diverged);
        // Only the games keep the original table alive.
        drop((tt, same));
        assert!(shared.0.lock().unwrap().tt.upgrade().is_none());
        assert!(shared.split_off(turn, second, &b).is_none());
    }
}

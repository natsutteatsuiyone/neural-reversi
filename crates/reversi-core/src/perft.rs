//! Performance testing (perft) for move generation verification.

use crate::board::Board;
use crate::eval::pattern_feature::PatternFeatures;
use crate::move_list::MoveList;
use crate::search::side_to_move::SideToMove;

/// Counts the total nodes reachable from the standard initial position.
///
/// A depth of 0 counts the root node; depth 1 counts the immediate legal
/// moves. Larger values walk the tree recursively while exercising
/// [`PatternFeatures`] updates along each move.
///
/// [`PatternFeatures`]: crate::eval::pattern_feature::PatternFeatures
pub fn perft_root(depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let board = Board::new();
    let mut pattern_features = PatternFeatures::new(&board, 0);
    let side_to_move = SideToMove::Player;
    perft(&board, &mut pattern_features, 0, side_to_move, depth)
}

/// Recursively counts nodes in the game tree.
fn perft(
    board: &Board,
    pattern_feature: &mut PatternFeatures,
    ply: usize,
    side_to_move: SideToMove,
    depth: u32,
) -> u64 {
    debug_assert!(depth > 0);

    let mut nodes = 0;
    let move_list = MoveList::new(board);

    if move_list.count() > 0 {
        for m in move_list.iter() {
            let next = board.make_move_with_flipped(m.flipped, m.sq);
            pattern_feature.update(m.sq, m.flipped, ply, side_to_move);

            if depth <= 1 {
                nodes += 1;
            } else {
                nodes += perft(
                    &next,
                    pattern_feature,
                    ply + 1,
                    side_to_move.switch(),
                    depth - 1,
                );
            }
        }
    } else {
        let next = board.switch_players();
        if next.has_legal_moves() {
            nodes += perft(&next, pattern_feature, ply, side_to_move.switch(), depth);
        } else {
            nodes += 1;
        }
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_recurses_on_the_opponent_without_consuming_depth() {
        // Player (row 2) has no legal move while the opponent (row 1) does: a
        // forced pass. Both preconditions are asserted so a mis-built board fails
        // loudly rather than silently passing.
        let board = Board::from_bitboards(0x000000000000ff00, 0x00000000000000ff);
        assert!(!board.has_legal_moves(), "player must have no move");
        assert!(
            board.switch_players().has_legal_moves(),
            "opponent must have a move"
        );

        let depth = 3;
        let switched = board.switch_players();
        let mut pf_pass = PatternFeatures::new(&board, 0);
        let via_pass = perft(&board, &mut pf_pass, 0, SideToMove::Player, depth);
        let mut pf_direct = PatternFeatures::new(&board, 0);
        let direct = perft(&switched, &mut pf_direct, 0, SideToMove::Opponent, depth);

        // A pass consumes no ply or depth.
        assert_eq!(via_pass, direct);
        assert!(via_pass > 0);
    }

    #[test]
    fn terminal_position_counts_as_a_single_node() {
        // A full board has no moves for either side: a terminal (game-over) node.
        let board = Board::from_bitboards(u64::MAX, 0);
        assert!(!board.has_legal_moves());
        assert!(!board.switch_players().has_legal_moves());

        let mut pf = PatternFeatures::new(&board, 0);
        assert_eq!(perft(&board, &mut pf, 0, SideToMove::Player, 5), 1);
    }
}

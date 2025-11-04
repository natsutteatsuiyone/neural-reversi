use crate::board::Board;
use crate::eval::pattern_feature::PatternFeatures;
use crate::move_list::MoveList;
use crate::search::side_to_move::SideToMove;

/// Executes a perft run starting from the standard initial position.
///
/// # Arguments
///
/// * `depth` - Number of plies to expand from the initial position. A depth of
///   `1` counts the immediate legal moves; larger values walk the tree
///   recursively.
///
/// # Returns
///
/// The total node count the search visits from the initial position.
pub fn perft_root(depth: u32) -> u64 {
    let board = Board::new();
    let mut pattern_features = PatternFeatures::new(&board, 0);
    let side_to_move = SideToMove::Player;
    perft(&board, &mut pattern_features, 0, side_to_move, depth)
}

fn perft(
    board: &Board,
    pattern_feature: &mut PatternFeatures,
    ply: usize,
    side_to_move: SideToMove,
    depth: u32,
) -> u64 {
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

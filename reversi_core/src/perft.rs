use crate::board::Board;
use crate::move_list::MoveList;

/// Performs a perft (performance test) on the given board to a specified depth.
///
/// # Arguments
///
/// * `board` - A reference to the current game board.
/// * `depth` - The depth to which the perft function should search.
///
/// # Returns
///
/// The number of nodes (positions) reached at the given depth.
pub fn perft(board: &Board, depth: u32) -> u64 {
    let mut nodes = 0;
    let move_list = MoveList::new(board);

    if move_list.count > 0 {
        for m in move_list.iter() {
            let next = board.make_move_with_flipped(m.flipped, m.sq);
            if depth <= 1 {
                nodes += 1;
            } else {
                nodes += perft(&next, depth - 1);
            }
        }
    } else {
        let next = board.switch_players();
        if next.has_legal_moves() {
            nodes += perft(&next, depth);
        } else {
            nodes += 1;
        }
    }
    nodes
}

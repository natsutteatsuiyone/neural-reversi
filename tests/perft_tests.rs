use reversi::engine::board::Board;
use reversi::engine::perft::perft;

#[test]
fn test_perft() {
    let board = Board::new();
    let nodes = perft(&board, 9);
    assert_eq!(nodes, 3_005_320);
}

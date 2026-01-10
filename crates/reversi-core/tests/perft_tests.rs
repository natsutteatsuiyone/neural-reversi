use reversi_core::perft::perft_root;

#[test]
fn test_perft() {
    let nodes = perft_root(9);
    assert_eq!(nodes, 3_005_320);
}

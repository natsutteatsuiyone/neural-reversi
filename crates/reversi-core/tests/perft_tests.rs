use reversi_core::perft::perft_root;

#[test]
fn test_perft() {
    for (depth, leaves) in [
        (0, 1),
        (1, 4),
        (2, 12),
        (3, 56),
        (4, 244),
        (5, 1_396),
        (9, 3_005_320),
    ] {
        assert_eq!(perft_root(depth), leaves, "depth {depth}");
    }
}

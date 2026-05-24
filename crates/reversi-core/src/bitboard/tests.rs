use super::*;

#[test]
fn test_apply_move() {
    let player_board = Square::A1.bitboard();
    let flipped = Square::B1.bitboard() | Square::C1.bitboard();
    let result = player_board.apply_move(flipped, Square::D1);

    // Should have original disc at A1, flipped discs at B1 and C1, and new disc at D1
    assert!(result.contains(Square::A1));
    assert!(result.contains(Square::B1));
    assert!(result.contains(Square::C1));
    assert!(result.contains(Square::D1));
}

#[test]
fn test_apply_flip() {
    let opponent_board = Square::A1.bitboard() | Square::B1.bitboard() | Square::C1.bitboard();
    let flipped = Square::B1.bitboard() | Square::C1.bitboard();
    let result = opponent_board.apply_flip(flipped);

    // Should only have disc at A1 (B1 and C1 were flipped away)
    assert!(result.contains(Square::A1));
    assert!(!result.contains(Square::B1));
    assert!(!result.contains(Square::C1));
}

#[test]
fn test_set_remove_contains() {
    let mut board = Bitboard::new(0);

    board = board.set(Square::A1);
    assert!(board.contains(Square::A1));
    assert!(!board.contains(Square::A2));

    board = board.set(Square::H8);
    assert!(board.contains(Square::A1));
    assert!(board.contains(Square::H8));
    assert!(!board.contains(Square::D4));

    board = board.set(Square::A1);
    assert!(board.contains(Square::A1));

    board = board.remove(Square::A1);
    assert!(!board.contains(Square::A1));
    assert!(board.contains(Square::H8));
}

#[test]
fn test_corner_stability() {
    let board = Square::D4.bitboard().0 | Square::E5.bitboard().0;
    assert_eq!(Bitboard(board).corner_stability(), 0);

    let board = Square::A1.bitboard().0;
    assert_eq!(Bitboard(board).corner_stability(), 1);

    assert_eq!(Bitboard(CORNER_MASK).corner_stability(), 4);

    let board = Square::A1.bitboard().0
        | Square::A2.bitboard().0
        | Square::B1.bitboard().0
        | Square::B2.bitboard().0;
    assert_eq!(Bitboard(board).corner_stability(), 3);
}

#[test]
fn test_bitboard_iterator() {
    let bitboard = Square::A1.bitboard().0 | Square::B1.bitboard().0 | Square::H8.bitboard().0;
    let mut iterator = BitboardIterator::new(Bitboard(bitboard));

    assert_eq!(iterator.next(), Some(Square::A1));
    assert_eq!(iterator.next(), Some(Square::B1));
    assert_eq!(iterator.next(), Some(Square::H8));
    assert_eq!(iterator.next(), None);
}

#[test]
fn test_bitboard_iterator_empty() {
    let mut iterator = BitboardIterator::new(Bitboard(0));
    assert_eq!(iterator.next(), None);
}

#[test]
fn test_bitboard_iterator_full() {
    let mut iterator = BitboardIterator::new(Bitboard(u64::MAX));
    for i in 0..64 {
        assert_eq!(iterator.next(), Square::from_u32(i));
    }
    assert_eq!(iterator.next(), None);
}

#[test]
fn test_has_adjacent_bit() {
    let cases = [
        (Square::B2.bitboard(), Square::A1, true),
        (Square::B2.bitboard(), Square::A2, true),
        (Square::B2.bitboard(), Square::A3, true),
        (Square::B2.bitboard(), Square::A4, false),
        (Square::B2.bitboard(), Square::B1, true),
        (Square::B2.bitboard(), Square::B2, false),
        (Square::B2.bitboard(), Square::B3, true),
        (Square::B2.bitboard(), Square::B4, false),
        (Square::B2.bitboard(), Square::C1, true),
        (Square::B2.bitboard(), Square::C2, true),
        (Square::B2.bitboard(), Square::C3, true),
        (Square::B2.bitboard(), Square::C4, false),
        (Square::B2.bitboard(), Square::D1, false),
        (
            Square::B1.bitboard() | Square::A2.bitboard(),
            Square::A1,
            true,
        ),
        (
            Square::G8.bitboard() | Square::H7.bitboard(),
            Square::H8,
            true,
        ),
        (
            Square::C1.bitboard() | Square::D2.bitboard() | Square::E1.bitboard(),
            Square::D1,
            true,
        ),
    ];

    for (bitboard, square, expected) in cases {
        assert_eq!(
            bitboard.has_adjacent_bit(square),
            expected,
            "adjacency mismatch for {square:?} on {bitboard:?}"
        );
    }
}

#[test]
fn test_bitboard_struct_clear_lsb() {
    let bb = Bitboard::new(0b1010);
    let bb = bb.clear_lsb();
    assert_eq!(bb.0, 0b1000);
    let bb = bb.clear_lsb();
    assert_eq!(bb.0, 0);
}

#[test]
fn test_bitboard_struct_lsb_square() {
    assert_eq!(Bitboard::new(0).lsb_square(), None);
    assert_eq!(Bitboard::new(1).lsb_square(), Some(Square::A1));
    assert_eq!(Bitboard::new(0b1000).lsb_square(), Some(Square::D1));
    assert_eq!(
        Bitboard::new(0x8000000000000000).lsb_square(),
        Some(Square::H8)
    );
}

#[test]
fn test_bitboard_struct_operators() {
    let a = Bitboard::new(0b1100);
    let b = Bitboard::new(0b1010);

    // BitAnd
    assert_eq!((a & b).0, 0b1000);

    // BitOr
    assert_eq!((a | b).0, 0b1110);

    // BitXor
    assert_eq!((a ^ b).0, 0b0110);

    // Not
    assert_eq!((!Bitboard::new(0)).0, u64::MAX);

    // Shl
    assert_eq!((Bitboard::new(1) << 3).0, 0b1000);

    // Shr
    assert_eq!((Bitboard::new(0b1000) >> 3).0, 1);
}

#[test]
fn test_bitboard_struct_assign_operators() {
    let mut bb = Bitboard::new(0b1100);

    // BitAndAssign
    bb &= Bitboard::new(0b1010);
    assert_eq!(bb.0, 0b1000);

    // BitOrAssign
    bb |= Bitboard::new(0b0001);
    assert_eq!(bb.0, 0b1001);

    // BitXorAssign
    bb ^= Bitboard::new(0b1111);
    assert_eq!(bb.0, 0b0110);

    // ShlAssign
    bb <<= 2;
    assert_eq!(bb.0, 0b11000);

    // ShrAssign
    bb >>= 1;
    assert_eq!(bb.0, 0b1100);
}

#[test]
fn test_bitboard_struct_conversions() {
    // From<u64>
    let bb: Bitboard = 0x1234u64.into();
    assert_eq!(bb.0, 0x1234);

    // From<Bitboard> for u64
    let val: u64 = bb.into();
    assert_eq!(val, 0x1234);

    // From<Square>
    let bb: Bitboard = Square::E4.into();
    assert_eq!(bb, Square::E4.bitboard());
}

#[test]
fn test_bitboard_struct_into_iter() {
    let bb = Square::A1.bitboard() | Square::C3.bitboard() | Square::H8.bitboard();

    let squares: Vec<Square> = bb.into_iter().collect();
    assert_eq!(squares.len(), 3);
    assert_eq!(squares[0], Square::A1);
    assert_eq!(squares[1], Square::C3);
    assert_eq!(squares[2], Square::H8);
}

#[test]
fn test_bitboard_struct_display() {
    let bb = Bitboard::new(CORNER_MASK);
    let display = format!("{}", bb);
    // H8 and A8 should be on first line, A1 and H1 on last line
    let lines: Vec<&str> = display.lines().collect();
    assert_eq!(lines.len(), 8);
    assert!(lines[0].starts_with("1")); // A8
    assert!(lines[0].ends_with("1")); // H8
    assert!(lines[7].starts_with("1")); // A1
    assert!(lines[7].ends_with("1")); // H1
}

#[test]
fn test_bitboard_struct_corner_weighted_count() {
    assert_eq!(Bitboard::new(CORNER_MASK).corner_weighted_count(), 8); // 4 corners * 2
    assert_eq!(Bitboard::new(0).corner_weighted_count(), 0);

    // Mixed: 2 corners + 2 non-corners = 2*2 + 2 = 6
    let bb = Square::A1.bitboard()
        | Square::H8.bitboard()
        | Square::D4.bitboard()
        | Square::E5.bitboard();
    assert_eq!(bb.corner_weighted_count(), 6);
}

#[test]
fn test_pop_lsb() {
    // Single bit
    let bb = Square::E4.bitboard();
    let (sq, rest) = bb.pop_lsb();
    assert_eq!(sq, Square::E4);
    assert!(rest.is_empty());

    // Multiple bits - should pop in LSB order
    let bb = Square::A1.bitboard() | Square::C3.bitboard() | Square::H8.bitboard();
    let (sq1, rest1) = bb.pop_lsb();
    assert_eq!(sq1, Square::A1);
    assert!(!rest1.is_empty());

    let (sq2, rest2) = rest1.pop_lsb();
    assert_eq!(sq2, Square::C3);
    assert!(!rest2.is_empty());

    let (sq3, rest3) = rest2.pop_lsb();
    assert_eq!(sq3, Square::H8);
    assert!(rest3.is_empty());

    // All corners
    let mut bb = Bitboard::new(CORNER_MASK);
    let mut popped = Vec::new();
    while !bb.is_empty() {
        let (sq, rest) = bb.pop_lsb();
        popped.push(sq);
        bb = rest;
    }
    assert_eq!(popped.len(), 4);
    assert_eq!(popped[0], Square::A1);
    assert_eq!(popped[1], Square::H1);
    assert_eq!(popped[2], Square::A8);
    assert_eq!(popped[3], Square::H8);
}

#[test]
fn test_from_square() {
    // Test all corners
    assert_eq!(Bitboard::from_square(Square::A1).0, 1);
    assert_eq!(Bitboard::from_square(Square::H1).0, 0x80);
    assert_eq!(Bitboard::from_square(Square::A8).0, 0x0100000000000000);
    assert_eq!(Bitboard::from_square(Square::H8).0, 0x8000000000000000);

    // Test center squares
    assert_eq!(Bitboard::from_square(Square::D4).0, 1 << 27);
    assert_eq!(Bitboard::from_square(Square::E5).0, 1 << 36);

    // Verify equivalence with Square::bitboard()
    for i in 0..64 {
        let sq = Square::from_u32(i).unwrap();
        assert_eq!(Bitboard::from_square(sq), sq.bitboard());
    }
}

#[test]
fn test_is_empty() {
    assert!(Bitboard::new(0).is_empty());
    assert!(!Bitboard::new(1).is_empty());
    assert!(!Bitboard::new(u64::MAX).is_empty());
    assert!(!Square::A1.bitboard().is_empty());

    // After clearing all bits
    let bb = Square::A1.bitboard();
    let bb = bb.remove(Square::A1);
    assert!(bb.is_empty());
}

#[test]
fn test_count() {
    assert_eq!(Bitboard::new(0).count(), 0);
    assert_eq!(Bitboard::new(1).count(), 1);
    assert_eq!(Bitboard::new(u64::MAX).count(), 64);
    assert_eq!(Bitboard::new(CORNER_MASK).count(), 4);

    // Sparse pattern
    let bb = Square::A1.bitboard() | Square::D4.bitboard() | Square::H8.bitboard();
    assert_eq!(bb.count(), 3);

    // Full rank
    assert_eq!(Bitboard::new(0xFF).count(), 8);

    // Checkerboard pattern
    assert_eq!(Bitboard::new(0x5555555555555555).count(), 32);
    assert_eq!(Bitboard::new(0xAAAAAAAAAAAAAAAA).count(), 32);
}

#[test]
fn test_has_single_bit_nonzero() {
    assert!(Bitboard::new(1).has_single_bit_nonzero());
    assert!(Square::A1.bitboard().has_single_bit_nonzero());
    assert!(Square::H8.bitboard().has_single_bit_nonzero());
    assert!(!Bitboard::new(3).has_single_bit_nonzero());
    assert!(!Bitboard::new(u64::MAX).has_single_bit_nonzero());
}

#[test]
fn test_corners() {
    // All corners from full board
    assert_eq!(Bitboard::new(u64::MAX).corners().0, CORNER_MASK);

    // No corners from center squares
    let center = Square::D4.bitboard()
        | Square::D5.bitboard()
        | Square::E4.bitboard()
        | Square::E5.bitboard();
    assert_eq!(center.corners().0, 0);

    // Partial corners
    let bb = Square::A1.bitboard() | Square::H8.bitboard() | Square::D4.bitboard();
    let corners = bb.corners();
    assert!(corners.contains(Square::A1));
    assert!(corners.contains(Square::H8));
    assert!(!corners.contains(Square::D4));
    assert!(!corners.contains(Square::H1));
    assert!(!corners.contains(Square::A8));
    assert_eq!(corners.count(), 2);

    // Empty board
    assert_eq!(Bitboard::new(0).corners().0, 0);
}

#[test]
fn test_non_corners() {
    // Full board minus corners
    let full = Bitboard::new(u64::MAX);
    let non_corners = full.non_corners();
    assert_eq!(non_corners.count(), 60);
    assert!(!non_corners.contains(Square::A1));
    assert!(!non_corners.contains(Square::H1));
    assert!(!non_corners.contains(Square::A8));
    assert!(!non_corners.contains(Square::H8));
    assert!(non_corners.contains(Square::D4));

    // Only corners gives empty
    assert_eq!(Bitboard::new(CORNER_MASK).non_corners().0, 0);

    // Mixed board
    let bb = Square::A1.bitboard() | Square::D4.bitboard() | Square::E5.bitboard();
    let non_corners = bb.non_corners();
    assert!(!non_corners.contains(Square::A1));
    assert!(non_corners.contains(Square::D4));
    assert!(non_corners.contains(Square::E5));
    assert_eq!(non_corners.count(), 2);

    // Empty board
    assert_eq!(Bitboard::new(0).non_corners().0, 0);
}

#[test]
fn test_flip_vertical() {
    // Simple pattern
    let board = Bitboard::new(0x0102030405060708);
    let flipped = board.flip_vertical();
    assert_eq!(flipped.0, 0x0807060504030201);

    // Symmetric pattern should be unchanged
    let symmetric = Bitboard::new(0x1818181818181818);
    assert_eq!(symmetric.flip_vertical().flip_vertical(), symmetric);

    // Empty
    assert_eq!(Bitboard::new(0).flip_vertical().0, 0);

    // Full
    assert_eq!(
        Bitboard::new(0xFFFFFFFFFFFFFFFF).flip_vertical().0,
        0xFFFFFFFFFFFFFFFF
    );
}

#[test]
fn test_flip_horizontal() {
    // Edge columns
    assert_eq!(
        Bitboard::new(0x0101010101010101).flip_horizontal().0,
        0x8080808080808080
    );
    assert_eq!(
        Bitboard::new(0x8080808080808080).flip_horizontal().0,
        0x0101010101010101
    );

    // Nibble pattern
    assert_eq!(
        Bitboard::new(0x0F0F0F0F0F0F0F0F).flip_horizontal().0,
        0xF0F0F0F0F0F0F0F0
    );

    // Double flip identity
    let original = Bitboard::new(0x123456789ABCDEF0);
    assert_eq!(original.flip_horizontal().flip_horizontal(), original);

    // Single rank
    assert_eq!(Bitboard::new(0xFF).flip_horizontal().0, 0xFF);
}

#[test]
fn test_rotate_90_clockwise() {
    // Corners
    assert_eq!(
        Bitboard::new(0x0000000000000001).rotate_90_clockwise().0,
        0x0000000000000080
    );
    assert_eq!(
        Bitboard::new(0x0000000000000080).rotate_90_clockwise().0,
        0x8000000000000000
    );
    assert_eq!(
        Bitboard::new(0x8000000000000000).rotate_90_clockwise().0,
        0x0100000000000000
    );
    assert_eq!(
        Bitboard::new(0x0100000000000000).rotate_90_clockwise().0,
        0x0000000000000001
    );

    // 4x rotation identity
    let original = Bitboard::new(0x123456789ABCDEF0);
    let rotated = original
        .rotate_90_clockwise()
        .rotate_90_clockwise()
        .rotate_90_clockwise()
        .rotate_90_clockwise();
    assert_eq!(rotated, original);
}

#[test]
fn test_rotate_180_clockwise() {
    // Test corners
    assert_eq!(
        Bitboard::new(0x0000000000000001).rotate_180_clockwise().0,
        0x8000000000000000
    );
    assert_eq!(
        Bitboard::new(0x8000000000000000).rotate_180_clockwise().0,
        0x0000000000000001
    );
    assert_eq!(
        Bitboard::new(0x0000000000000080).rotate_180_clockwise().0,
        0x0100000000000000
    );
    assert_eq!(
        Bitboard::new(0x0100000000000000).rotate_180_clockwise().0,
        0x0000000000000080
    );

    // Test a full row
    assert_eq!(
        Bitboard::new(0x00000000000000FF).rotate_180_clockwise().0,
        0xFF00000000000000
    );
    assert_eq!(
        Bitboard::new(0xFF00000000000000).rotate_180_clockwise().0,
        0x00000000000000FF
    );

    // Test a pattern
    let original = Bitboard::new(0x0F0F0F0F00000000);
    let rotated = Bitboard::new(0x00000000F0F0F0F0);
    assert_eq!(original.rotate_180_clockwise(), rotated);

    // Double rotation identity
    let test_board = Bitboard::new(0x123456789ABCDEF0);
    assert_eq!(
        test_board.rotate_180_clockwise().rotate_180_clockwise(),
        test_board
    );

    // Empty and full boards
    assert_eq!(Bitboard::new(0).rotate_180_clockwise().0, 0);
    assert_eq!(Bitboard::new(u64::MAX).rotate_180_clockwise().0, u64::MAX);
}

#[test]
fn test_rotate_270_clockwise() {
    // Test corners
    assert_eq!(
        Bitboard::new(0x0000000000000001).rotate_270_clockwise().0,
        0x0100000000000000
    );
    assert_eq!(
        Bitboard::new(0x0100000000000000).rotate_270_clockwise().0,
        0x8000000000000000
    );
    assert_eq!(
        Bitboard::new(0x8000000000000000).rotate_270_clockwise().0,
        0x0000000000000080
    );
    assert_eq!(
        Bitboard::new(0x0000000000000080).rotate_270_clockwise().0,
        0x0000000000000001
    );

    // 4x rotation identity
    let original = Bitboard::new(0x123456789ABCDEF0);
    let rotated = original
        .rotate_270_clockwise()
        .rotate_270_clockwise()
        .rotate_270_clockwise()
        .rotate_270_clockwise();
    assert_eq!(rotated, original);

    // Equivalence to 3x 90-degree rotation
    let rotated_90_3x = original
        .rotate_90_clockwise()
        .rotate_90_clockwise()
        .rotate_90_clockwise();
    assert_eq!(original.rotate_270_clockwise(), rotated_90_3x);
}

#[test]
fn test_flip_diag_a1h8() {
    // Diagonal invariant
    assert_eq!(
        Bitboard::new(0x8040201008040201).flip_diag_a1h8().0,
        0x8040201008040201
    );

    // Corners
    assert_eq!(
        Bitboard::new(0x0000000000000001).flip_diag_a1h8().0,
        0x0000000000000001
    );
    assert_eq!(
        Bitboard::new(0x8000000000000000).flip_diag_a1h8().0,
        0x8000000000000000
    );
    assert_eq!(
        Bitboard::new(0x0000000000000080).flip_diag_a1h8().0,
        0x0100000000000000
    );
    assert_eq!(
        Bitboard::new(0x0100000000000000).flip_diag_a1h8().0,
        0x0000000000000080
    );

    // Double flip identity
    let original = Bitboard::new(0x123456789ABCDEF0);
    assert_eq!(original.flip_diag_a1h8().flip_diag_a1h8(), original);
}

#[test]
fn test_flip_diag_a8h1() {
    // Anti-diagonal invariant
    assert_eq!(
        Bitboard::new(0x0102040810204080).flip_diag_a8h1().0,
        0x0102040810204080
    );

    // Corners
    assert_eq!(
        Bitboard::new(0x0100000000000000).flip_diag_a8h1().0,
        0x0100000000000000
    );
    assert_eq!(
        Bitboard::new(0x0000000000000080).flip_diag_a8h1().0,
        0x0000000000000080
    );
    assert_eq!(
        Bitboard::new(0x0000000000000001).flip_diag_a8h1().0,
        0x8000000000000000
    );
    assert_eq!(
        Bitboard::new(0x8000000000000000).flip_diag_a8h1().0,
        0x0000000000000001
    );

    // Double flip identity
    let original = Bitboard::new(0x123456789ABCDEF0);
    assert_eq!(original.flip_diag_a8h1().flip_diag_a8h1(), original);
}

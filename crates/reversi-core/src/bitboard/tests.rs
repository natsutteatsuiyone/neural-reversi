use super::*;

const SAMPLE_BOARDS: [Bitboard; 9] = [
    Bitboard::new(0),
    Bitboard::new(u64::MAX),
    Bitboard::new(CORNER_MASK),
    Bitboard::new(0x8040_2010_0804_0201),
    Bitboard::new(0x0102_0408_1020_4080),
    Bitboard::new(0x00FF_0000_0000_FF00),
    Bitboard::new(0xAA55_AA55_55AA_55AA),
    Bitboard::new(0x1234_5678_9ABC_DEF0),
    Bitboard::new(0x0000_0018_2442_8100),
];

fn bitboard_from_squares(squares: &[Square]) -> Bitboard {
    squares.iter().fold(Bitboard::new(0), |bitboard, &square| {
        bitboard | square.bitboard()
    })
}

fn expected_squares(bits: u64) -> Vec<Square> {
    (0..64)
        .filter(|&index| bits & (1u64 << index) != 0)
        .map(|index| Square::from_u32(index).unwrap())
        .collect()
}

fn bit_at(file: u8, rank: u8) -> u64 {
    1u64 << (u32::from(rank) * 8 + u32::from(file))
}

fn transform_bits<F>(bits: u64, transform: F) -> u64
where
    F: Fn(u8, u8) -> (u8, u8),
{
    let mut transformed = 0;

    for rank in 0u8..8 {
        for file in 0u8..8 {
            let bit = bit_at(file, rank);
            if bits & bit == 0 {
                continue;
            }

            let (transformed_file, transformed_rank) = transform(file, rank);
            transformed |= bit_at(transformed_file, transformed_rank);
        }
    }

    transformed
}

fn reference_bracketable_adjacent_mask(square: Square) -> u64 {
    let file = square.file() as i8;
    let rank = square.rank() as i8;
    let mut mask = 0;

    for rank_delta in -1i8..=1 {
        for file_delta in -1i8..=1 {
            if file_delta == 0 && rank_delta == 0 {
                continue;
            }

            let adjacent_file = file + file_delta;
            let adjacent_rank = rank + rank_delta;
            let beyond_file = adjacent_file + file_delta;
            let beyond_rank = adjacent_rank + rank_delta;
            if (0..8).contains(&adjacent_file)
                && (0..8).contains(&adjacent_rank)
                && (0..8).contains(&beyond_file)
                && (0..8).contains(&beyond_rank)
            {
                mask |= bit_at(adjacent_file as u8, adjacent_rank as u8);
            }
        }
    }

    mask
}

fn reference_corner_stability(board: Bitboard) -> u32 {
    const CORNER_NEIGHBORS: [(Square, [Square; 2]); 4] = [
        (Square::A1, [Square::B1, Square::A2]),
        (Square::H1, [Square::G1, Square::H2]),
        (Square::A8, [Square::B8, Square::A7]),
        (Square::H8, [Square::G8, Square::H7]),
    ];

    let mut stable = Bitboard::new(0);
    for (corner, neighbors) in CORNER_NEIGHBORS {
        if !board.contains(corner) {
            continue;
        }

        stable = stable.set(corner);
        for neighbor in neighbors {
            if board.contains(neighbor) {
                stable = stable.set(neighbor);
            }
        }
    }

    stable.count()
}

fn assert_transform_matches_reference<F>(name: &str, actual: fn(Bitboard) -> Bitboard, reference: F)
where
    F: Fn(u8, u8) -> (u8, u8),
{
    for board in SAMPLE_BOARDS {
        let expected = transform_bits(board.bits(), &reference);
        assert_eq!(
            actual(board).bits(),
            expected,
            "{name} mismatch for board {:016x}",
            board.bits()
        );
    }
}

#[test]
fn constructors_and_conversions_preserve_raw_bits() {
    for bits in [0, 1, 0x1234_5678_9ABC_DEF0, u64::MAX] {
        let board = Bitboard::new(bits);
        assert_eq!(board.bits(), bits);
        assert_eq!(Bitboard::from(bits), board);
        assert_eq!(u64::from(board), bits);
    }

    for square in Square::iter() {
        let expected = 1u64 << (square.index() as u32);
        let from_square = Bitboard::from_square(square);
        let from_trait: Bitboard = square.into();

        assert_eq!(from_square.bits(), expected, "from_square({square:?})");
        assert_eq!(
            square.bitboard(),
            from_square,
            "Square::bitboard({square:?})"
        );
        assert_eq!(from_trait, from_square, "From<Square> for {square:?}");
    }
}

#[test]
fn set_remove_contains_are_pure_and_idempotent() {
    let original = bitboard_from_squares(&[Square::B2, Square::H8]);
    let with_a1 = original.set(Square::A1);

    assert_eq!(original, bitboard_from_squares(&[Square::B2, Square::H8]));
    assert_eq!(with_a1.set(Square::A1), with_a1);
    assert_eq!(with_a1.remove(Square::C3), with_a1);

    let removed = with_a1.remove(Square::B2);
    assert_eq!(removed, bitboard_from_squares(&[Square::A1, Square::H8]));

    for square in Square::iter() {
        let expected = [Square::A1, Square::B2, Square::H8].contains(&square);
        assert_eq!(
            with_a1.contains(square),
            expected,
            "contains({square:?}) did not match the constructed board"
        );
    }
}

#[test]
fn bit_counting_and_lsb_operations_match_u64_semantics() {
    for board in SAMPLE_BOARDS {
        let bits = board.bits();

        assert_eq!(board.count(), bits.count_ones(), "count for {bits:016x}");
        assert_eq!(board.is_empty(), bits == 0, "is_empty for {bits:016x}");
        assert_eq!(
            board.clear_lsb().bits(),
            bits & bits.wrapping_sub(1),
            "clear_lsb for {bits:016x}"
        );

        if bits == 0 {
            assert_eq!(board.lsb_square(), None);
            continue;
        }

        let expected_square = Square::from_u32(bits.trailing_zeros()).unwrap();
        let expected_rest = bits & bits.wrapping_sub(1);
        let (popped_square, rest) = board.pop_lsb();

        assert_eq!(board.lsb_square(), Some(expected_square));
        assert_eq!(board.lsb_square_unchecked(), expected_square);
        assert_eq!(popped_square, expected_square);
        assert_eq!(rest.bits(), expected_rest);
        assert_eq!(board.has_single_bit_nonzero(), bits.is_power_of_two());
    }
}

#[test]
fn iterators_yield_all_squares_in_lsb_order() {
    for board in SAMPLE_BOARDS {
        let expected = expected_squares(board.bits());
        let from_iter_method: Vec<Square> = board.iter().collect();
        let from_into_iter: Vec<Square> = board.into_iter().collect();
        let reconstructed = from_iter_method
            .iter()
            .fold(Bitboard::new(0), |bitboard, &square| bitboard.set(square));

        assert_eq!(
            from_iter_method,
            expected,
            "iter() for {:016x}",
            board.bits()
        );
        assert_eq!(
            from_into_iter,
            expected,
            "IntoIterator for {:016x}",
            board.bits()
        );
        assert_eq!(
            reconstructed,
            board,
            "reconstructed board for {:016x}",
            board.bits()
        );
    }
}

#[test]
fn bitwise_and_shift_operators_match_raw_u64_operations() {
    let pairs = [
        (0xF0F0_F0F0_0000_0000, 0x0F0F_0000_F0F0_0000),
        (0x1234_5678_9ABC_DEF0, 0xFFFF_0000_FFFF_0000),
        (0, u64::MAX),
    ];

    for (lhs, rhs) in pairs {
        let lhs_board = Bitboard::new(lhs);
        let rhs_board = Bitboard::new(rhs);

        assert_eq!((lhs_board & rhs_board).bits(), lhs & rhs);
        assert_eq!((lhs_board | rhs_board).bits(), lhs | rhs);
        assert_eq!((lhs_board ^ rhs_board).bits(), lhs ^ rhs);
        assert_eq!((!lhs_board).bits(), !lhs);
    }

    let shifted = Bitboard::new(0x0000_0001_0000_0081);
    for shift in [0u32, 1, 7, 8, 31, 63] {
        assert_eq!((shifted << shift).bits(), shifted.bits() << shift);
        assert_eq!((shifted >> shift).bits(), shifted.bits() >> shift);
    }

    let mut assigned = Bitboard::new(0b1100);
    assigned &= Bitboard::new(0b1010);
    assert_eq!(assigned.bits(), 0b1000);
    assigned |= Bitboard::new(0b0001);
    assert_eq!(assigned.bits(), 0b1001);
    assigned ^= Bitboard::new(0b1111);
    assert_eq!(assigned.bits(), 0b0110);
    assigned <<= 2;
    assert_eq!(assigned.bits(), 0b11000);
    assigned >>= 1;
    assert_eq!(assigned.bits(), 0b1100);
}

#[test]
fn apply_move_and_apply_flip_are_xor_updates() {
    let player = bitboard_from_squares(&[Square::A1, Square::D4]);
    let flipped = bitboard_from_squares(&[Square::B1, Square::C1, Square::E4]);

    assert_eq!(
        player.apply_move(flipped, Square::D1),
        bitboard_from_squares(&[
            Square::A1,
            Square::B1,
            Square::C1,
            Square::D1,
            Square::D4,
            Square::E4,
        ])
    );

    let opponent =
        bitboard_from_squares(&[Square::A1, Square::B1, Square::C1, Square::D4, Square::E4]);
    assert_eq!(
        opponent.apply_flip(flipped),
        bitboard_from_squares(&[Square::A1, Square::D4])
    );
    assert_eq!(opponent.apply_flip(flipped).apply_flip(flipped), opponent);
}

#[test]
fn corner_masks_weights_and_partitioning_match_their_definitions() {
    for board in SAMPLE_BOARDS {
        let corners = board.corners();
        let non_corners = board.non_corners();

        assert_eq!(corners.bits(), board.bits() & CORNER_MASK);
        assert_eq!(non_corners.bits(), board.bits() & !CORNER_MASK);
        assert_eq!(corners | non_corners, board);
        assert!((corners & non_corners).is_empty());
        assert_eq!(
            board.corner_weighted_count(),
            board.count() + corners.count()
        );
    }
}

#[test]
fn corner_stability_counts_occupied_corners_and_immediate_edge_neighbors() {
    let explicit_cases = [
        Bitboard::new(0),
        Square::D4.bitboard(),
        bitboard_from_squares(&[Square::A1]),
        bitboard_from_squares(&[Square::A1, Square::B1, Square::A2, Square::B2]),
        bitboard_from_squares(&[Square::A1, Square::B1, Square::C1, Square::A2, Square::A3]),
        Bitboard::new(u64::MAX),
    ];

    for board in SAMPLE_BOARDS.into_iter().chain(explicit_cases) {
        assert_eq!(
            board.corner_stability(),
            reference_corner_stability(board),
            "corner_stability for {:016x}",
            board.bits()
        );
    }

    assert_eq!(Bitboard::new(u64::MAX).corner_stability(), 12);
}

#[test]
fn display_prints_rank_eight_first_with_exact_board_shape() {
    let board = bitboard_from_squares(&[
        Square::A1,
        Square::H1,
        Square::D4,
        Square::E5,
        Square::A8,
        Square::H8,
    ]);

    assert_eq!(
        format!("{board}"),
        "1......1\n........\n........\n....1...\n...1....\n........\n........\n1......1\n"
    );
}

#[test]
fn bracketable_adjacency_matches_coordinate_reference_for_every_square_pair() {
    for target in Square::iter() {
        let adjacent_mask = reference_bracketable_adjacent_mask(target);
        assert!(!target.bitboard().has_adjacent_bit(target));
        assert_eq!(
            Bitboard::new(adjacent_mask).has_adjacent_bit(target),
            adjacent_mask != 0
        );

        for source in Square::iter() {
            let expected = adjacent_mask & source.bitboard().bits() != 0;
            assert_eq!(
                source.bitboard().has_adjacent_bit(target),
                expected,
                "source={source:?}, target={target:?}"
            );
        }
    }

    let d4_non_neighbors =
        !(reference_bracketable_adjacent_mask(Square::D4) | Square::D4.bitboard().bits());
    assert!(!Bitboard::new(d4_non_neighbors).has_adjacent_bit(Square::D4));
}

#[test]
fn geometric_transforms_match_coordinate_reference() {
    assert_transform_matches_reference("flip_vertical", Bitboard::flip_vertical, |file, rank| {
        (file, 7 - rank)
    });
    assert_transform_matches_reference(
        "flip_horizontal",
        Bitboard::flip_horizontal,
        |file, rank| (7 - file, rank),
    );
    assert_transform_matches_reference("flip_diag_a1h8", Bitboard::flip_diag_a1h8, |file, rank| {
        (rank, file)
    });
    assert_transform_matches_reference("flip_diag_a8h1", Bitboard::flip_diag_a8h1, |file, rank| {
        (7 - rank, 7 - file)
    });
    assert_transform_matches_reference(
        "rotate_90_clockwise",
        Bitboard::rotate_90_clockwise,
        |file, rank| (7 - rank, file),
    );
    assert_transform_matches_reference(
        "rotate_180_clockwise",
        Bitboard::rotate_180_clockwise,
        |file, rank| (7 - file, 7 - rank),
    );
    assert_transform_matches_reference(
        "rotate_270_clockwise",
        Bitboard::rotate_270_clockwise,
        |file, rank| (rank, 7 - file),
    );
}

#[test]
fn geometric_transform_identities_hold_for_representative_boards() {
    for board in SAMPLE_BOARDS {
        assert_eq!(board.flip_vertical().flip_vertical(), board);
        assert_eq!(board.flip_horizontal().flip_horizontal(), board);
        assert_eq!(board.flip_diag_a1h8().flip_diag_a1h8(), board);
        assert_eq!(board.flip_diag_a8h1().flip_diag_a8h1(), board);

        let rotate_90_twice = board.rotate_90_clockwise().rotate_90_clockwise();
        let rotate_90_four_times = rotate_90_twice.rotate_90_clockwise().rotate_90_clockwise();
        assert_eq!(rotate_90_twice, board.rotate_180_clockwise());
        assert_eq!(rotate_90_four_times, board);
        assert_eq!(
            board.rotate_270_clockwise(),
            board
                .rotate_90_clockwise()
                .rotate_90_clockwise()
                .rotate_90_clockwise()
        );
    }
}

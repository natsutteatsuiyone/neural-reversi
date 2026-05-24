use super::movegen::*;
use super::*;

type Position = (u64, u64);

const DIRECTIONS: [(i8, i8); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

fn initial_position() -> Position {
    (
        Square::D5.bitboard().bits() | Square::E4.bitboard().bits(),
        Square::D4.bitboard().bits() | Square::E5.bitboard().bits(),
    )
}

fn core_positions() -> [Position; 4] {
    [
        initial_position(),
        (0x00003C3C3C000000, 0x0000C3C3C3000000),
        (0xFF00000000000000, 0x00FF000000000000),
        (0x0000001824428100, 0x0000002442810000),
    ]
}

fn extended_positions() -> [Position; 10] {
    [
        initial_position(),
        (0xFF000000000000FF, 0x00FFFFFFFFFFFF00),
        (0x8181818181818181, 0x7E7E7E7E7E7E7E7E),
        (0x8040201008040201, 0x0102040810204080),
        (CORNER_MASK, 0x4281000000008142),
        (0x0000001000000000, 0x0000002800000000),
        (0x00003C3C00000000, 0x0000C3C300000000),
        (0xFFFFFFFF00000000, 0x00000000FFFFFFFF),
        (0xAAAAAAAAAAAAAAAA, 0x5555555555555554),
        (0x0F0F0F0F00000000, 0x00000000F0F0F0F0),
    ]
}

fn bitboard_from_squares(squares: &[Square]) -> Bitboard {
    squares.iter().fold(Bitboard::new(0), |bitboard, &square| {
        bitboard | square.bitboard()
    })
}

fn position_bitboards(position: Position) -> (Bitboard, Bitboard) {
    (Bitboard::new(position.0), Bitboard::new(position.1))
}

fn reference_get_moves(player: u64, opponent: u64) -> u64 {
    let occupied = player | opponent;
    let mut moves = 0;

    for square in 0..64 {
        let bit = 1u64 << square;
        if occupied & bit != 0 {
            continue;
        }

        let file = (square % 8) as i8;
        let rank = (square / 8) as i8;
        if DIRECTIONS
            .iter()
            .any(|&(df, dr)| reference_captures(player, opponent, file, rank, df, dr))
        {
            moves |= bit;
        }
    }

    moves
}

fn reference_captures(player: u64, opponent: u64, file: i8, rank: i8, df: i8, dr: i8) -> bool {
    let mut file = file + df;
    let mut rank = rank + dr;
    let mut saw_opponent = false;

    while on_board(file) && on_board(rank) {
        let bit = 1u64 << ((rank as u32 * 8) + file as u32);
        if opponent & bit != 0 {
            saw_opponent = true;
            file += df;
            rank += dr;
            continue;
        }

        return saw_opponent && (player & bit != 0);
    }

    false
}

fn on_board(coord: i8) -> bool {
    (0..8).contains(&coord)
}

fn next_random(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *seed
}

fn assert_moves_match_reference(player: u64, opponent: u64) {
    assert_eq!(
        player & opponent,
        0,
        "test position overlaps: player={player:016x}, opponent={opponent:016x}"
    );
    let expected = reference_get_moves(player, opponent);

    assert_eq!(
        get_moves_portable(player, opponent),
        expected,
        "portable moves differ from reference for player={player:016x}, opponent={opponent:016x}"
    );
    assert_eq!(
        Bitboard::new(player)
            .get_moves(Bitboard::new(opponent))
            .bits(),
        expected,
        "dispatched moves differ from reference for player={player:016x}, opponent={opponent:016x}"
    );
}

#[cfg(target_arch = "x86_64")]
fn detected_x86_backends() -> Option<(bool, bool)> {
    let has_avx2 = is_x86_feature_detected!("avx2");
    let has_avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl");
    (has_avx2 || has_avx512).then_some((has_avx2, has_avx512))
}

#[cfg(target_arch = "x86_64")]
fn assert_x86_moves_match_portable(player: u64, opponent: u64, has_avx2: bool, has_avx512: bool) {
    let moves_portable = get_moves_portable(player, opponent);

    if has_avx2 {
        let moves_avx2 = unsafe { get_moves_avx2(player, opponent) };
        assert_eq!(
            moves_portable, moves_avx2,
            "Extended: Portable vs AVX2 differ for player={player:016x}, opponent={opponent:016x}"
        );
    }

    if has_avx512 {
        let moves_avx512 = unsafe { get_moves_avx512(player, opponent) };
        assert_eq!(
            moves_portable, moves_avx512,
            "Extended: Portable vs AVX-512 differ for player={player:016x}, opponent={opponent:016x}"
        );
    }
}

#[cfg(target_arch = "x86_64")]
fn assert_x86_combined_matches_scalar(
    player: u64,
    opponent: u64,
    has_avx2: bool,
    has_avx512: bool,
) {
    let moves_scalar = get_moves_portable(player, opponent);
    let potential_scalar = get_potential_moves(player, opponent);

    if has_avx2 {
        let (moves_avx2, potential_avx2) =
            unsafe { get_moves_and_potential_avx2(player, opponent) };
        assert_eq!(
            moves_scalar, moves_avx2,
            "Extended: Moves mismatch AVX2 for player={player:016x}, opponent={opponent:016x}"
        );
        assert_eq!(
            potential_scalar, potential_avx2,
            "Extended: Potential mismatch AVX2 for player={player:016x}, opponent={opponent:016x}"
        );
    }

    if has_avx512 {
        let (moves_avx512, potential_avx512) =
            unsafe { get_moves_and_potential_avx512(player, opponent) };
        assert_eq!(
            moves_scalar, moves_avx512,
            "Extended: Moves mismatch AVX512 for player={player:016x}, opponent={opponent:016x}"
        );
        assert_eq!(
            potential_scalar, potential_avx512,
            "Extended: Potential mismatch AVX512 for player={player:016x}, opponent={opponent:016x}"
        );
    }
}

#[test]
fn test_get_moves_matches_reference_scanner() {
    let fixed_positions = [(0, 0), (0, u64::MAX), (u64::MAX, 0)];

    for (player, opponent) in fixed_positions.into_iter().chain(extended_positions()) {
        assert_moves_match_reference(player, opponent);
    }

    let mut seed = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..128 {
        let player = next_random(&mut seed);
        let opponent = next_random(&mut seed) & !player;
        assert_moves_match_reference(player, opponent);
    }
}

#[test]
fn test_get_moves_initial_position() {
    let (player, opponent) = position_bitboards(initial_position());
    let moves = player.get_moves(opponent);
    let expected = bitboard_from_squares(&[Square::C4, Square::F5, Square::D3, Square::E6]);

    assert_eq!(moves, expected);
}

#[test]
fn test_get_moves_no_moves() {
    let player = Bitboard::new(0);
    let opponent = Bitboard::new(u64::MAX);
    let moves = player.get_moves(opponent);

    assert_eq!(moves, Bitboard::new(0));
}

#[test]
fn test_get_moves_capture_all_directions() {
    let player = bitboard_from_squares(&[
        Square::A1,
        Square::H1,
        Square::A8,
        Square::H8,
        Square::A4,
        Square::H4,
        Square::D1,
        Square::D8,
    ]);

    let opponent = bitboard_from_squares(&[
        Square::B2,
        Square::C3,
        Square::E5,
        Square::F6,
        Square::G7,
        Square::D2,
        Square::D3,
        Square::D5,
        Square::D6,
        Square::D7,
        Square::B4,
        Square::C4,
        Square::E4,
        Square::F4,
        Square::G4,
        Square::C2,
        Square::E2,
        Square::F3,
        Square::C5,
        Square::B5,
        Square::B3,
        Square::F5,
        Square::E6,
        Square::C6,
        Square::B6,
    ]);

    let moves = player.get_moves(opponent);

    assert!(moves.contains(Square::D4));
}

#[test]
fn test_get_potential_moves_initial_position() {
    let (player, opponent) = position_bitboards(initial_position());
    let potential = player.get_potential_moves(opponent);
    let expected = bitboard_from_squares(&[
        Square::C3,
        Square::C4,
        Square::C5,
        Square::D3,
        Square::E3,
        Square::D6,
        Square::E6,
        Square::F4,
        Square::F5,
        Square::F6,
    ]);

    assert_eq!(potential, expected);
}

#[test]
fn test_get_moves_and_potential_initial_position() {
    let (player, opponent) = position_bitboards(initial_position());
    let (moves, potential) = player.get_moves_and_potential(opponent);
    let expected_moves = player.get_moves(opponent);
    let expected_potential = player.get_potential_moves(opponent);

    assert_eq!(moves, expected_moves);
    assert_eq!(potential, expected_potential);
}

#[test]
fn test_get_moves_and_potential_portable_consistency() {
    for (player, opponent) in core_positions() {
        let (moves_portable, potential_portable) =
            get_moves_and_potential_portable(player, opponent);

        assert_eq!(
            get_moves_portable(player, opponent),
            moves_portable,
            "Moves mismatch portable combined for player={player:016x}, opponent={opponent:016x}"
        );
        assert_eq!(
            get_potential_moves(player, opponent),
            potential_portable,
            "Potential mismatch portable combined for player={player:016x}, opponent={opponent:016x}"
        );
    }
}

#[test]
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn test_get_moves_and_potential_consistency_neon() {
    for (player, opponent) in core_positions() {
        let moves_scalar = get_moves_portable(player, opponent);
        let potential_scalar = get_potential_moves(player, opponent);
        let (moves_neon, pot_neon) = unsafe { get_moves_and_potential_neon(player, opponent) };

        assert_eq!(
            moves_scalar, moves_neon,
            "Moves mismatch NEON for player={player:016x}, opponent={opponent:016x}"
        );
        assert_eq!(
            potential_scalar, pot_neon,
            "Potential mismatch NEON for player={player:016x}, opponent={opponent:016x}"
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_x86_movegen_consistency_extended() {
    let Some((has_avx2, has_avx512)) = detected_x86_backends() else {
        return;
    };

    for (player, opponent) in extended_positions() {
        assert_x86_moves_match_portable(player, opponent, has_avx2, has_avx512);
        assert_x86_combined_matches_scalar(player, opponent, has_avx2, has_avx512);
    }
}

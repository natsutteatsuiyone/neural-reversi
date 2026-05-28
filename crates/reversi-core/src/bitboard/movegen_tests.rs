use super::movegen::*;
use super::*;

#[derive(Clone, Copy)]
struct Direction {
    name: &'static str,
    file_delta: i8,
    rank_delta: i8,
}

#[derive(Clone, Copy)]
struct Position {
    name: &'static str,
    player: u64,
    opponent: u64,
}

impl Position {
    fn new(name: &'static str, player: u64, opponent: u64) -> Self {
        Self {
            name,
            player,
            opponent,
        }
    }

    fn bitboards(self) -> (Bitboard, Bitboard) {
        (Bitboard::new(self.player), Bitboard::new(self.opponent))
    }
}

const DIRECTIONS: [Direction; 8] = [
    Direction {
        name: "southwest",
        file_delta: -1,
        rank_delta: -1,
    },
    Direction {
        name: "south",
        file_delta: 0,
        rank_delta: -1,
    },
    Direction {
        name: "southeast",
        file_delta: 1,
        rank_delta: -1,
    },
    Direction {
        name: "west",
        file_delta: -1,
        rank_delta: 0,
    },
    Direction {
        name: "east",
        file_delta: 1,
        rank_delta: 0,
    },
    Direction {
        name: "northwest",
        file_delta: -1,
        rank_delta: 1,
    },
    Direction {
        name: "north",
        file_delta: 0,
        rank_delta: 1,
    },
    Direction {
        name: "northeast",
        file_delta: 1,
        rank_delta: 1,
    },
];

fn bitboard_from_squares(squares: &[Square]) -> Bitboard {
    squares.iter().fold(Bitboard::new(0), |bitboard, &square| {
        bitboard | square.bitboard()
    })
}

fn position_from_squares(
    name: &'static str,
    player_squares: &[Square],
    opponent_squares: &[Square],
) -> Position {
    Position::new(
        name,
        bitboard_from_squares(player_squares).bits(),
        bitboard_from_squares(opponent_squares).bits(),
    )
}

fn initial_position() -> Position {
    position_from_squares(
        "initial position",
        &[Square::D5, Square::E4],
        &[Square::D4, Square::E5],
    )
}

fn fixed_positions() -> [Position; 14] {
    [
        initial_position(),
        Position::new("empty board", 0, 0),
        Position::new("no player discs", 0, u64::MAX),
        Position::new("no opponent discs", u64::MAX, 0),
        Position::new(
            "edge players against filled center",
            0xFF00_0000_0000_00FF,
            0x00FF_FFFF_FFFF_FF00,
        ),
        Position::new(
            "file edge players",
            0x8181_8181_8181_8181,
            0x7E7E_7E7E_7E7E_7E7E,
        ),
        Position::new(
            "crossed diagonals",
            0x8040_2010_0804_0201,
            0x0102_0408_1020_4080,
        ),
        Position::new(
            "corners against x-squares",
            CORNER_MASK,
            0x4281_0000_0000_8142,
        ),
        Position::new(
            "center pressure",
            0x0000_0010_0000_0000,
            0x0000_0028_0000_0000,
        ),
        Position::new(
            "upper and lower halves",
            0xFFFF_FFFF_0000_0000,
            0x0000_0000_FFFF_FFFF,
        ),
        Position::new(
            "checkerboard with one empty",
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5554,
        ),
        Position::new("nibble split", 0x0F0F_0F0F_0000_0000, 0x0000_0000_F0F0_F0F0),
        position_from_squares(
            "h-file and a-file are not adjacent",
            &[Square::H1],
            &[Square::A1],
        ),
        position_from_squares(
            "a-file and h-file are not adjacent",
            &[Square::A1],
            &[Square::H1],
        ),
    ]
}

fn next_random(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *seed
}

fn random_position(seed: &mut u64) -> Position {
    let player = next_random(seed);
    let opponent = next_random(seed) & !player;
    Position::new("deterministic random position", player, opponent)
}

fn for_each_reference_position(mut assert_position: impl FnMut(Position)) {
    for position in fixed_positions() {
        assert_position(position);
    }

    let mut seed = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..512 {
        assert_position(random_position(&mut seed));
    }
}

fn on_board(coord: i8) -> bool {
    (0..8).contains(&coord)
}

fn bit_at(file: i8, rank: i8) -> u64 {
    debug_assert!(on_board(file));
    debug_assert!(on_board(rank));
    1u64 << (u32::from(rank as u8) * 8 + u32::from(file as u8))
}

fn offset_square(square: Square, file_delta: i8, rank_delta: i8) -> Square {
    let file = square.file() as i8 + file_delta;
    let rank = square.rank() as i8 + rank_delta;
    assert!(on_board(file), "file offset moved {square:?} off board");
    assert!(on_board(rank), "rank offset moved {square:?} off board");
    Square::from_file_rank(file as u8, rank as u8)
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
        if DIRECTIONS.iter().any(|direction| {
            reference_captures(
                player,
                opponent,
                file,
                rank,
                direction.file_delta,
                direction.rank_delta,
            )
        }) {
            moves |= bit;
        }
    }

    moves
}

fn reference_captures(
    player: u64,
    opponent: u64,
    file: i8,
    rank: i8,
    file_delta: i8,
    rank_delta: i8,
) -> bool {
    let mut file = file + file_delta;
    let mut rank = rank + rank_delta;
    let mut saw_opponent = false;

    while on_board(file) && on_board(rank) {
        let bit = bit_at(file, rank);
        if opponent & bit != 0 {
            saw_opponent = true;
            file += file_delta;
            rank += rank_delta;
            continue;
        }

        return saw_opponent && (player & bit != 0);
    }

    false
}

fn reference_get_potential_moves(player: u64, opponent: u64) -> u64 {
    let occupied = player | opponent;
    let mut potential = 0;

    for square in 0..64 {
        let bit = 1u64 << square;
        if occupied & bit != 0 {
            continue;
        }

        let file = (square % 8) as i8;
        let rank = (square / 8) as i8;
        if DIRECTIONS.iter().any(|direction| {
            let adjacent_file = file + direction.file_delta;
            let adjacent_rank = rank + direction.rank_delta;
            let beyond_file = adjacent_file + direction.file_delta;
            let beyond_rank = adjacent_rank + direction.rank_delta;

            on_board(adjacent_file)
                && on_board(adjacent_rank)
                && on_board(beyond_file)
                && on_board(beyond_rank)
                && (opponent & bit_at(adjacent_file, adjacent_rank)) != 0
        }) {
            potential |= bit;
        }
    }

    potential
}

fn assert_disjoint(position: Position) {
    assert_eq!(
        position.player & position.opponent,
        0,
        "{} overlaps: player={:016x}, opponent={:016x}",
        position.name,
        position.player,
        position.opponent
    );
}

fn assert_moves_match_reference(position: Position) {
    assert_disjoint(position);
    let expected = reference_get_moves(position.player, position.opponent);
    let (player, opponent) = position.bitboards();

    assert_eq!(
        get_moves_portable(position.player, position.opponent),
        expected,
        "{}: portable moves differ from reference for player={:016x}, opponent={:016x}",
        position.name,
        position.player,
        position.opponent
    );
    assert_eq!(
        player.get_moves(opponent).bits(),
        expected,
        "{}: dispatched moves differ from reference for player={:016x}, opponent={:016x}",
        position.name,
        position.player,
        position.opponent
    );
}

fn assert_potential_matches_reference(position: Position) {
    assert_disjoint(position);
    let expected = reference_get_potential_moves(position.player, position.opponent);
    let (player, opponent) = position.bitboards();

    assert_eq!(
        get_potential_moves(position.player, position.opponent),
        expected,
        "{}: scalar potential differs from reference for player={:016x}, opponent={:016x}",
        position.name,
        position.player,
        position.opponent
    );
    assert_eq!(
        player.get_potential_moves(opponent).bits(),
        expected,
        "{}: public potential differs from reference for player={:016x}, opponent={:016x}",
        position.name,
        position.player,
        position.opponent
    );
}

fn assert_combined_matches_separate_paths(position: Position) {
    assert_disjoint(position);
    let expected_moves = get_moves_portable(position.player, position.opponent);
    let expected_potential = get_potential_moves(position.player, position.opponent);
    let (portable_moves, portable_potential) =
        get_moves_and_potential_portable(position.player, position.opponent);
    let (player, opponent) = position.bitboards();
    let (public_moves, public_potential) = player.get_moves_and_potential(opponent);

    assert_eq!(
        portable_moves, expected_moves,
        "{}: portable combined moves differ from scalar moves for player={:016x}, opponent={:016x}",
        position.name, position.player, position.opponent
    );
    assert_eq!(
        portable_potential, expected_potential,
        "{}: portable combined potential differs from scalar potential for player={:016x}, opponent={:016x}",
        position.name, position.player, position.opponent
    );
    assert_eq!(
        public_moves,
        player.get_moves(opponent),
        "{}: public combined moves differ from public separate moves for player={:016x}, opponent={:016x}",
        position.name,
        position.player,
        position.opponent
    );
    assert_eq!(
        public_potential,
        player.get_potential_moves(opponent),
        "{}: public combined potential differs from public separate potential for player={:016x}, opponent={:016x}",
        position.name,
        position.player,
        position.opponent
    );
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn assert_neon_combined_matches_scalar(position: Position) {
    let expected_moves = get_moves_portable(position.player, position.opponent);
    let expected_potential = get_potential_moves(position.player, position.opponent);
    let (moves_neon, potential_neon) =
        unsafe { get_moves_and_potential_neon(position.player, position.opponent) };

    assert_eq!(
        moves_neon, expected_moves,
        "{}: NEON moves differ from scalar for player={:016x}, opponent={:016x}",
        position.name, position.player, position.opponent
    );
    assert_eq!(
        potential_neon, expected_potential,
        "{}: NEON potential differs from scalar for player={:016x}, opponent={:016x}",
        position.name, position.player, position.opponent
    );
}

#[cfg(target_arch = "x86_64")]
fn detected_x86_backends() -> Option<(bool, bool)> {
    let has_avx2 = is_x86_feature_detected!("avx2");
    let has_avx512 = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl");
    (has_avx2 || has_avx512).then_some((has_avx2, has_avx512))
}

#[cfg(target_arch = "x86_64")]
fn assert_x86_backends_match_scalar(position: Position, has_avx2: bool, has_avx512: bool) {
    let expected_moves = get_moves_portable(position.player, position.opponent);
    let expected_potential = get_potential_moves(position.player, position.opponent);

    if has_avx2 {
        let moves_avx2 = unsafe { get_moves_avx2(position.player, position.opponent) };
        let (combined_moves_avx2, combined_potential_avx2) =
            unsafe { get_moves_and_potential_avx2(position.player, position.opponent) };

        assert_eq!(
            moves_avx2, expected_moves,
            "{}: AVX2 moves differ from scalar for player={:016x}, opponent={:016x}",
            position.name, position.player, position.opponent
        );
        assert_eq!(
            combined_moves_avx2, expected_moves,
            "{}: AVX2 combined moves differ from scalar for player={:016x}, opponent={:016x}",
            position.name, position.player, position.opponent
        );
        assert_eq!(
            combined_potential_avx2, expected_potential,
            "{}: AVX2 potential differs from scalar for player={:016x}, opponent={:016x}",
            position.name, position.player, position.opponent
        );
    }

    if has_avx512 {
        let moves_avx512 = unsafe { get_moves_avx512(position.player, position.opponent) };
        let (combined_moves_avx512, combined_potential_avx512) =
            unsafe { get_moves_and_potential_avx512(position.player, position.opponent) };

        assert_eq!(
            moves_avx512, expected_moves,
            "{}: AVX512 moves differ from scalar for player={:016x}, opponent={:016x}",
            position.name, position.player, position.opponent
        );
        assert_eq!(
            combined_moves_avx512, expected_moves,
            "{}: AVX512 combined moves differ from scalar for player={:016x}, opponent={:016x}",
            position.name, position.player, position.opponent
        );
        assert_eq!(
            combined_potential_avx512, expected_potential,
            "{}: AVX512 potential differs from scalar for player={:016x}, opponent={:016x}",
            position.name, position.player, position.opponent
        );
    }
}

#[test]
fn legal_moves_match_independent_reference_for_representative_positions() {
    for_each_reference_position(assert_moves_match_reference);
}

#[test]
fn legal_moves_are_detected_in_each_direction() {
    let move_square = Square::D4;
    let expected = move_square.bitboard();

    for direction in DIRECTIONS {
        let adjacent_opponent =
            offset_square(move_square, direction.file_delta, direction.rank_delta);
        let closing_player = offset_square(
            move_square,
            direction.file_delta * 2,
            direction.rank_delta * 2,
        );
        let position =
            position_from_squares(direction.name, &[closing_player], &[adjacent_opponent]);

        assert_eq!(
            Bitboard::new(reference_get_moves(position.player, position.opponent)),
            expected,
            "directional fixture should have exactly one move for {}",
            direction.name
        );
        assert_moves_match_reference(position);
    }
}

#[test]
fn legal_moves_do_not_wrap_edges_or_jump_over_gaps() {
    let cases = [
        (
            position_from_squares(
                "h1/a1 are not horizontal neighbors",
                &[Square::H1],
                &[Square::A1],
            ),
            Bitboard::new(0),
        ),
        (
            position_from_squares(
                "a1/h1 are not horizontal neighbors",
                &[Square::A1],
                &[Square::H1],
            ),
            Bitboard::new(0),
        ),
        (
            position_from_squares("gap breaks a horizontal run", &[Square::A1], &[Square::C1]),
            Bitboard::new(0),
        ),
        (
            position_from_squares("opponent run without closing player", &[], &[Square::B1]),
            Bitboard::new(0),
        ),
        (
            position_from_squares("single bounded opponent", &[Square::A1], &[Square::B1]),
            Square::C1.bitboard(),
        ),
        (
            position_from_squares(
                "long bounded opponent run",
                &[Square::A1],
                &[
                    Square::B1,
                    Square::C1,
                    Square::D1,
                    Square::E1,
                    Square::F1,
                    Square::G1,
                ],
            ),
            Square::H1.bitboard(),
        ),
    ];

    for (position, expected) in cases {
        assert_eq!(
            Bitboard::new(reference_get_moves(position.player, position.opponent)),
            expected,
            "{} reference fixture mismatch",
            position.name
        );
        assert_moves_match_reference(position);
        assert_eq!(
            Bitboard::new(get_moves_portable(position.player, position.opponent)),
            expected,
            "{} portable exact moves mismatch",
            position.name
        );
    }
}

#[test]
fn potential_moves_match_independent_reference_for_representative_positions() {
    for_each_reference_position(assert_potential_matches_reference);
}

#[test]
fn initial_position_has_exact_legal_and_potential_moves() {
    let position = initial_position();
    let (player, opponent) = position.bitboards();

    assert_eq!(
        player.get_moves(opponent),
        bitboard_from_squares(&[Square::C4, Square::D3, Square::E6, Square::F5])
    );
    assert_eq!(
        player.get_potential_moves(opponent),
        bitboard_from_squares(&[
            Square::C3,
            Square::D3,
            Square::E3,
            Square::C4,
            Square::C5,
            Square::F4,
            Square::F5,
            Square::D6,
            Square::E6,
            Square::F6,
        ])
    );
}

#[test]
fn combined_move_and_potential_matches_separate_paths() {
    for_each_reference_position(assert_combined_matches_separate_paths);
}

#[test]
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn neon_combined_move_and_potential_matches_scalar_paths() {
    for_each_reference_position(assert_neon_combined_matches_scalar);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn x86_movegen_backends_match_scalar_paths() {
    let Some((has_avx2, has_avx512)) = detected_x86_backends() else {
        return;
    };

    for_each_reference_position(|position| {
        assert_x86_backends_match_scalar(position, has_avx2, has_avx512);
    });
}

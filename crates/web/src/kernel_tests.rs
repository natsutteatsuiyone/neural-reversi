//! Checks of the kernels the browser build runs against scalar references.
//!
//! On `wasm32` these run the simd128 code, including the `reversi-core` flip
//! backend, which no native test reaches. Run them with `bun run test:wasm`.

use rand::{RngExt, SeedableRng, rngs::StdRng};
use reversi_core::{bitboard::Bitboard, board::Board, flip, square::Square};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

use crate::move_list::legal_moves_pair;

fn reference_flip(sq: Square, player: u64, opponent: u64) -> u64 {
    const DIRECTIONS: [(i32, i32); 8] = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];

    let mut flipped = 0;
    for (dx, dy) in DIRECTIONS {
        let (mut x, mut y) = (sq.file() as i32 + dx, sq.rank() as i32 + dy);
        let mut line = 0;
        while (0..8).contains(&x) && (0..8).contains(&y) {
            let bit = 1u64 << (y * 8 + x);
            if opponent & bit == 0 {
                if player & bit != 0 {
                    flipped |= line;
                }
                break;
            }
            line |= bit;
            x += dx;
            y += dy;
        }
    }
    flipped
}

fn reference_legal_moves(board: &Board) -> u64 {
    let (player, opponent) = (board.player().bits(), board.opponent().bits());
    Square::iter()
        .filter(|&sq| (player | opponent) & sq.bitboard().bits() == 0)
        .filter(|&sq| reference_flip(sq, player, opponent) != 0)
        .fold(0, |moves, sq| moves | sq.bitboard().bits())
}

/// Returns boards from sparse to full, with at least four empty squares.
fn random_boards(seed: u64, count: usize) -> impl Iterator<Item = Board> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(move |case| {
        let mut occupied = rng.random::<u64>();
        for _ in 0..case % 5 {
            occupied |= rng.random::<u64>();
        }
        while occupied.count_ones() > 60 {
            occupied &= !(1 << rng.random_range(0..64));
        }
        let player = rng.random::<u64>() & occupied;
        Board::from_bitboards(player, occupied & !player)
    })
}

#[test]
#[cfg(target_arch = "wasm32")]
fn wasm_build_uses_simd128() {
    assert!(cfg!(target_feature = "simd128"));
}

#[test]
fn flip_matches_reference() {
    for board in random_boards(0xF11B, 2000) {
        let (player, opponent) = (board.player(), board.opponent());
        let expected = |sq| Bitboard::new(reference_flip(sq, player.bits(), opponent.bits()));
        let empties: Vec<Square> = board.empty_squares().iter().collect();

        for &sq in &empties {
            assert_eq!(
                flip::flip(sq, player, opponent),
                expected(sq),
                "{sq:?} on {board:?}"
            );
        }
        for sqs in empties.windows(4) {
            let [a, b, c, d] = [sqs[0], sqs[1], sqs[2], sqs[3]];
            assert_eq!(
                flip::flip2(a, b, player, opponent),
                (expected(a), expected(b)),
                "{a:?} {b:?} on {board:?}"
            );
            assert_eq!(
                flip::flip3(a, b, c, player, opponent),
                (expected(a), expected(b), expected(c)),
                "{a:?} {b:?} {c:?} on {board:?}"
            );
            assert_eq!(
                flip::flip4(a, b, c, d, player, opponent),
                (expected(a), expected(b), expected(c), expected(d)),
                "{a:?} {b:?} {c:?} {d:?} on {board:?}"
            );
        }
    }
}

#[test]
fn legal_moves_pair_matches_reference() {
    let boards: Vec<Board> = random_boards(0x1E6A1, 4000).collect();

    for [a, b] in boards.as_chunks().0 {
        let (moves_a, moves_b) = legal_moves_pair(a, b);

        assert_eq!(
            moves_a.bits(),
            reference_legal_moves(a),
            "first of {a:?} {b:?}"
        );
        assert_eq!(
            moves_b.bits(),
            reference_legal_moves(b),
            "second of {a:?} {b:?}"
        );
    }
}

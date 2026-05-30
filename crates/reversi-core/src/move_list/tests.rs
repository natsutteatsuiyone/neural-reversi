use std::collections::HashSet;

use crate::board::Board;
use crate::disc::Disc;
use crate::square::Square;

use super::{ConcurrentMoveIterator, MoveList};

/// Tests move generation for the starting position.
#[test]
fn test_move_list_new() {
    let board = Board::new();
    let move_list = MoveList::new(&board);
    assert_eq!(move_list.count(), 4);

    let moves: Vec<Square> = move_list.iter().map(|m| m.sq).collect();
    assert!(moves.contains(&Square::D3));
    assert!(moves.contains(&Square::C4));
    assert!(moves.contains(&Square::F5));
    assert!(moves.contains(&Square::E6));
}

/// Tests move generation with more complex position.
#[test]
fn test_move_list_generation_complex() {
    let board = Board::from_string(
        "--------\
             --------\
             ---OX---\
             --OXX---\
             --XXX---\
             --------\
             --------\
             --------",
        Disc::Black,
    )
    .unwrap();

    let move_list = MoveList::new(&board);
    for mv in move_list.iter() {
        assert!(!mv.flipped.is_empty());
    }
}

/// Tests move generation when no moves are available.
#[test]
fn test_move_list_no_moves() {
    let board = Board::from_bitboards(u64::MAX, 0);
    let move_list = MoveList::new(&board);
    assert_eq!(move_list.count(), 0);
    assert!(move_list.first().is_none());
}

/// Tests first() method.
#[test]
fn test_first() {
    let board = Board::new();
    let move_list = MoveList::new(&board);

    let first = move_list.first().unwrap();
    let first_iter = move_list.iter().next().unwrap();
    assert_eq!(first.sq, first_iter.sq);
}

/// Tests iterator methods.
#[test]
fn test_iterators() {
    let board = Board::new();
    let move_list = MoveList::new(&board);

    let count_iter = move_list.iter().count();
    assert_eq!(count_iter, move_list.count());

    let squares: Vec<Square> = move_list.iter().map(|m| m.sq).collect();
    assert_eq!(squares.len(), 4);
}

/// Tests the sort method.
#[test]
fn test_sort() {
    let board = Board::new();
    let mut move_list = MoveList::new(&board);

    move_list.moves[0].value = 10;
    move_list.moves[1].value = 30;
    move_list.moves[2].value = 20;
    move_list.moves[3].value = 40;

    move_list.sort();

    let values: Vec<i32> = move_list.iter().map(|m| m.value).collect();
    assert_eq!(values, vec![40, 30, 20, 10]);
}

/// Tests the best-first iterator with manually set move values.
#[test]
fn test_best_first_iter() {
    let board = Board::new();
    let mut move_list = MoveList::new(&board);
    move_list.moves[0].value = 10;
    move_list.moves[1].value = 5;
    move_list.moves[2].value = 15;
    move_list.moves[3].value = 2;

    let mut iter = move_list.best_first_iter();
    assert_eq!(iter.next().unwrap().value, 15);
    assert_eq!(iter.next().unwrap().value, 10);
    assert_eq!(iter.next().unwrap().value, 5);
    assert_eq!(iter.next().unwrap().value, 2);
    assert!(iter.next().is_none());
}

/// Tests best-first iterator with equal values.
#[test]
fn test_best_first_iter_equal_values() {
    let board = Board::new();
    let mut move_list = MoveList::new(&board);

    for i in 0..move_list.count() {
        move_list.moves[i].value = 100;
    }

    let total = move_list.count();
    let iter = move_list.best_first_iter();
    let mut count = 0;
    for mv in iter {
        assert_eq!(mv.value, 100);
        count += 1;
    }
    assert_eq!(count, total);
}

/// Tests best-first iterator behavior with no legal moves.
#[test]
fn test_best_first_iter_empty_list() {
    let board = Board::from_bitboards(u64::MAX, 0);
    let mut move_list = MoveList::new(&board);
    assert_eq!(move_list.count(), 0);

    let mut iter = move_list.best_first_iter();
    assert!(iter.next().is_none());
}

/// Tests best-first iterator with single move.
#[test]
fn test_best_first_iter_single_move() {
    let board = Board::from_string(
        "XXXXXXXX\
             XXXXXXXX\
             XXXXXXXX\
             XXXXXXXX\
             XXXXXXXX\
             XXXXXXXX\
             XXXXXXXO\
             XXXXXXO-",
        Disc::Black,
    )
    .unwrap();

    let mut move_list = MoveList::new(&board);
    assert_eq!(move_list.count(), 1);

    let mut iter = move_list.best_first_iter();
    assert!(iter.next().is_some());
    assert!(iter.next().is_none());
}

/// Tests best-first iterator preserves all moves.
#[test]
fn test_best_first_iter_completeness() {
    let board = Board::new();
    let mut move_list = MoveList::new(&board);

    for i in 0..move_list.count() {
        move_list.moves[i].value = (i * 10) as i32;
    }

    let count = move_list.count();
    let iter = move_list.best_first_iter();
    let mut seen_values = HashSet::new();

    for mv in iter {
        assert!(seen_values.insert(mv.value));
    }

    assert_eq!(seen_values.len(), count);
}

/// Tests concurrent move iterator.
#[test]
fn test_concurrent_move_iterator() {
    let board = Board::new();
    let move_list = MoveList::new(&board);
    let concurrent_iter = ConcurrentMoveIterator::new(move_list);

    assert_eq!(concurrent_iter.count(), 4);
    assert_eq!(concurrent_iter.remaining(), 4);

    let mut moves = Vec::new();
    while let Some((mv, idx)) = concurrent_iter.next() {
        moves.push((mv.sq, idx));
    }

    assert_eq!(moves.len(), 4);
    assert_eq!(concurrent_iter.remaining(), 0);
    for (i, (_, idx)) in moves.iter().enumerate() {
        assert_eq!(*idx, i + 1);
    }

    assert!(concurrent_iter.next().is_none());
}

#[test]
fn test_get_move() {
    let board = Board::new();
    let move_list = MoveList::new(&board);

    let moves_from_iter: Vec<Square> = move_list.iter().map(|m| m.sq).collect();
    for (i, expected_sq) in moves_from_iter.iter().enumerate() {
        assert_eq!(move_list.get_move(i).sq, *expected_sq);
    }
}

#[test]
fn test_concurrent_move_iterator_from_offset() {
    let board = Board::new();
    let move_list = MoveList::new(&board);
    let all_moves: Vec<Square> = move_list.iter().map(|m| m.sq).collect();

    // Create iterator starting from offset 2 (skip first 2 moves)
    let concurrent_iter = ConcurrentMoveIterator::from_offset(move_list, 2);

    assert_eq!(concurrent_iter.remaining(), 2);

    // First next() should return move at index 2 with move_count 3
    let (mv1, idx1) = concurrent_iter.next().unwrap();
    assert_eq!(mv1.sq, all_moves[2]);
    assert_eq!(idx1, 3); // 1-based: offset(2) + 1

    // Second next() should return move at index 3 with move_count 4
    let (mv2, idx2) = concurrent_iter.next().unwrap();
    assert_eq!(mv2.sq, all_moves[3]);
    assert_eq!(idx2, 4);

    // No more moves
    assert!(concurrent_iter.next().is_none());
    assert_eq!(concurrent_iter.remaining(), 0);
}

#[test]
fn test_concurrent_move_iterator_from_offset_zero() {
    let board = Board::new();
    let move_list = MoveList::new(&board);

    let concurrent_iter = ConcurrentMoveIterator::from_offset(move_list, 0);
    assert_eq!(concurrent_iter.remaining(), 4);

    let mut count = 0;
    while concurrent_iter.next().is_some() {
        count += 1;
    }
    assert_eq!(count, 4);
}

/// Independent eight-direction flip reference used to cross-check move
/// generation without depending on any SIMD backend.
fn reference_flip(sq: Square, p: u64, o: u64) -> u64 {
    const DIRECTIONS: [(i32, i32); 8] = [
        (1, 0),
        (0, 1),
        (1, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (-1, -1),
        (1, -1),
    ];

    let sq_idx = sq.index() as i32;
    let x = sq_idx & 7;
    let y = sq_idx >> 3;
    let mut flipped = 0u64;

    for (dx, dy) in DIRECTIONS {
        let mut cx = x + dx;
        let mut cy = y + dy;
        let mut line = 0u64;

        while (0..8).contains(&cx) && (0..8).contains(&cy) {
            let bit = 1u64 << (cy * 8 + cx);
            if o & bit != 0 {
                line |= bit;
            } else {
                if line != 0 && p & bit != 0 {
                    flipped |= line;
                }
                break;
            }
            cx += dx;
            cy += dy;
        }
    }

    flipped
}

/// Cross-checks generated moves against the legal-move bitboard and an
/// independent flip reference over a random board sweep. When run on an AVX-512
/// build (e.g. `cargo make test-x86-v4` on an AVX-512 host) this also exercises
/// the move-list `flip2_wide_load` kernel, which the dispatcher-level `flip`
/// tests never reach -- CI compiles the v4 backend but does not run it, so that
/// kernel is covered only when the suite runs on AVX-512 hardware.
#[test]
fn generated_moves_match_legal_bitboard_and_flip_reference() {
    let mut seed = 0xa1b2_c3d4_e5f6_0718u64;
    let mut saw_multi_move = false;

    for _ in 0..4096 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let p = seed;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let o = seed & !p; // keep player/opponent disjoint

        let board = Board::from_bitboards(p, o);
        let move_list = MoveList::new(&board);

        let legal = board.get_moves().bits();
        assert_eq!(move_list.count() as u32, legal.count_ones());
        if move_list.count() >= 2 {
            saw_multi_move = true;
        }

        let mut squares = 0u64;
        for mv in move_list.iter() {
            let bit = 1u64 << mv.sq.index();
            assert_eq!(legal & bit, bit, "generated an illegal move {:?}", mv.sq);
            squares |= bit;
            assert_eq!(
                mv.flipped.bits(),
                reference_flip(mv.sq, p, o),
                "flip mismatch at {:?} p={p:#018x} o={o:#018x}",
                mv.sq
            );
        }
        assert_eq!(
            squares, legal,
            "move set must equal the legal-move bitboard"
        );
    }

    assert!(saw_multi_move, "expected to exercise multi-move positions");
}

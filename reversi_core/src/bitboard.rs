use crate::bit;
use crate::square::Square;

/// Pre-computed masks for adjacent squares around each board position.
///
/// Each entry corresponds to the 8 squares adjacent to a given position on the board.
/// Used for checking if pieces have adjacent neighbors.
/// Reference: https://eukaryote.hateblo.jp/entry/2020/04/26/031246
#[rustfmt::skip]
const NEIGHBOUR_MASK: [u64; 64] = [
	0x0000000000000302, 0x0000000000000604, 0x0000000000000e0a, 0x0000000000001c14,
	0x0000000000003828, 0x0000000000007050, 0x0000000000006020, 0x000000000000c040,
	0x0000000000030200, 0x0000000000060400, 0x00000000000e0a00, 0x00000000001c1400,
	0x0000000000382800, 0x0000000000705000, 0x0000000000602000, 0x0000000000c04000,
	0x0000000003020300, 0x0000000006040600, 0x000000000e0a0e00, 0x000000001c141c00,
	0x0000000038283800, 0x0000000070507000, 0x0000000060206000, 0x00000000c040c000,
	0x0000000302030000, 0x0000000604060000, 0x0000000e0a0e0000, 0x0000001c141c0000,
	0x0000003828380000, 0x0000007050700000, 0x0000006020600000, 0x000000c040c00000,
	0x0000030203000000, 0x0000060406000000, 0x00000e0a0e000000, 0x00001c141c000000,
	0x0000382838000000, 0x0000705070000000, 0x0000602060000000, 0x0000c040c0000000,
	0x0003020300000000, 0x0006040600000000, 0x000e0a0e00000000, 0x001c141c00000000,
	0x0038283800000000, 0x0070507000000000, 0x0060206000000000, 0x00c040c000000000,
	0x0002030000000000, 0x0004060000000000, 0x000a0e0000000000, 0x00141c0000000000,
	0x0028380000000000, 0x0050700000000000, 0x0020600000000000, 0x0040c00000000000,
	0x0203000000000000, 0x0406000000000000, 0x0a0e000000000000, 0x141c000000000000,
	0x2838000000000000, 0x5070000000000000, 0x2060000000000000, 0x40c0000000000000,
];

/// Bitboard mask representing the four corner squares (A1, H1, A8, H8).
const CORNER_MASK: u64 = 0x8100000000000081;

/// Flips the player's bitboard at the specified square and the flipped bits.
///
/// # Arguments
///
/// * `b` - The player's bitboard.
/// * `flipped` - The bitboard representing the flipped pieces.
/// * `sq` - The index of the square (0-based).
///
/// # Returns
///
/// A `u64` value representing the new player's bitboard after the move.
#[inline]
pub fn player_flip(b: u64, flipped: u64, sq: Square) -> u64 {
    b ^ flipped ^ sq.bitboard()
}

/// Flips the opponent's bitboard at the specified flipped bits.
///
/// # Arguments
///
/// * `b` - The opponent's bitboard.
/// * `flipped` - The bitboard representing the flipped pieces.
///
/// # Returns
///
/// A `u64` value representing the new opponent's bitboard after the move.
#[inline]
pub fn opponent_flip(b: u64, flipped: u64) -> u64 {
    b ^ flipped
}

/// Sets a bit at the specified square in the bitboard.
///
/// # Arguments
///
/// * `b` - The bitboard.
/// * `sq` - The index of the square (0-based).
///
/// # Returns
///
/// A `u64` value with the bit set at the specified square.
#[inline]
pub fn set(b: u64, sq: Square) -> u64 {
    b | sq.bitboard()
}

/// Checks if a bit is set at the specified square in the bitboard.
///
/// # Arguments
///
/// * `b` - The bitboard.
/// * `sq` - The index of the square (0-based).
///
/// # Returns
///
/// `true` if the bit is set at the specified square, otherwise `false`.
#[inline]
pub fn is_set(b: u64, sq: Square) -> bool {
    b & sq.bitboard() != 0
}

/// Creates a bitboard representing the empty squares.
///
/// # Arguments
///
/// * `player` - The player's bitboard.
/// * `opponent` - The opponent's bitboard.
///
/// # Returns
///
/// A `u64` value representing the empty squares on the board.
#[inline]
pub fn empty_board(player: u64, opponent: u64) -> u64 {
    !(player | opponent)
}

/// Gets the possible moves for the player.
///
/// Reference: https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/board.c#L822
///
/// # Arguments
///
/// * `player` - The player's bitboard.
/// * `opponent` - The opponent's bitboard.
///
/// # Returns
///
/// A `u64` value representing the possible moves for the player.
#[inline]
pub fn get_moves(player: u64, opponent: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if cfg!(target_feature = "avx512vl") {
            return unsafe { get_moves_avx512(player, opponent) };
        } else if cfg!(target_feature = "avx2") {
            return unsafe { get_moves_avx2(player, opponent) };
        }
    }

    get_moves_fallback(player, opponent)
}

/// Fallback implementation of `get_moves` for architectures without AVX2 support.
///
/// # Arguments
///
/// * `player` - The player's bitboard.
/// * `opponent` - The opponent's bitboard.
///
/// # Returns
///
/// A `u64` value representing the possible moves for the player.
#[inline]
fn get_moves_fallback(player: u64, opponent: u64) -> u64 {
    let empty = empty_board(player, opponent);
    (get_some_moves(player, opponent & 0x007E7E7E7E7E7E00, 7) & empty)
        | (get_some_moves(player, opponent & 0x007E7E7E7E7E7E00, 9) & empty)
        | (get_some_moves(player, opponent & 0x7E7E7E7E7E7E7E7E, 1) & empty)
        | (get_some_moves(player, opponent & 0x00FFFFFFFFFFFF00, 8) & empty)
}

/// Gets the possible moves in a specific direction.
///
/// # Arguments
///
/// * `b` - The bitboard.
/// * `mask` - The mask for the direction.
/// * `dir` - The direction (in bits).
///
/// # Returns
///
/// A `u64` value representing the possible moves in the specified direction.
#[inline]
fn get_some_moves(b: u64, mask: u64, dir: u32) -> u64 {
    let mut flip = ((b << dir) | (b >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;

    (flip << dir) | (flip >> dir)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vl")]
#[inline]
pub fn get_moves_avx512(player: u64, opponent: u64) -> u64 {
    use std::arch::x86_64::*;

    let empty: u64 = !(player | opponent);

    let pp = _mm256_set1_epi64x(player as i64);
    let oo = _mm256_set1_epi64x(opponent as i64);

    let sh = _mm256_set_epi64x(7, 9, 8, 1);
    let masked_oo = _mm256_and_si256(
        oo,
        _mm256_set_epi64x(
            0x007E7E7E7E7E7E00u64 as i64, // NW
            0x007E7E7E7E7E7E00u64 as i64, // NE
            0x00FFFFFFFFFFFF00u64 as i64, // N
            0x7E7E7E7E7E7E7E7Eu64 as i64, // E
        ),
    );

    let mut fl = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(pp, sh)); // 左系（負方向）
    let mut fr = _mm256_and_si256(masked_oo, _mm256_srlv_epi64(pp, sh)); // 右系（正方向）

    let sh2 = _mm256_add_epi64(sh, sh);

    fl = _mm256_ternarylogic_epi64(fl, masked_oo, _mm256_sllv_epi64(fl, sh), 0xF8);
    fr = _mm256_ternarylogic_epi64(fr, masked_oo, _mm256_srlv_epi64(fr, sh), 0xF8);

    let pre = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(masked_oo, sh));

    fl = _mm256_ternarylogic_epi64(fl, pre, _mm256_sllv_epi64(fl, sh2), 0xF8);
    fr = _mm256_ternarylogic_epi64(
        fr,
        _mm256_srlv_epi64(pre, sh),
        _mm256_srlv_epi64(fr, sh2),
        0xF8,
    );

    fl = _mm256_ternarylogic_epi64(fl, pre, _mm256_sllv_epi64(fl, sh2), 0xF8);
    fr = _mm256_ternarylogic_epi64(
        fr,
        _mm256_srlv_epi64(pre, sh),
        _mm256_srlv_epi64(fr, sh2),
        0xF8,
    );

    let mm = _mm256_or_si256(_mm256_sllv_epi64(fl, sh), _mm256_srlv_epi64(fr, sh));

    let m128 = _mm_or_si128(_mm256_castsi256_si128(mm), _mm256_extracti128_si256(mm, 1));
    let moves64 = _mm_or_si128(m128, _mm_unpackhi_epi64(m128, m128));
    let moves = _mm_cvtsi128_si64(moves64) as u64;

    moves & empty
}

/// AVX2-optimized implementation of `get_moves`.
///
/// # Arguments
///
/// * `player` - The player's bitboard.
/// * `opponent` - The opponent's bitboard.
///
/// # Returns
///
/// A `u64` value representing the possible moves for the player.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn get_moves_avx2(player: u64, opponent: u64) -> u64 {
    use std::arch::x86_64::*;

    let pp = _mm256_broadcastq_epi64(_mm_cvtsi64_si128(player as i64));
    let oo = _mm256_broadcastq_epi64(_mm_cvtsi64_si128(opponent as i64));
    let shift1897 = _mm256_set_epi64x(7, 9, 8, 1);
    let mask = _mm256_set_epi64x(
        0x007E7E7E7E7E7E00,
        0x007E7E7E7E7E7E00,
        0x00FFFFFFFFFFFF00,
        0x7E7E7E7E7E7E7E7E,
    );
    let masked_oo = _mm256_and_si256(oo, mask);
    let occupied = _mm_or_si128(_mm256_castsi256_si128(pp), _mm256_castsi256_si128(oo));

    let flip_l = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(pp, shift1897));
    let flip_r = _mm256_and_si256(masked_oo, _mm256_srlv_epi64(pp, shift1897));
    let flip_l = _mm256_or_si256(
        flip_l,
        _mm256_and_si256(masked_oo, _mm256_sllv_epi64(flip_l, shift1897)),
    );
    let flip_r = _mm256_or_si256(
        flip_r,
        _mm256_and_si256(masked_oo, _mm256_srlv_epi64(flip_r, shift1897)),
    );
    let pre_l = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(masked_oo, shift1897));
    let pre_r = _mm256_srlv_epi64(pre_l, shift1897);
    let shift2 = _mm256_add_epi64(shift1897, shift1897);
    let flip_l = _mm256_or_si256(
        flip_l,
        _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)),
    );
    let flip_r = _mm256_or_si256(
        flip_r,
        _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)),
    );
    let flip_l = _mm256_or_si256(
        flip_l,
        _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)),
    );
    let flip_r = _mm256_or_si256(
        flip_r,
        _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)),
    );
    let mm = _mm256_or_si256(
        _mm256_sllv_epi64(flip_l, shift1897),
        _mm256_srlv_epi64(flip_r, shift1897),
    );

    let m = _mm_or_si128(_mm256_castsi256_si128(mm), _mm256_extracti128_si256(mm, 1));
    let masked = _mm_andnot_si128(occupied, _mm_or_si128(m, _mm_unpackhi_epi64(m, m)));
    _mm_cvtsi128_si64(masked) as u64
}

/// Counts the number of set bits in the bitboard, giving double weight to corner squares.
///
/// Reference: https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/board.c#L918C5-L918C26
///
/// # Arguments
///
/// * `b` - A 64-bit bitboard where each bit represents a square on the board.
///
/// # Returns
///
/// The total weighted count of set bits, with corner bits counted twice.
#[inline(always)]
pub fn corner_weighted_count(b: u64) -> u32 {
    b.count_ones() + (b & CORNER_MASK).count_ones()
}

/// Gets some potential moves in a specific direction.
///
/// # Arguments
///
/// * `o` - The opponent's bitboard.
/// * `dir` - The direction (in bits).
///
/// # Returns
///
/// A `u64` value representing some potential moves in the specified direction.
#[inline(always)]
fn get_some_potential_moves(o: u64, dir: u32) -> u64 {
    (o << dir) | (o >> dir)
}

/// Gets the potential moves for the player.
///
/// Reference: https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/board.c#L944
///
/// # Arguments
///
/// * `p` - The player's bitboard.
/// * `o` - The opponent's bitboard.
///
/// # Returns
///
/// A `u64` value representing the potential moves for the player.
#[inline(always)]
pub fn get_potential_moves(p: u64, o: u64) -> u64 {
    let h = get_some_potential_moves(o & 0x7E7E_7E7E_7E7E_7E7E_u64, 1);
    let v = get_some_potential_moves(o & 0x00FF_FFFF_FFFF_FF00_u64, 8);
    let d1 = get_some_potential_moves(o & 0x007E_7E7E_7E7E_7E00_u64, 7);
    let d2 = get_some_potential_moves(o & 0x007E_7E7E_7E7E_7E00_u64, 9);

    (h | v | d1 | d2) & !(p | o)
}

/// Counts the number of stable corners in the bitboard.
///
/// Reference: https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/board.c#L1453
///
/// # Arguments
///
/// * `p` - The player's bitboard.
///
/// # Returns
///
/// The number of stable corners.
#[inline]
pub fn get_corner_stability(p: u64) -> u32 {
    let stable: u64 = (((0x0100000000000001 & p) << 1)
        | ((0x8000000000000080 & p) >> 1)
        | ((0x0000000000000081 & p) << 8)
        | ((0x8100000000000000 & p) >> 8)
        | 0x8100000000000081)
        & p;
    stable.count_ones()
}

/// Checks if there is an adjacent bit set in the bitboard.
///
/// # Arguments
///
/// * `b` - The bitboard.
/// * `sq` - The index of the square (0-based).
///
/// # Returns
///
/// `true` if there is an adjacent bit set, otherwise `false`.
#[inline]
pub fn has_adjacent_bit(b: u64, sq: Square) -> bool {
    (b & NEIGHBOUR_MASK[sq.index()]) != 0
}

/// An iterator that yields each set bit position in a bitboard as a `Square`.
pub struct BitboardIterator {
    bitboard: u64,
}

impl BitboardIterator {
    /// Creates a new `BitboardIterator`.
    ///
    /// # Arguments
    ///
    /// * `bitboard` - The bitboard to iterate over.
    pub fn new(bitboard: u64) -> BitboardIterator {
        BitboardIterator { bitboard }
    }
}

impl Iterator for BitboardIterator {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bitboard == 0 {
            return None;
        }

        let square = Square::from_u32_unchecked(self.bitboard.trailing_zeros());
        self.bitboard = bit::clear_lsb_u64(self.bitboard);
        Some(square)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_player_flip() {
        let player_board: u64 = Square::A1.bitboard();
        let flipped: u64 = Square::B1.bitboard() | Square::C1.bitboard();
        let result = player_flip(player_board, flipped, Square::D1);

        // Should have original piece at A1, flipped pieces at B1 and C1, and new piece at D1
        assert!(is_set(result, Square::A1));
        assert!(is_set(result, Square::B1));
        assert!(is_set(result, Square::C1));
        assert!(is_set(result, Square::D1));
    }

    #[test]
    fn test_opponent_flip() {
        let opponent_board: u64 =
            Square::A1.bitboard() | Square::B1.bitboard() | Square::C1.bitboard();
        let flipped: u64 = Square::B1.bitboard() | Square::C1.bitboard();
        let result = opponent_flip(opponent_board, flipped);

        // Should only have piece at A1 (B1 and C1 were flipped away)
        assert!(is_set(result, Square::A1));
        assert!(!is_set(result, Square::B1));
        assert!(!is_set(result, Square::C1));
    }

    #[test]
    fn test_set_and_is_set() {
        let mut board: u64 = 0;

        // Test setting bits
        board = set(board, Square::A1);
        assert!(is_set(board, Square::A1));
        assert!(!is_set(board, Square::A2));

        board = set(board, Square::H8);
        assert!(is_set(board, Square::A1));
        assert!(is_set(board, Square::H8));
        assert!(!is_set(board, Square::D4));

        // Test setting already set bit
        board = set(board, Square::A1);
        assert!(is_set(board, Square::A1));
    }

    #[test]
    fn test_empty_board() {
        let player: u64 = Square::A1.bitboard() | Square::B2.bitboard();
        let opponent: u64 = Square::C3.bitboard() | Square::D4.bitboard();
        let empty = empty_board(player, opponent);

        // Should have all squares except A1, B2, C3, D4
        assert!(!is_set(empty, Square::A1));
        assert!(!is_set(empty, Square::B2));
        assert!(!is_set(empty, Square::C3));
        assert!(!is_set(empty, Square::D4));
        assert!(is_set(empty, Square::E5));
        assert!(is_set(empty, Square::H8));

        // Total should be 60 empty squares
        assert_eq!(empty.count_ones(), 60);
    }

    #[test]
    fn test_get_moves_initial_position() {
        // Standard Reversi initial position
        let player: u64 = Square::D5.bitboard() | Square::E4.bitboard();
        let opponent: u64 = Square::D4.bitboard() | Square::E5.bitboard();
        let moves = get_moves(player, opponent);

        // Valid moves for black (first player) in initial position
        assert!(is_set(moves, Square::C4));
        assert!(is_set(moves, Square::F5));
        assert!(is_set(moves, Square::D3));
        assert!(is_set(moves, Square::E6));
        assert_eq!(moves.count_ones(), 4);
    }

    #[test]
    fn test_get_moves_no_moves() {
        // Position where player has no moves
        let player: u64 = 0;
        let opponent: u64 = u64::MAX;
        let moves = get_moves(player, opponent);

        assert_eq!(moves, 0);
    }

    #[test]
    fn test_get_moves_capture_all_directions() {
        // Position where a move captures in all 8 directions
        // Center piece surrounded by opponent pieces
        let player: u64 = Square::A1.bitboard()
            | Square::H1.bitboard()
            | Square::A8.bitboard()
            | Square::H8.bitboard()
            | Square::A4.bitboard()
            | Square::H4.bitboard()
            | Square::D1.bitboard()
            | Square::D8.bitboard();
        let opponent: u64 = Square::B2.bitboard()
            | Square::C3.bitboard()
            | Square::E5.bitboard()
            | Square::F6.bitboard()
            | Square::G7.bitboard()
            | Square::D2.bitboard()
            | Square::D3.bitboard()
            | Square::D5.bitboard()
            | Square::D6.bitboard()
            | Square::D7.bitboard()
            | Square::B4.bitboard()
            | Square::C4.bitboard()
            | Square::E4.bitboard()
            | Square::F4.bitboard()
            | Square::G4.bitboard()
            | Square::C2.bitboard()
            | Square::E2.bitboard()
            | Square::F3.bitboard()
            | Square::C5.bitboard()
            | Square::B5.bitboard()
            | Square::B3.bitboard()
            | Square::F5.bitboard()
            | Square::E6.bitboard()
            | Square::C6.bitboard()
            | Square::B6.bitboard();

        let moves = get_moves(player, opponent);

        // D4 should be a valid move that captures in all directions
        assert!(is_set(moves, Square::D4));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_get_moves_consistency() {
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_avx512 =
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl");

        if !(has_avx2 || has_avx512) {
            // Host CPU does not expose either SIMD path; nothing to validate.
            return;
        }

        let test_positions = [
            (
                Square::D5.bitboard() | Square::E4.bitboard(),
                Square::D4.bitboard() | Square::E5.bitboard(),
            ),
            (0x00003C3C3C000000, 0x0000C3C3C3000000),
            (0xFF00000000000000, 0x00FF000000000000),
        ];

        for (player, opponent) in test_positions {
            let moves_fallback = get_moves_fallback(player, opponent);

            let moves_avx2 = if has_avx2 {
                Some(unsafe { get_moves_avx2(player, opponent) })
            } else {
                None
            };

            let moves_avx512 = if has_avx512 {
                Some(unsafe { get_moves_avx512(player, opponent) })
            } else {
                None
            };

            if let Some(moves) = moves_avx2 {
                assert_eq!(
                    moves_fallback, moves,
                    "Fallback and AVX2 implementations differ for player={player:016x}, opponent={opponent:016x}"
                );
            }

            if let Some(moves) = moves_avx512 {
                assert_eq!(
                    moves_fallback, moves,
                    "Fallback and AVX-512 implementations differ for player={player:016x}, opponent={opponent:016x}"
                );
            }

            if let (Some(avx2), Some(avx512)) = (moves_avx2, moves_avx512) {
                assert_eq!(
                    avx2, avx512,
                    "AVX2 and AVX-512 implementations differ for player={player:016x}, opponent={opponent:016x}"
                );
            }
        }
    }

    #[test]
    fn test_corner_stability() {
        // No corners
        let board: u64 = Square::D4.bitboard() | Square::E5.bitboard();
        assert_eq!(get_corner_stability(board), 0);

        // One corner (A1)
        let board: u64 = Square::A1.bitboard();
        assert_eq!(get_corner_stability(board), 1);

        // All corners - the function checks for stable corners which includes
        // corners that are protected by adjacent corners
        let board: u64 = CORNER_MASK;
        assert_eq!(get_corner_stability(board), 4);

        // Corner with adjacent pieces - A1 with A2 and B1
        // The function counts corners that form stable groups
        let board: u64 = Square::A1.bitboard()
            | Square::A2.bitboard()
            | Square::B1.bitboard()
            | Square::B2.bitboard();
        assert_eq!(get_corner_stability(board), 3); // A1, A2, B1 form a stable group
    }

    #[test]
    fn test_bitboard_iterator() {
        // Example bitboard: bits 0, 1, and 63 are set
        let bitboard: u64 = Square::A1.bitboard() | Square::B1.bitboard() | Square::H8.bitboard();
        let mut iterator = BitboardIterator::new(bitboard);

        assert_eq!(iterator.next(), Some(Square::A1));
        assert_eq!(iterator.next(), Some(Square::B1));
        assert_eq!(iterator.next(), Some(Square::H8));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_bitboard_iterator_empty() {
        let bitboard: u64 = 0;
        let mut iterator = BitboardIterator::new(bitboard);
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_bitboard_iterator_full() {
        let bitboard: u64 = u64::MAX;
        let count = bitboard.count_ones();
        let mut iterator = BitboardIterator::new(bitboard);
        for i in 0..count {
            assert_eq!(iterator.next(), Some(Square::from_u32_unchecked(i)));
        }
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_bitboard_iterator_single_bit() {
        let bitboard: u64 = Square::E4.bitboard();
        let mut iterator = BitboardIterator::new(bitboard);
        assert_eq!(iterator.next(), Some(Square::E4));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_bitboard_iterator_diagonal() {
        // Main diagonal A1-H8
        let bitboard: u64 = 0x8040201008040201;
        let squares: Vec<Square> = BitboardIterator::new(bitboard).collect();
        assert_eq!(squares.len(), 8);
        assert_eq!(squares[0], Square::A1);
        assert_eq!(squares[7], Square::H8);
    }

    #[test]
    fn test_corner_weighted_count_only_corners() {
        let bitboard: u64 = Square::A1.bitboard()
            | Square::H1.bitboard()
            | Square::A8.bitboard()
            | Square::H8.bitboard();
        assert_eq!(corner_weighted_count(bitboard), 8);
    }

    #[test]
    fn test_corner_weighted_count_mixed() {
        // Two corners and two regular squares
        let bitboard: u64 = Square::A1.bitboard()
            | Square::H8.bitboard()
            | Square::D4.bitboard()
            | Square::E5.bitboard();
        assert_eq!(corner_weighted_count(bitboard), 6); // 4 regular + 2 corner bonus
    }

    #[test]
    fn test_corner_weighted_count_no_corners() {
        // No corners
        let bitboard: u64 = Square::D4.bitboard()
            | Square::E4.bitboard()
            | Square::D5.bitboard()
            | Square::E5.bitboard();
        assert_eq!(corner_weighted_count(bitboard), 4);
    }

    #[test]
    fn test_corner_weighted_count_all_bits() {
        let bitboard: u64 = u64::MAX;
        assert_eq!(corner_weighted_count(bitboard), 68);
    }

    #[test]
    fn test_has_adjacent_bit() {
        let bitboard: u64 = Square::B2.bitboard();
        assert!(has_adjacent_bit(bitboard, Square::A1));
        assert!(has_adjacent_bit(bitboard, Square::A2));
        assert!(has_adjacent_bit(bitboard, Square::A3));
        assert!(!has_adjacent_bit(bitboard, Square::A4));
        assert!(has_adjacent_bit(bitboard, Square::B1));
        assert!(!has_adjacent_bit(bitboard, Square::B2));
        assert!(has_adjacent_bit(bitboard, Square::B3));
        assert!(!has_adjacent_bit(bitboard, Square::B4));
        assert!(has_adjacent_bit(bitboard, Square::C1));
        assert!(has_adjacent_bit(bitboard, Square::C2));
        assert!(has_adjacent_bit(bitboard, Square::C3));
        assert!(!has_adjacent_bit(bitboard, Square::C4));
        assert!(!has_adjacent_bit(bitboard, Square::D1));
    }

    #[test]
    fn test_has_adjacent_bit_corners() {
        // Test corner adjacency
        let bitboard: u64 = Square::B1.bitboard() | Square::A2.bitboard();
        assert!(has_adjacent_bit(bitboard, Square::A1));

        let bitboard: u64 = Square::G8.bitboard() | Square::H7.bitboard();
        assert!(has_adjacent_bit(bitboard, Square::H8));
    }

    #[test]
    fn test_has_adjacent_bit_edge() {
        // Test edge square adjacency
        let bitboard: u64 = Square::C1.bitboard() | Square::D2.bitboard() | Square::E1.bitboard();
        assert!(has_adjacent_bit(bitboard, Square::D1));
    }
}

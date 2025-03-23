use crate::bit;
use crate::square::Square;
use std::arch::x86_64::*;

/// The mask for the adjacent squares.
/// https://eukaryote.hateblo.jp/entry/2020/04/26/031246
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

/// The mask for the corner squares.
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
    if is_x86_feature_detected!("avx2") {
        unsafe { get_moves_avx(player, opponent) }
    } else {
        get_moves_fallback(player, opponent)
    }
}

/// Fallback implementation of `get_moves` for architectures without AVX2 support.
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

/// AVX2-optimized implementation of `get_moves`.
#[inline]
unsafe fn get_moves_avx(player: u64, opponent: u64) -> u64 {
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
/// # Arguments
///
/// * `b` - A 64-bit bitboard where each bit represents a square on the board.
///
/// # Returns
///
/// The total weighted count of set bits, with corner bits counted twice.
#[inline]
pub fn corner_weighted_mobility(b: u64) -> u32 {
    b.count_ones() + (b & CORNER_MASK).count_ones()
}

/// Counts the number of stable corners in the bitboard.
///
/// # Arguments
///
/// * `p` - The player's bitboard.
///
/// # Returns
///
/// The number of stable corners.
#[inline]
pub fn get_corner_stability(p: u64) -> u32
{
	let stable : u64 = (((0x0100000000000001 & p) << 1) | ((0x8000000000000080 & p) >> 1)
	                      | ((0x0000000000000081 & p) << 8) | ((0x8100000000000000 & p) >> 8)
	                      | 0x8100000000000081) & p;
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
    (b & NEIGHBOUR_MASK[sq as usize]) != 0
}

/// An iterator over the bits in a bitboard.
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

        let square = Square::from_usize_unchecked(self.bitboard.trailing_zeros() as usize);
        self.bitboard = bit::clear_lsb_u64(self.bitboard);
        Some(square)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            assert_eq!(iterator.next(), Some(Square::from_usize_unchecked(i as usize)));
        }
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_corner_weighted_count_only_corners() {
        let bitboard: u64 = Square::A1.bitboard()
            | Square::H1.bitboard()
            | Square::A8.bitboard()
            | Square::H8.bitboard();
        assert_eq!(corner_weighted_mobility(bitboard), 8);
    }

    #[test]
    fn test_corner_weighted_count_all_bits() {
        let bitboard: u64 = u64::MAX;
        assert_eq!(corner_weighted_mobility(bitboard), 68);
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
}

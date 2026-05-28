//! Bitboard operations and types.
//!
//! This module provides a [`Bitboard`] type that represents a 64-square Reversi board
//! using a single `u64`, where each bit corresponds to a square (bit 0 = A1, bit 63 = H8).

use crate::square::Square;

mod movegen;

const A1_MASK: u64 = 0x0000000000000001;
const H1_MASK: u64 = 0x0000000000000080;
const A8_MASK: u64 = 0x0100000000000000;
const H8_MASK: u64 = 0x8000000000000000;

/// Bitboard mask representing the four corner squares.
const CORNER_MASK: u64 = A1_MASK | H1_MASK | A8_MASK | H8_MASK;
/// Horizontal: excludes files A and H.
const HORIZONTAL_MASK: u64 = 0x7E7E7E7E7E7E7E7E;
/// Vertical: excludes ranks 1 and 8.
const VERTICAL_MASK: u64 = 0x00FFFFFFFFFFFF00;
/// Diagonal: excludes all edge files and ranks.
const DIAGONAL_MASK: u64 = 0x007E7E7E7E7E7E00;

/// Newtype wrapper for a 64-bit bitboard (bit 0 = A1, bit 63 = H8).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Bitboard(u64);

impl Bitboard {
    /// Creates a new bitboard from raw bits.
    #[inline(always)]
    pub const fn new(bits: u64) -> Self {
        Bitboard(bits)
    }

    /// Returns the raw 64-bit value.
    #[inline(always)]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Creates a bitboard with a single bit set at the given square.
    #[inline(always)]
    pub const fn from_square(sq: Square) -> Self {
        Bitboard(1 << sq as u8)
    }

    /// Returns a new bitboard with the bit at the given square set.
    #[inline(always)]
    pub fn set(self, sq: Square) -> Self {
        Bitboard(self.0 | sq.bitboard().0)
    }

    /// Returns a new bitboard with the bit at the given square cleared.
    #[inline(always)]
    pub fn remove(self, sq: Square) -> Self {
        Bitboard(self.0 & !sq.bitboard().0)
    }

    /// Checks whether the given square's bit is set.
    #[inline(always)]
    pub fn contains(self, sq: Square) -> bool {
        self.0 & sq.bitboard().0 != 0
    }

    /// Checks whether the bitboard has no bits set.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Checks whether exactly one bit is set, assuming the bitboard is non-empty.
    #[inline(always)]
    pub const fn has_single_bit_nonzero(self) -> bool {
        (self.0 & self.0.wrapping_sub(1)) == 0
    }

    /// Returns the number of set bits (population count).
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Returns a new bitboard with the least significant bit cleared.
    #[inline(always)]
    pub const fn clear_lsb(self) -> Self {
        Bitboard(self.0 & self.0.wrapping_sub(1))
    }

    /// Returns the [`Square`] corresponding to the least significant set bit,
    /// or [`None`] if the bitboard is empty.
    #[inline(always)]
    pub fn lsb_square(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            // SAFETY: bitboard is non-empty, so trailing_zeros() is 0..=63.
            Some(unsafe { Square::from_u32_unchecked(self.0.trailing_zeros()) })
        }
    }

    /// Returns the [`Square`] corresponding to the least significant set bit.
    ///
    /// # Panics
    ///
    /// Panics if the bitboard is empty in debug mode.
    #[inline(always)]
    pub fn lsb_square_unchecked(self) -> Square {
        debug_assert!(
            !self.is_empty(),
            "lsb_square_unchecked called on empty bitboard"
        );
        // SAFETY: caller ensures non-empty; trailing_zeros() on a non-zero u64 is 0..=63.
        unsafe { Square::from_u32_unchecked(self.0.trailing_zeros()) }
    }

    /// Removes and returns the least significant set bit as a [`Square`],
    /// along with the updated bitboard.
    ///
    /// # Panics
    ///
    /// Panics if the bitboard is empty in debug mode.
    #[inline(always)]
    pub fn pop_lsb(self) -> (Square, Self) {
        debug_assert!(!self.is_empty(), "pop_lsb called on empty bitboard");
        (self.lsb_square_unchecked(), self.clear_lsb())
    }

    /// Flips the bitboard vertically (rank 1 ↔ rank 8, etc.).
    #[inline(always)]
    pub fn flip_vertical(self) -> Self {
        Bitboard(self.0.swap_bytes())
    }

    /// Flips the bitboard horizontally (file A ↔ file H, etc.).
    #[inline(always)]
    pub fn flip_horizontal(self) -> Self {
        const MASK1: u64 = 0x5555555555555555;
        const MASK2: u64 = 0x3333333333333333;
        const MASK3: u64 = 0x0f0f0f0f0f0f0f0f;

        let mut b = self.0;
        b = ((b >> 1) & MASK1) | ((b & MASK1) << 1);
        b = ((b >> 2) & MASK2) | ((b & MASK2) << 2);
        b = ((b >> 4) & MASK3) | ((b & MASK3) << 4);
        Bitboard(b)
    }

    /// Flips the bitboard along the A1-H8 diagonal.
    #[inline(always)]
    pub fn flip_diag_a1h8(self) -> Self {
        const MASK1: u64 = 0x5500550055005500;
        const MASK2: u64 = 0x3333000033330000;
        const MASK3: u64 = 0x0f0f0f0f00000000;

        let mut bits = self.0;
        bits = delta_swap(bits, MASK3, 28);
        bits = delta_swap(bits, MASK2, 14);
        bits = delta_swap(bits, MASK1, 7);
        Bitboard(bits)
    }

    /// Flips the bitboard along the A8-H1 diagonal.
    #[inline(always)]
    pub fn flip_diag_a8h1(self) -> Self {
        const MASK1: u64 = 0xaa00aa00aa00aa00;
        const MASK2: u64 = 0xcccc0000cccc0000;
        const MASK3: u64 = 0xf0f0f0f000000000;

        let mut bits = self.0;
        bits = delta_swap(bits, MASK3, 36);
        bits = delta_swap(bits, MASK2, 18);
        bits = delta_swap(bits, MASK1, 9);
        Bitboard(bits)
    }

    /// Rotates the bitboard 90 degrees clockwise.
    #[inline(always)]
    pub fn rotate_90_clockwise(self) -> Self {
        self.flip_diag_a8h1().flip_vertical()
    }

    /// Rotates the bitboard 180 degrees.
    #[inline(always)]
    pub fn rotate_180_clockwise(self) -> Self {
        Bitboard(self.0.reverse_bits())
    }

    /// Rotates the bitboard 270 degrees clockwise (90 degrees counter-clockwise).
    #[inline(always)]
    pub fn rotate_270_clockwise(self) -> Self {
        self.flip_diag_a1h8().flip_vertical()
    }

    /// Checks whether any bit is adjacent to `sq` in a bracketable direction.
    ///
    /// Edge-adjacent directions with no square beyond the adjacent bit are excluded
    /// because they cannot produce a legal Reversi flip.
    #[inline(always)]
    pub fn has_adjacent_bit(self, sq: Square) -> bool {
        /// Pre-computed masks for bracketable adjacent squares around each board position.
        /// Reference: <https://eukaryote.hateblo.jp/entry/2020/04/26/031246>
        #[rustfmt::skip]
        const NEIGHBOUR_MASK: [u64; 65] = [
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
            0 // Square::None
        ];

        (self.0 & unsafe { *NEIGHBOUR_MASK.get_unchecked(sq.index()) }) != 0
    }

    /// Returns the population count with corner squares weighted double.
    ///
    /// Reference: <https://github.com/abulmo/edax-reversi/blob/master/src/bit.c#L237>
    #[inline(always)]
    pub fn corner_weighted_count(self) -> u32 {
        self.count() + self.corners().count()
    }

    /// Returns a new bitboard with only the corner squares (A1, H1, A8, H8).
    #[inline(always)]
    pub const fn corners(self) -> Self {
        Bitboard(self.0 & CORNER_MASK)
    }

    /// Returns a new bitboard with corner squares cleared.
    #[inline(always)]
    pub const fn non_corners(self) -> Self {
        Bitboard(self.0 & !CORNER_MASK)
    }

    /// Returns an iterator over all set squares in LSB-first order.
    #[inline(always)]
    pub fn iter(self) -> BitboardIterator {
        BitboardIterator::new(self)
    }

    /// Returns a new bitboard after applying a player's move.
    ///
    /// XORs the current bitboard with both the flipped discs and the placed disc.
    #[inline(always)]
    pub fn apply_move(self, flipped: Bitboard, sq: Square) -> Bitboard {
        self ^ flipped ^ sq.bitboard()
    }

    /// Returns a new bitboard after toggling the flipped discs.
    #[inline(always)]
    pub fn apply_flip(self, flipped: Bitboard) -> Bitboard {
        self ^ flipped
    }

    /// Returns the number of stable discs around corners.
    ///
    /// Counts corners plus adjacent edge squares that form stable groups.
    /// A corner is stable if occupied. An edge square adjacent to a corner
    /// is stable if both it and the corner are occupied by the same player.
    ///
    /// Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/board.c#L1453>
    #[inline(always)]
    pub fn corner_stability(self) -> u32 {
        let p = self.0;
        const A_FILE_CORNERS: u64 = A1_MASK | A8_MASK;
        const H_FILE_CORNERS: u64 = H1_MASK | H8_MASK;
        const RANK1_CORNERS: u64 = A1_MASK | H1_MASK;
        const RANK8_CORNERS: u64 = A8_MASK | H8_MASK;

        let stable: u64 = (((A_FILE_CORNERS & p) << 1)
            | ((H_FILE_CORNERS & p) >> 1)
            | ((RANK1_CORNERS & p) << 8)
            | ((RANK8_CORNERS & p) >> 8)
            | CORNER_MASK)
            & p;
        stable.count_ones()
    }

    /// Returns the legal moves for the player given the opponent's bitboard.
    #[inline(always)]
    pub fn get_moves(self, opponent: Bitboard) -> Bitboard {
        Bitboard(movegen::get_moves(self.0, opponent.0))
    }

    /// Returns the potential moves for the player.
    ///
    /// Potential moves are empty squares next to an opponent disc in a direction
    /// where a bracketing player disc could exist beyond that opponent disc.
    #[inline(always)]
    pub fn get_potential_moves(self, opponent: Bitboard) -> Bitboard {
        Bitboard(movegen::get_potential_moves(self.0, opponent.0))
    }

    /// Returns both the legal moves and potential moves for the current player.
    ///
    /// More efficient than calling [`get_moves`](Self::get_moves) and
    /// [`get_potential_moves`](Self::get_potential_moves) separately.
    #[inline(always)]
    pub fn get_moves_and_potential(self, opponent: Bitboard) -> (Bitboard, Bitboard) {
        let (m, p) = movegen::get_moves_and_potential(self.0, opponent.0);
        (Bitboard(m), Bitboard(p))
    }
}

// Operator trait implementations

impl std::ops::BitAnd for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Bitboard(self.0 & rhs.0)
    }
}

impl std::ops::BitOr for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Bitboard(self.0 | rhs.0)
    }
}

impl std::ops::BitXor for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl std::ops::Not for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Bitboard(!self.0)
    }
}

impl std::ops::Shl<u32> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u32) -> Self::Output {
        Bitboard(self.0 << rhs)
    }
}

impl std::ops::Shr<u32> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u32) -> Self::Output {
        Bitboard(self.0 >> rhs)
    }
}

impl std::ops::BitAndAssign for Bitboard {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl std::ops::BitOrAssign for Bitboard {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitXorAssign for Bitboard {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl std::ops::ShlAssign<u32> for Bitboard {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: u32) {
        self.0 <<= rhs;
    }
}

impl std::ops::ShrAssign<u32> for Bitboard {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: u32) {
        self.0 >>= rhs;
    }
}

// Conversion trait implementations

impl From<u64> for Bitboard {
    #[inline(always)]
    fn from(bits: u64) -> Self {
        Bitboard(bits)
    }
}

impl From<Bitboard> for u64 {
    #[inline(always)]
    fn from(bb: Bitboard) -> Self {
        bb.0
    }
}

impl From<Square> for Bitboard {
    #[inline(always)]
    fn from(sq: Square) -> Self {
        sq.bitboard()
    }
}

// Iterator support

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = BitboardIterator;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        BitboardIterator::new(self)
    }
}

// Display trait

impl std::fmt::Display for Bitboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = rank * 8 + file;
                if (self.0 >> sq) & 1 != 0 {
                    write!(f, "1")?;
                } else {
                    write!(f, ".")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Swaps bit pairs separated by `delta` positions where `mask` has bits set.
#[inline(always)]
fn delta_swap(bits: u64, mask: u64, delta: u32) -> u64 {
    let tmp = mask & (bits ^ (bits << delta));
    bits ^ tmp ^ (tmp >> delta)
}

/// An iterator that yields each set bit position in a bitboard as a [`Square`].
pub struct BitboardIterator {
    bitboard: Bitboard,
}

impl BitboardIterator {
    /// Creates a new [`BitboardIterator`].
    #[inline(always)]
    pub fn new(bitboard: Bitboard) -> BitboardIterator {
        BitboardIterator { bitboard }
    }
}

impl Iterator for BitboardIterator {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.bitboard.is_empty() {
            return None;
        }

        let (square, rest) = self.bitboard.pop_lsb();
        self.bitboard = rest;
        Some(square)
    }
}

#[cfg(test)]
mod movegen_tests;

#[cfg(test)]
mod tests;

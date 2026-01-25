//! Bitboard operations and types.
//!
//! This module provides a [`Bitboard`] type that represents a 64-square Reversi board
//! using a single `u64`, where each bit corresponds to a square (bit 0 = A1, bit 63 = H8).

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

use cfg_if::cfg_if;

use crate::square::Square;

/// Bitboard mask representing the four corner squares (A1, H1, A8, H8).
const CORNER_MASK: u64 = 0x8100000000000081;

/// Newtype wrapper for a 64-bit bitboard (bit 0 = A1, bit 63 = H8).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Bitboard(u64);

impl Bitboard {
    /// Creates a new bitboard from raw bits.
    ///
    /// # Arguments
    ///
    /// * `bits` - Raw 64-bit value where each bit represents a square.
    ///
    /// # Returns
    ///
    /// A new `Bitboard` wrapping the given bits.
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
    ///
    /// # Arguments
    ///
    /// * `sq` - The square to set.
    ///
    /// # Returns
    ///
    /// A new `Bitboard` with only the specified square's bit set.
    #[inline(always)]
    pub const fn from_square(sq: Square) -> Self {
        Bitboard(1 << sq as u8)
    }

    /// Returns a new bitboard with the bit at the given square set.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square to set.
    ///
    /// # Returns
    ///
    /// A new `Bitboard` with the specified square's bit set.
    #[inline(always)]
    pub fn set(self, sq: Square) -> Self {
        Bitboard(self.0 | sq.bitboard().0)
    }

    /// Returns a new bitboard with the bit at the given square removed.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square to clear.
    ///
    /// # Returns
    ///
    /// A new `Bitboard` with the specified square's bit cleared.
    #[inline(always)]
    pub fn remove(self, sq: Square) -> Self {
        Bitboard(self.0 & !sq.bitboard().0)
    }

    /// Checks if the bitboard contains the bit at the given square.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square to check.
    ///
    /// # Returns
    ///
    /// `true` if the specified square's bit is set, `false` otherwise.
    #[inline(always)]
    pub fn contains(self, sq: Square) -> bool {
        self.0 & sq.bitboard().0 != 0
    }

    /// Checks if the bitboard has no bits set.
    ///
    /// # Returns
    ///
    /// `true` if no bits are set, `false` otherwise.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the number of set bits (population count).
    ///
    /// # Returns
    ///
    /// The number of bits set in the bitboard (0-64).
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Returns a new bitboard with the least significant bit cleared.
    ///
    /// # Returns
    ///
    /// A new `Bitboard` with the LSB cleared.
    #[inline(always)]
    pub const fn clear_lsb(self) -> Self {
        Bitboard(self.0 & self.0.wrapping_sub(1))
    }

    /// Returns the square corresponding to the least significant set bit.
    ///
    /// # Returns
    ///
    /// `Some(Square)` for the LSB position, or `None` if the bitboard is empty.
    #[inline(always)]
    pub fn lsb_square(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(Square::from_u32_unchecked(self.0.trailing_zeros()))
        }
    }

    /// Returns the square corresponding to the least significant set bit.
    ///
    /// # Returns
    ///
    /// The `Square` corresponding to the LSB position.
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
        Square::from_u32_unchecked(self.0.trailing_zeros())
    }

    /// Removes and returns the least significant set bit as a square,
    /// along with the updated bitboard.
    ///
    /// # Returns
    ///
    /// A tuple of `(Square, Bitboard)` where the square is the LSB position
    /// and the bitboard has that bit cleared.
    ///
    /// # Panics
    ///
    /// Panics if the bitboard is empty in debug mode.
    #[inline(always)]
    pub fn pop_lsb(self) -> (Square, Self) {
        debug_assert!(!self.is_empty(), "pop_lsb called on empty bitboard");
        (self.lsb_square_unchecked(), self.clear_lsb())
    }

    /// Flips the bitboard vertically (swaps ranks 1-8).
    ///
    /// # Returns
    ///
    /// A new `Bitboard` with ranks mirrored (rank 1 ↔ rank 8, etc.).
    #[inline(always)]
    pub fn flip_vertical(self) -> Self {
        Bitboard(self.0.swap_bytes())
    }

    /// Flips the bitboard horizontally (swaps files A-H).
    ///
    /// # Returns
    ///
    /// A new `Bitboard` with files mirrored (file A ↔ file H, etc.).
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
    ///
    /// # Returns
    ///
    /// A new `Bitboard` transposed along the main diagonal.
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
    ///
    /// # Returns
    ///
    /// A new `Bitboard` transposed along the anti-diagonal.
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
    ///
    /// # Returns
    ///
    /// A new `Bitboard` rotated 90° clockwise.
    #[inline(always)]
    pub fn rotate_90_clockwise(self) -> Self {
        self.flip_diag_a8h1().flip_vertical()
    }

    /// Rotates the bitboard 180 degrees.
    ///
    /// # Returns
    ///
    /// A new `Bitboard` rotated 180°.
    #[inline(always)]
    pub fn rotate_180_clockwise(self) -> Self {
        Bitboard(self.0.reverse_bits())
    }

    /// Rotates the bitboard 270 degrees clockwise (90 degrees counter-clockwise).
    ///
    /// # Returns
    ///
    /// A new `Bitboard` rotated 270° clockwise (or 90° counter-clockwise).
    #[inline(always)]
    pub fn rotate_270_clockwise(self) -> Self {
        self.flip_diag_a1h8().flip_vertical()
    }

    /// Checks if there is an adjacent bit set in the bitboard.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square to check adjacency for.
    ///
    /// # Returns
    ///
    /// `true` if there is an adjacent bit set, otherwise `false`.
    #[inline(always)]
    pub fn has_adjacent_bit(self, sq: Square) -> bool {
        /// Pre-computed masks for adjacent squares around each board position.
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

    /// Counts the number of set bits in the bitboard, giving double weight to corner squares.
    ///
    /// Reference: <https://github.com/abulmo/edax-reversi/blob/master/src/bit.c#L237>
    ///
    /// # Returns
    ///
    /// The total weighted count of set bits, with corner bits counted twice.
    #[inline(always)]
    pub fn corner_weighted_count(self) -> u32 {
        self.count() + self.corners().count()
    }

    /// Returns a new bitboard with only the corner squares (A1, H1, A8, H8).
    ///
    /// This applies the [`CORNER_MASK`] to extract only the corner bits.
    ///
    /// # Returns
    ///
    /// A new `Bitboard` containing only the corner bits from the original.
    #[inline(always)]
    pub const fn corners(self) -> Self {
        Bitboard(self.0 & CORNER_MASK)
    }

    /// Returns a new bitboard with only the non-corner squares.
    ///
    /// This applies the inverse of [`CORNER_MASK`] to exclude corner bits.
    ///
    /// # Returns
    ///
    /// A new `Bitboard` with the corner bits cleared.
    #[inline(always)]
    pub const fn non_corners(self) -> Self {
        Bitboard(self.0 & !CORNER_MASK)
    }

    /// Returns an iterator over all set squares in the bitboard.
    ///
    /// # Returns
    ///
    /// A [`BitboardIterator`] that yields each set square in LSB-first order.
    #[inline(always)]
    pub fn iter(self) -> BitboardIterator {
        BitboardIterator::new(self)
    }

    /// Returns a new bitboard after applying a player's move.
    ///
    /// XORs the current bitboard with both the flipped discs and the placed disc.
    ///
    /// # Arguments
    ///
    /// * `flipped` - Bitboard of opponent discs flipped by this move.
    /// * `sq` - Square where the disc was placed.
    ///
    /// # Returns
    ///
    /// A new `Bitboard` representing the player's discs after the move.
    #[inline(always)]
    pub fn apply_move(self, flipped: Bitboard, sq: Square) -> Bitboard {
        self ^ flipped ^ sq.bitboard()
    }

    /// Returns a new bitboard after applying a flip.
    ///
    /// XORs the current bitboard with the flipped discs.
    ///
    /// # Arguments
    ///
    /// * `flipped` - Bitboard of discs flipped by the move.
    ///
    /// # Returns
    ///
    /// A new `Bitboard` with the flipped discs toggled.
    #[inline(always)]
    pub fn apply_flip(self, flipped: Bitboard) -> Bitboard {
        self ^ flipped
    }

    /// Returns the number of stable discs around corners in this bitboard.
    ///
    /// Counts corners plus adjacent edge squares that form stable groups.
    /// A corner is stable if occupied. An edge square adjacent to a corner
    /// is stable if both it and the corner are occupied by the same player.
    ///
    /// Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/board.c#L1453>
    ///
    /// # Returns
    ///
    /// The count of stable discs (0-12: up to 4 corners + 8 adjacent edge squares).
    #[inline(always)]
    pub fn corner_stability(self) -> u32 {
        let p = self.0;
        let stable: u64 = (((0x0100000000000001 & p) << 1)
            | ((0x8000000000000080 & p) >> 1)
            | ((0x0000000000000081 & p) << 8)
            | ((0x8100000000000000 & p) >> 8)
            | 0x8100000000000081)
            & p;
        stable.count_ones()
    }

    /// Gets the legal moves for the player.
    ///
    /// # Arguments
    ///
    /// * `opponent` - The opponent's bitboard.
    ///
    /// # Returns
    ///
    /// A `Bitboard` with bits set for each legal move position.
    #[inline(always)]
    pub fn get_moves(self, opponent: Bitboard) -> Bitboard {
        Bitboard(get_moves(self.0, opponent.0))
    }

    /// Gets the potential moves for the player.
    ///
    /// Potential moves are empty squares that are adjacent (including diagonally) to at least
    /// one opponent disc.
    ///
    /// # Arguments
    ///
    /// * `opponent` - The opponent's bitboard.
    ///
    /// # Returns
    ///
    /// A `Bitboard` with bits set for each potential move position.
    #[inline(always)]
    pub fn get_potential_moves(self, opponent: Bitboard) -> Bitboard {
        Bitboard(get_potential_moves(self.0, opponent.0))
    }

    /// Gets both the legal moves and potential moves for the current player.
    ///
    /// This is more efficient than calling [`get_moves`](Self::get_moves) and
    /// [`get_potential_moves`](Self::get_potential_moves) separately.
    ///
    /// # Arguments
    ///
    /// * `opponent` - The opponent's bitboard.
    ///
    /// # Returns
    ///
    /// A tuple of `(legal_moves, potential_moves)`.
    #[inline(always)]
    pub fn get_moves_and_potential(self, opponent: Bitboard) -> (Bitboard, Bitboard) {
        let (m, p) = get_moves_and_potential(self.0, opponent.0);
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

/// Gets the legal moves for the player.
///
/// Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/board.c#L822>
///
/// # Arguments
///
/// * `player` - The player's bitboard.
/// * `opponent` - The opponent's bitboard.
///
/// # Returns
///
/// A `u64` value representing the legal moves for the player.
#[inline(always)]
fn get_moves(player: u64, opponent: u64) -> u64 {
    cfg_if! {
        if #[cfg(all(target_arch = "x86_64", target_feature = "avx512vl"))] {
            unsafe { get_moves_avx512(player, opponent) }
        } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
            unsafe { get_moves_avx2(player, opponent) }
        } else if #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] {
            get_moves_wasm(player, opponent)
        } else {
            get_moves_fallback(player, opponent)
        }
    }
}

/// Fallback implementation of `get_moves` for architectures without SIMD support.
///
/// # Arguments
///
/// * `player` - The player's bitboard.
/// * `opponent` - The opponent's bitboard.
///
/// # Returns
///
/// A `u64` value representing the legal moves for the player.
#[inline(always)]
#[allow(dead_code)]
fn get_moves_fallback(player: u64, opponent: u64) -> u64 {
    let empty = !(player | opponent);
    (get_some_moves(player, opponent & 0x007E7E7E7E7E7E00, 7) & empty)
        | (get_some_moves(player, opponent & 0x007E7E7E7E7E7E00, 9) & empty)
        | (get_some_moves(player, opponent & 0x7E7E7E7E7E7E7E7E, 1) & empty)
        | (get_some_moves(player, opponent & 0x00FFFFFFFFFFFF00, 8) & empty)
}

/// Propagates flipped discs in a specific direction.
///
/// This is a helper function for move generation that calculates the flip propagation
/// along a ray direction. The result is then used to determine where legal moves exist.
///
/// # Arguments
///
/// * `b` - The player's bitboard.
/// * `mask` - The mask for the direction (opponent's discs with edge masking).
/// * `dir` - The direction (in bits).
///
/// # Returns
///
/// A `u64` value representing the flip propagation in the specified direction.
#[inline(always)]
#[allow(dead_code)]
fn get_some_moves(b: u64, mask: u64, dir: u32) -> u64 {
    let mut flip = ((b << dir) | (b >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    flip |= ((flip << dir) | (flip >> dir)) & mask;
    (flip << dir) | (flip >> dir)
}

/// AVX-512-optimized implementation of `get_moves`.
///
/// # Arguments
///
/// * `player` - The player's bitboard.
/// * `opponent` - The opponent's bitboard.
///
/// # Returns
///
/// A `u64` value representing the legal moves for the player.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vl")]
#[allow(dead_code)]
fn get_moves_avx512(player: u64, opponent: u64) -> u64 {
    use std::arch::x86_64::*;

    let sh = _mm256_set_epi64x(7, 9, 8, 1);
    let masks = _mm256_set_epi64x(
        0x007E7E7E7E7E7E00u64 as i64,
        0x007E7E7E7E7E7E00u64 as i64,
        0x00FFFFFFFFFFFF00u64 as i64,
        0x7E7E7E7E7E7E7E7Eu64 as i64,
    );

    let empty = !(player | opponent);

    let pp = _mm256_set1_epi64x(player as i64);
    let oo = _mm256_set1_epi64x(opponent as i64);

    let masked_oo = _mm256_and_si256(oo, masks);

    let mut fl = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(pp, sh));
    let mut fr = _mm256_and_si256(masked_oo, _mm256_srlv_epi64(pp, sh));

    fl = _mm256_ternarylogic_epi64(fl, masked_oo, _mm256_sllv_epi64(fl, sh), 0xF8);
    fr = _mm256_ternarylogic_epi64(fr, masked_oo, _mm256_srlv_epi64(fr, sh), 0xF8);

    let pre_l = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(masked_oo, sh));
    let pre_r = _mm256_srlv_epi64(pre_l, sh);

    let sh2 = _mm256_add_epi64(sh, sh);

    fl = _mm256_ternarylogic_epi64(fl, pre_l, _mm256_sllv_epi64(fl, sh2), 0xF8);
    fr = _mm256_ternarylogic_epi64(fr, pre_r, _mm256_srlv_epi64(fr, sh2), 0xF8);

    fl = _mm256_ternarylogic_epi64(fl, pre_l, _mm256_sllv_epi64(fl, sh2), 0xF8);
    fr = _mm256_ternarylogic_epi64(fr, pre_r, _mm256_srlv_epi64(fr, sh2), 0xF8);

    let mm = _mm256_or_si256(_mm256_sllv_epi64(fl, sh), _mm256_srlv_epi64(fr, sh));
    let m128 = _mm_or_si128(_mm256_castsi256_si128(mm), _mm256_extracti128_si256(mm, 1));
    let moves = _mm_cvtsi128_si64(_mm_or_si128(m128, _mm_srli_si128(m128, 8))) as u64;

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
/// A `u64` value representing the legal moves for the player.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
fn get_moves_avx2(player: u64, opponent: u64) -> u64 {
    use std::arch::x86_64::*;

    let sh = _mm256_set_epi64x(7, 9, 8, 1);
    let masks = _mm256_set_epi64x(
        0x007E7E7E7E7E7E00,
        0x007E7E7E7E7E7E00,
        0x00FFFFFFFFFFFF00,
        0x7E7E7E7E7E7E7E7E,
    );

    let empty = !(player | opponent);

    let pp = _mm256_set1_epi64x(player as i64);
    let oo = _mm256_set1_epi64x(opponent as i64);
    let masked_oo = _mm256_and_si256(oo, masks);

    let mut fl = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(pp, sh));
    let mut fr = _mm256_and_si256(masked_oo, _mm256_srlv_epi64(pp, sh));

    fl = _mm256_or_si256(fl, _mm256_and_si256(masked_oo, _mm256_sllv_epi64(fl, sh)));
    fr = _mm256_or_si256(fr, _mm256_and_si256(masked_oo, _mm256_srlv_epi64(fr, sh)));

    let pre_l = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(masked_oo, sh));
    let pre_r = _mm256_srlv_epi64(pre_l, sh);

    let shift2 = _mm256_add_epi64(sh, sh);

    fl = _mm256_or_si256(fl, _mm256_and_si256(pre_l, _mm256_sllv_epi64(fl, shift2)));
    fr = _mm256_or_si256(fr, _mm256_and_si256(pre_r, _mm256_srlv_epi64(fr, shift2)));

    fl = _mm256_or_si256(fl, _mm256_and_si256(pre_l, _mm256_sllv_epi64(fl, shift2)));
    fr = _mm256_or_si256(fr, _mm256_and_si256(pre_r, _mm256_srlv_epi64(fr, shift2)));

    let mm = _mm256_or_si256(_mm256_sllv_epi64(fl, sh), _mm256_srlv_epi64(fr, sh));
    let m128 = _mm_or_si128(_mm256_castsi256_si128(mm), _mm256_extracti128_si256(mm, 1));
    let moves = _mm_cvtsi128_si64(_mm_or_si128(m128, _mm_srli_si128(m128, 8))) as u64;

    moves & empty
}

/// Expands a ray in the left-shift direction using SIMD.
///
/// Uses the "parallel prefix" (doubling) algorithm to propagate bits along a ray in O(log n)
/// iterations instead of O(n). For a 6-square maximum ray length in Reversi:
///
/// 1. Shift by 1 and OR with masked bits → covers distances 1-2
/// 2. Shift by 2 and OR with masked bits → covers distances 1-4
/// 3. Shift by 4 and OR with masked bits → covers distances 1-7 (sufficient for 6-max)
///
/// The mask is also doubled at each step to track valid continuation squares, preventing
/// ray expansion from wrapping around board edges.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn expand_ray_double_shl(mut mask: v128, mut x: v128, shift: u32) -> v128 {
    let mut tmp = u64x2_shl(x, shift);
    x = v128_or(x, v128_and(mask, tmp));
    mask = v128_and(mask, u64x2_shl(mask, shift));

    tmp = u64x2_shl(x, shift * 2);
    x = v128_or(x, v128_and(mask, tmp));
    mask = v128_and(mask, u64x2_shl(mask, shift * 2));

    v128_or(x, v128_and(mask, u64x2_shl(x, shift * 4)))
}

/// Expands a ray in the right-shift direction using SIMD.
///
/// Mirror of [`expand_ray_double_shl`] for the opposite direction. See that function
/// for algorithm details.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn expand_ray_double_shr(mut mask: v128, mut x: v128, shift: u32) -> v128 {
    let mut tmp = u64x2_shr(x, shift);
    x = v128_or(x, v128_and(mask, tmp));
    mask = v128_and(mask, u64x2_shr(mask, shift));

    tmp = u64x2_shr(x, shift * 2);
    x = v128_or(x, v128_and(mask, tmp));
    mask = v128_and(mask, u64x2_shr(mask, shift * 2));

    v128_or(x, v128_and(mask, u64x2_shr(x, shift * 4)))
}

/// WASM SIMD128-optimized implementation of `get_moves`.
///
/// # Arguments
///
/// * `player` - The player's bitboard.
/// * `opponent` - The opponent's bitboard.
///
/// # Returns
///
/// A `u64` value representing the legal moves for the player.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
fn get_moves_wasm(player: u64, opponent: u64) -> u64 {
    let empty = !(player | opponent);

    let pp = u64x2_splat(player);
    let oo = u64x2_splat(opponent);

    let masked_oo_hv = v128_and(oo, u64x2(0x7E7E7E7E7E7E7E7E, 0x00FFFFFFFFFFFF00));

    let adj_h_l = v128_and(masked_oo_hv, u64x2_shl(pp, 1));
    let adj_h_r = v128_and(masked_oo_hv, u64x2_shr(pp, 1));
    let adj_v_l = v128_and(masked_oo_hv, u64x2_shl(pp, 8));
    let adj_v_r = v128_and(masked_oo_hv, u64x2_shr(pp, 8));

    let flip_h_l = expand_ray_double_shl(masked_oo_hv, adj_h_l, 1);
    let flip_h_r = expand_ray_double_shr(masked_oo_hv, adj_h_r, 1);
    let flip_v_l = expand_ray_double_shl(masked_oo_hv, adj_v_l, 8);
    let flip_v_r = expand_ray_double_shr(masked_oo_hv, adj_v_r, 8);

    let moves_h = v128_or(u64x2_shl(flip_h_l, 1), u64x2_shr(flip_h_r, 1));
    let moves_v = v128_or(u64x2_shl(flip_v_l, 8), u64x2_shr(flip_v_r, 8));

    let masked_oo_d = v128_and(oo, u64x2_splat(0x007E7E7E7E7E7E00));

    let adj_d7_l = v128_and(masked_oo_d, u64x2_shl(pp, 7));
    let adj_d7_r = v128_and(masked_oo_d, u64x2_shr(pp, 7));
    let adj_d9_l = v128_and(masked_oo_d, u64x2_shl(pp, 9));
    let adj_d9_r = v128_and(masked_oo_d, u64x2_shr(pp, 9));

    let flip_d7_l = expand_ray_double_shl(masked_oo_d, adj_d7_l, 7);
    let flip_d7_r = expand_ray_double_shr(masked_oo_d, adj_d7_r, 7);
    let flip_d9_l = expand_ray_double_shl(masked_oo_d, adj_d9_l, 9);
    let flip_d9_r = expand_ray_double_shr(masked_oo_d, adj_d9_r, 9);

    let moves_d7 = v128_or(u64x2_shl(flip_d7_l, 7), u64x2_shr(flip_d7_r, 7));
    let moves_d9 = v128_or(u64x2_shl(flip_d9_l, 9), u64x2_shr(flip_d9_r, 9));

    let h_moves = u64x2_extract_lane::<0>(moves_h);
    let v_moves = u64x2_extract_lane::<1>(moves_v);
    let d7_moves = u64x2_extract_lane::<0>(moves_d7);
    let d9_moves = u64x2_extract_lane::<1>(moves_d9);

    (h_moves | v_moves | d7_moves | d9_moves) & empty
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
#[allow(dead_code)]
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
fn get_potential_moves(p: u64, o: u64) -> u64 {
    let h = get_some_potential_moves(o & 0x7E7E_7E7E_7E7E_7E7E_u64, 1);
    let v = get_some_potential_moves(o & 0x00FF_FFFF_FFFF_FF00_u64, 8);
    let d1 = get_some_potential_moves(o & 0x007E_7E7E_7E7E_7E00_u64, 7);
    let d2 = get_some_potential_moves(o & 0x007E_7E7E_7E7E_7E00_u64, 9);

    (h | v | d1 | d2) & !(p | o)
}

/// Gets both the legal moves and potential moves for the current player.
///
/// # Arguments
///
/// * `p` - The player's bitboard.
/// * `o` - The opponent's bitboard.
///
/// # Returns
/// A tuple containing two `u64` values:
/// - The first value represents the legal moves.
/// - The second value represents the potential moves.
#[inline(always)]
fn get_moves_and_potential(player: u64, opponent: u64) -> (u64, u64) {
    cfg_if! {
        if #[cfg(all(target_arch = "x86_64", target_feature = "avx512vl"))] {
            unsafe { get_moves_and_potential_avx512(player, opponent) }
        } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
            unsafe { get_moves_and_potential_avx2(player, opponent) }
        } else {
            (get_moves(player, opponent), get_potential_moves(player, opponent))
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vl")]
#[allow(dead_code)]
fn get_moves_and_potential_avx512(player: u64, opponent: u64) -> (u64, u64) {
    use std::arch::x86_64::*;

    let sh = _mm256_set_epi64x(7, 9, 8, 1);
    let masks = _mm256_set_epi64x(
        0x007E7E7E7E7E7E00u64 as i64,
        0x007E7E7E7E7E7E00u64 as i64,
        0x00FFFFFFFFFFFF00u64 as i64,
        0x7E7E7E7E7E7E7E7Eu64 as i64,
    );

    let empty = !(player | opponent);

    let pp = _mm256_set1_epi64x(player as i64);
    let oo = _mm256_set1_epi64x(opponent as i64);

    let masked_oo = _mm256_and_si256(oo, masks);

    // Potential moves calculation
    let pot_l = _mm256_sllv_epi64(masked_oo, sh);
    let pot_r = _mm256_srlv_epi64(masked_oo, sh);
    let pot_mm = _mm256_or_si256(pot_l, pot_r);

    let pot_m128 = _mm_or_si128(
        _mm256_castsi256_si128(pot_mm),
        _mm256_extracti128_si256(pot_mm, 1),
    );
    let potential = _mm_cvtsi128_si64(_mm_or_si128(pot_m128, _mm_srli_si128(pot_m128, 8))) as u64;

    // Moves calculation
    let mut fl = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(pp, sh));
    let mut fr = _mm256_and_si256(masked_oo, _mm256_srlv_epi64(pp, sh));

    fl = _mm256_ternarylogic_epi64(fl, masked_oo, _mm256_sllv_epi64(fl, sh), 0xF8);
    fr = _mm256_ternarylogic_epi64(fr, masked_oo, _mm256_srlv_epi64(fr, sh), 0xF8);

    let pre_l = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(masked_oo, sh));
    let pre_r = _mm256_srlv_epi64(pre_l, sh);

    let sh2 = _mm256_add_epi64(sh, sh);

    fl = _mm256_ternarylogic_epi64(fl, pre_l, _mm256_sllv_epi64(fl, sh2), 0xF8);
    fr = _mm256_ternarylogic_epi64(fr, pre_r, _mm256_srlv_epi64(fr, sh2), 0xF8);

    fl = _mm256_ternarylogic_epi64(fl, pre_l, _mm256_sllv_epi64(fl, sh2), 0xF8);
    fr = _mm256_ternarylogic_epi64(fr, pre_r, _mm256_srlv_epi64(fr, sh2), 0xF8);

    let mm = _mm256_or_si256(_mm256_sllv_epi64(fl, sh), _mm256_srlv_epi64(fr, sh));
    let m128 = _mm_or_si128(_mm256_castsi256_si128(mm), _mm256_extracti128_si256(mm, 1));
    let moves = _mm_cvtsi128_si64(_mm_or_si128(m128, _mm_srli_si128(m128, 8))) as u64;

    (moves & empty, potential & empty)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
fn get_moves_and_potential_avx2(player: u64, opponent: u64) -> (u64, u64) {
    use std::arch::x86_64::*;

    let sh = _mm256_set_epi64x(7, 9, 8, 1);
    let masks = _mm256_set_epi64x(
        0x007E7E7E7E7E7E00u64 as i64,
        0x007E7E7E7E7E7E00u64 as i64,
        0x00FFFFFFFFFFFF00u64 as i64,
        0x7E7E7E7E7E7E7E7Eu64 as i64,
    );

    let empty = !(player | opponent);

    let pp = _mm256_set1_epi64x(player as i64);
    let oo = _mm256_set1_epi64x(opponent as i64);
    let masked_oo = _mm256_and_si256(oo, masks);

    // Potential moves calculation
    let pot_l = _mm256_sllv_epi64(masked_oo, sh);
    let pot_r = _mm256_srlv_epi64(masked_oo, sh);
    let pot_mm = _mm256_or_si256(pot_l, pot_r);

    let pot_m128 = _mm_or_si128(
        _mm256_castsi256_si128(pot_mm),
        _mm256_extracti128_si256(pot_mm, 1),
    );
    let potential = _mm_cvtsi128_si64(_mm_or_si128(pot_m128, _mm_srli_si128(pot_m128, 8))) as u64;

    // Moves calculation
    let mut fl = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(pp, sh));
    let mut fr = _mm256_and_si256(masked_oo, _mm256_srlv_epi64(pp, sh));

    fl = _mm256_or_si256(fl, _mm256_and_si256(masked_oo, _mm256_sllv_epi64(fl, sh)));
    fr = _mm256_or_si256(fr, _mm256_and_si256(masked_oo, _mm256_srlv_epi64(fr, sh)));

    let pre_l = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(masked_oo, sh));
    let pre_r = _mm256_srlv_epi64(pre_l, sh);

    let shift2 = _mm256_add_epi64(sh, sh);

    fl = _mm256_or_si256(fl, _mm256_and_si256(pre_l, _mm256_sllv_epi64(fl, shift2)));
    fr = _mm256_or_si256(fr, _mm256_and_si256(pre_r, _mm256_srlv_epi64(fr, shift2)));

    fl = _mm256_or_si256(fl, _mm256_and_si256(pre_l, _mm256_sllv_epi64(fl, shift2)));
    fr = _mm256_or_si256(fr, _mm256_and_si256(pre_r, _mm256_srlv_epi64(fr, shift2)));

    let mm = _mm256_or_si256(_mm256_sllv_epi64(fl, sh), _mm256_srlv_epi64(fr, sh));
    let m128 = _mm_or_si128(_mm256_castsi256_si128(mm), _mm256_extracti128_si256(mm, 1));
    let moves = _mm_cvtsi128_si64(_mm_or_si128(m128, _mm_srli_si128(m128, 8))) as u64;

    (moves & empty, potential & empty)
}

/// Delta swap - a fundamental bit manipulation operation.
///
/// # Arguments
///
/// * `bits` - The value to perform the swap on.
/// * `mask` - Specifies which bit pairs to swap (must have 1s in positions that are `delta` apart).
/// * `delta` - The distance between bit pairs to swap.
///
/// # Returns
///
/// A `u64` value with the specified bit pairs swapped.
#[inline(always)]
fn delta_swap(bits: u64, mask: u64, delta: u32) -> u64 {
    let tmp = mask & (bits ^ (bits << delta));
    bits ^ tmp ^ (tmp >> delta)
}

/// An iterator that yields each set bit position in a bitboard as a `Square`.
pub struct BitboardIterator {
    bitboard: Bitboard,
}

impl BitboardIterator {
    /// Creates a new `BitboardIterator`.
    ///
    /// # Arguments
    ///
    /// * `bitboard` - The bitboard to iterate over.
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
mod tests {
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
    fn test_set_and_contains() {
        let mut board = Bitboard::new(0);

        // Test setting bits
        board = board.set(Square::A1);
        assert!(board.contains(Square::A1));
        assert!(!board.contains(Square::A2));

        board = board.set(Square::H8);
        assert!(board.contains(Square::A1));
        assert!(board.contains(Square::H8));
        assert!(!board.contains(Square::D4));

        // Test setting already set bit
        board = board.set(Square::A1);
        assert!(board.contains(Square::A1));
    }

    #[test]
    fn test_get_moves_initial_position() {
        // Standard Reversi initial position
        let player = Square::D5.bitboard() | Square::E4.bitboard();
        let opponent = Square::D4.bitboard() | Square::E5.bitboard();
        let moves = player.get_moves(opponent);

        // Valid moves for black (first player) in initial position
        assert!(moves.contains(Square::C4));
        assert!(moves.contains(Square::F5));
        assert!(moves.contains(Square::D3));
        assert!(moves.contains(Square::E6));
        assert_eq!(moves.count(), 4);
    }

    #[test]
    fn test_get_moves_no_moves() {
        // Position where player has no moves
        let player = Bitboard(0);
        let opponent = Bitboard(u64::MAX);
        let moves = player.get_moves(opponent);

        assert_eq!(moves, Bitboard(0));
    }

    #[test]
    fn test_get_moves_capture_all_directions() {
        // Position where a move captures in all 8 directions
        // Center disc surrounded by opponent discs
        let player = Square::A1.bitboard()
            | Square::H1.bitboard()
            | Square::A8.bitboard()
            | Square::H8.bitboard()
            | Square::A4.bitboard()
            | Square::H4.bitboard()
            | Square::D1.bitboard()
            | Square::D8.bitboard();

        let opponent = Square::B2.bitboard()
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

        let moves = player.get_moves(opponent);

        // D4 should be a valid move that captures in all directions
        assert!(moves.contains(Square::D4));
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

        let test_positions: [(u64, u64); 3] = [
            (
                Square::D5.bitboard().0 | Square::E4.bitboard().0,
                Square::D4.bitboard().0 | Square::E5.bitboard().0,
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
    fn test_get_potential_moves_initial_position() {
        // Standard Reversi initial position
        let player = Square::D5.bitboard() | Square::E4.bitboard();
        let opponent = Square::D4.bitboard() | Square::E5.bitboard();
        let potential = player.get_potential_moves(opponent);

        // Potential moves are empty squares adjacent to opponent discs
        // Around D4: C3, C4, C5, D3, D5(occupied), E3, E4(occupied), E5(occupied)
        // Around E5: D4(occupied), D5(occupied), D6, E4(occupied), E6, F4, F5, F6
        // Union minus occupied: C3, C4, C5, D3, E3, D6, E6, F4, F5, F6

        assert!(potential.contains(Square::C3));
        assert!(potential.contains(Square::C4));
        assert!(potential.contains(Square::C5));
        assert!(potential.contains(Square::D3));
        assert!(potential.contains(Square::E3));
        assert!(potential.contains(Square::D6));
        assert!(potential.contains(Square::E6));
        assert!(potential.contains(Square::F4));
        assert!(potential.contains(Square::F5));
        assert!(potential.contains(Square::F6));

        // Should not have bits on occupied squares
        assert!(!potential.contains(Square::D4));
        assert!(!potential.contains(Square::D5));
        assert!(!potential.contains(Square::E4));
        assert!(!potential.contains(Square::E5));
    }

    #[test]
    fn test_get_moves_and_potential_initial_position() {
        // Standard Reversi initial position
        let player = Square::D5.bitboard() | Square::E4.bitboard();
        let opponent = Square::D4.bitboard() | Square::E5.bitboard();

        let (moves, potential) = player.get_moves_and_potential(opponent);
        let expected_moves = player.get_moves(opponent);
        let expected_potential = player.get_potential_moves(opponent);

        assert_eq!(moves, expected_moves);
        assert_eq!(potential, expected_potential);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_get_moves_and_potential_consistency() {
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_avx512 =
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl");

        if !(has_avx2 || has_avx512) {
            return;
        }

        let test_positions: [(u64, u64); 4] = [
            (
                Square::D5.bitboard().0 | Square::E4.bitboard().0,
                Square::D4.bitboard().0 | Square::E5.bitboard().0,
            ),
            (0x00003C3C3C000000, 0x0000C3C3C3000000),
            (0xFF00000000000000, 0x00FF000000000000),
            (0x0000001824428100, 0x0000002442810000),
        ];

        for (player, opponent) in test_positions {
            let moves_fallback = get_moves_fallback(player, opponent);
            let potential_scalar = get_potential_moves(player, opponent);

            if has_avx2 {
                let (moves_avx2, pot_avx2) =
                    unsafe { get_moves_and_potential_avx2(player, opponent) };
                assert_eq!(
                    moves_fallback, moves_avx2,
                    "Moves mismatch AVX2 for player={player:016x}, opponent={opponent:016x}"
                );
                assert_eq!(
                    potential_scalar, pot_avx2,
                    "Potential mismatch AVX2 for player={player:016x}, opponent={opponent:016x}"
                );
            }

            if has_avx512 {
                let (moves_avx512, pot_avx512) =
                    unsafe { get_moves_and_potential_avx512(player, opponent) };
                assert_eq!(
                    moves_fallback, moves_avx512,
                    "Moves mismatch AVX512 for player={player:016x}, opponent={opponent:016x}"
                );
                assert_eq!(
                    potential_scalar, pot_avx512,
                    "Potential mismatch AVX512 for player={player:016x}, opponent={opponent:016x}"
                );
            }
        }
    }

    #[test]
    fn test_corner_stability() {
        // No corners
        let board = Square::D4.bitboard().0 | Square::E5.bitboard().0;
        assert_eq!(Bitboard(board).corner_stability(), 0);

        // One corner (A1)
        let board = Square::A1.bitboard().0;
        assert_eq!(Bitboard(board).corner_stability(), 1);

        // All corners - the function checks for stable corners which includes
        // corners that are protected by adjacent corners
        let board: u64 = CORNER_MASK;
        assert_eq!(Bitboard(board).corner_stability(), 4);

        // Corner with adjacent discs - A1 with A2 and B1
        // The function counts corners that form stable groups
        let board = Square::A1.bitboard().0
            | Square::A2.bitboard().0
            | Square::B1.bitboard().0
            | Square::B2.bitboard().0;
        assert_eq!(Bitboard(board).corner_stability(), 3); // A1, A2, B1 form a stable group
    }

    #[test]
    fn test_bitboard_iterator() {
        // Example bitboard: bits 0, 1, and 63 are set
        let bitboard = Square::A1.bitboard().0 | Square::B1.bitboard().0 | Square::H8.bitboard().0;
        let mut iterator = BitboardIterator::new(Bitboard(bitboard));

        assert_eq!(iterator.next(), Some(Square::A1));
        assert_eq!(iterator.next(), Some(Square::B1));
        assert_eq!(iterator.next(), Some(Square::H8));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_bitboard_iterator_empty() {
        let bitboard: u64 = 0;
        let mut iterator = BitboardIterator::new(Bitboard(bitboard));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_bitboard_iterator_full() {
        let bitboard: u64 = u64::MAX;
        let count = bitboard.count_ones();
        let mut iterator = BitboardIterator::new(Bitboard(bitboard));
        for i in 0..count {
            assert_eq!(iterator.next(), Some(Square::from_u32_unchecked(i)));
        }
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_bitboard_iterator_single_bit() {
        let bitboard = Square::E4.bitboard().0;
        let mut iterator = BitboardIterator::new(Bitboard(bitboard));
        assert_eq!(iterator.next(), Some(Square::E4));
        assert_eq!(iterator.next(), None);
    }

    #[test]
    fn test_bitboard_iterator_diagonal() {
        // Main diagonal A1-H8
        let bitboard: u64 = 0x8040201008040201;
        let squares: Vec<Square> = BitboardIterator::new(Bitboard(bitboard)).collect();
        assert_eq!(squares.len(), 8);
        assert_eq!(squares[0], Square::A1);
        assert_eq!(squares[7], Square::H8);
    }

    #[test]
    fn test_has_adjacent_bit() {
        let bitboard = Square::B2.bitboard();
        assert!(bitboard.has_adjacent_bit(Square::A1));
        assert!(bitboard.has_adjacent_bit(Square::A2));
        assert!(bitboard.has_adjacent_bit(Square::A3));
        assert!(!bitboard.has_adjacent_bit(Square::A4));
        assert!(bitboard.has_adjacent_bit(Square::B1));
        assert!(!bitboard.has_adjacent_bit(Square::B2));
        assert!(bitboard.has_adjacent_bit(Square::B3));
        assert!(!bitboard.has_adjacent_bit(Square::B4));
        assert!(bitboard.has_adjacent_bit(Square::C1));
        assert!(bitboard.has_adjacent_bit(Square::C2));
        assert!(bitboard.has_adjacent_bit(Square::C3));
        assert!(!bitboard.has_adjacent_bit(Square::C4));
        assert!(!bitboard.has_adjacent_bit(Square::D1));
    }

    #[test]
    fn test_has_adjacent_bit_corners() {
        // Test corner adjacency
        let bitboard = Square::B1.bitboard() | Square::A2.bitboard();
        assert!(bitboard.has_adjacent_bit(Square::A1));

        let bitboard = Square::G8.bitboard() | Square::H7.bitboard();
        assert!(bitboard.has_adjacent_bit(Square::H8));
    }

    #[test]
    fn test_has_adjacent_bit_edge() {
        // Test edge square adjacency
        let bitboard = Square::C1.bitboard() | Square::D2.bitboard() | Square::E1.bitboard();
        assert!(bitboard.has_adjacent_bit(Square::D1));
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

    #[test]
    fn test_delta_swap() {
        let bits = 0b10100000;
        let mask = 0b01000000;
        let delta = 1;
        let result = delta_swap(bits, mask, delta);
        assert_eq!(result, 0b11000000);

        let bits2 = 0b11110000;
        let mask2 = 0b00001111;
        let delta2 = 4;
        let result2 = delta_swap(bits2, mask2, delta2);
        assert_eq!(result2, 0b11110000);

        let bits3 = 0b11111111;
        let mask3 = 0b00001111;
        let delta3 = 4;
        let result3 = delta_swap(bits3, mask3, delta3);
        assert_eq!(result3, 0b11110000);
    }

    #[test]
    fn test_bitboard_transformations_consistency() {
        let test_board = Bitboard::new(0x123456789ABCDEF0);

        // Identity tests
        assert_eq!(test_board.flip_vertical().flip_vertical(), test_board);
        assert_eq!(test_board.flip_horizontal().flip_horizontal(), test_board);
        assert_eq!(test_board.flip_diag_a1h8().flip_diag_a1h8(), test_board);
        assert_eq!(test_board.flip_diag_a8h1().flip_diag_a8h1(), test_board);

        let mut rotated = test_board;
        for _ in 0..4 {
            rotated = rotated.rotate_90_clockwise();
        }
        assert_eq!(rotated, test_board);

        // 180° rotation equivalence
        let rotate_180_v1 = test_board.flip_horizontal().flip_vertical();
        let rotate_180_v2 = test_board.rotate_90_clockwise().rotate_90_clockwise();
        assert_eq!(rotate_180_v1, rotate_180_v2);
    }

    // Tests for Bitboard struct

    #[test]
    fn test_bitboard_struct_set_remove() {
        let bb = Bitboard::new(0);

        // set()
        let bb = bb.set(Square::A1);
        assert!(bb.contains(Square::A1));
        assert!(!bb.contains(Square::H8));

        let bb = bb.set(Square::H8);
        assert!(bb.contains(Square::A1));
        assert!(bb.contains(Square::H8));

        let bb = bb.remove(Square::A1);
        assert!(!bb.contains(Square::A1));
        assert!(bb.contains(Square::H8));
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
    fn test_bitboard_struct_has_adjacent_bit() {
        let bb = Square::B2.bitboard();
        assert!(bb.has_adjacent_bit(Square::A1));
        assert!(bb.has_adjacent_bit(Square::C3));
        assert!(!bb.has_adjacent_bit(Square::D4));
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
            let sq = Square::from_u32_unchecked(i);
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
    #[cfg(target_arch = "x86_64")]
    fn test_get_moves_consistency_extended() {
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_avx512 =
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl");

        if !(has_avx2 || has_avx512) {
            return;
        }

        // Extended test positions covering edge cases
        let test_positions: [(u64, u64); 10] = [
            // Initial position
            (
                Square::D5.bitboard().0 | Square::E4.bitboard().0,
                Square::D4.bitboard().0 | Square::E5.bitboard().0,
            ),
            // Edge-heavy positions
            (0xFF000000000000FF, 0x00FFFFFFFFFFFF00), // ranks 1 and 8
            (0x8181818181818181, 0x7E7E7E7E7E7E7E7E), // files A and H
            // Diagonal positions
            (0x8040201008040201, 0x0102040810204080), // both diagonals
            // Corner-heavy
            (CORNER_MASK, 0x4281000000008142), // corners vs X-squares
            // Sparse positions
            (0x0000001000000000, 0x0000002800000000), // single disc each side
            // Dense center
            (0x00003C3C00000000, 0x0000C3C300000000), // 4x4 blocks
            // One side dominating
            (0xFFFFFFFF00000000, 0x00000000FFFFFFFF), // split board
            // Near-endgame (few empties)
            (0xAAAAAAAAAAAAAAAA, 0x5555555555555554), // checkerboard with 1 empty
            // Asymmetric
            (0x0F0F0F0F00000000, 0x00000000F0F0F0F0),
        ];

        for (player, opponent) in test_positions {
            // Skip invalid positions (overlapping bits)
            if player & opponent != 0 {
                continue;
            }

            let moves_fallback = get_moves_fallback(player, opponent);

            if has_avx2 {
                let moves_avx2 = unsafe { get_moves_avx2(player, opponent) };
                assert_eq!(
                    moves_fallback, moves_avx2,
                    "Extended: Fallback vs AVX2 differ for player={player:016x}, opponent={opponent:016x}"
                );
            }

            if has_avx512 {
                let moves_avx512 = unsafe { get_moves_avx512(player, opponent) };
                assert_eq!(
                    moves_fallback, moves_avx512,
                    "Extended: Fallback vs AVX-512 differ for player={player:016x}, opponent={opponent:016x}"
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_get_moves_and_potential_consistency_extended() {
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_avx512 =
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl");

        if !(has_avx2 || has_avx512) {
            return;
        }

        // Extended test positions
        let test_positions: [(u64, u64); 8] = [
            // Initial position
            (
                Square::D5.bitboard().0 | Square::E4.bitboard().0,
                Square::D4.bitboard().0 | Square::E5.bitboard().0,
            ),
            // Edge cases
            (0xFF000000000000FF, 0x00FFFFFFFFFFFF00),
            (0x8181818181818181, 0x7E7E7E7E7E7E7E7E),
            (CORNER_MASK, 0x4281000000008142),
            // Various game states
            (0x00003C3C00000000, 0x0000C3C300000000),
            (0x0000001000000000, 0x0000002800000000),
            (0x0F0F0F0F00000000, 0x00000000F0F0F0F0),
            (0x8040201008040201, 0x0102040810204080),
        ];

        for (player, opponent) in test_positions {
            if player & opponent != 0 {
                continue;
            }

            let moves_scalar = get_moves_fallback(player, opponent);
            let potential_scalar = get_potential_moves(player, opponent);

            if has_avx2 {
                let (moves_avx2, pot_avx2) =
                    unsafe { get_moves_and_potential_avx2(player, opponent) };
                assert_eq!(
                    moves_scalar, moves_avx2,
                    "Extended: Moves mismatch AVX2 for player={player:016x}, opponent={opponent:016x}"
                );
                assert_eq!(
                    potential_scalar, pot_avx2,
                    "Extended: Potential mismatch AVX2 for player={player:016x}, opponent={opponent:016x}"
                );
            }

            if has_avx512 {
                let (moves_avx512, pot_avx512) =
                    unsafe { get_moves_and_potential_avx512(player, opponent) };
                assert_eq!(
                    moves_scalar, moves_avx512,
                    "Extended: Moves mismatch AVX512 for player={player:016x}, opponent={opponent:016x}"
                );
                assert_eq!(
                    potential_scalar, pot_avx512,
                    "Extended: Potential mismatch AVX512 for player={player:016x}, opponent={opponent:016x}"
                );
            }
        }
    }
}

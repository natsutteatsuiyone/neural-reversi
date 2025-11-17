/// Clear least significant bit (BLSR).
///
/// Clears the least significant set bit (rightmost 1) in `a`.
/// This operation is also known as "reset lowest set bit" and uses the bit manipulation
/// trick `a & (a - 1)`.
///
/// # Arguments
///
/// * `a` - The value to clear the least significant bit from.
///
/// # Returns
///
/// A `u64` value with the least significant set bit cleared. If `a` is 0, returns 0.
#[inline]
pub fn clear_lsb_u64(a: u64) -> u64 {
    a & a.wrapping_sub(1)
}

/// Flips the bitboard vertically.
///
/// Reverses the order of ranks (rows) in the bitboard. The top rank becomes the bottom rank and vice versa.
/// This is efficiently implemented using byte swapping.
///
/// # Arguments
///
/// * `b` - The bitboard to flip.
///
/// # Returns
///
/// A `u64` value representing the vertically flipped bitboard.
#[inline]
pub fn flip_vertical(b: u64) -> u64 {
    b.swap_bytes()
}

/// Flips the bitboard horizontally.
///
/// Reverses the order of files (columns) in the bitboard. The leftmost file becomes the rightmost file and vice versa.
/// This is implemented using a series of bit manipulation operations.
///
/// # Arguments
///
/// * `b` - The bitboard to flip.
///
/// # Returns
///
/// A `u64` value representing the horizontally flipped bitboard.
#[inline]
pub fn flip_horizontal(mut b: u64) -> u64 {
    let mask1: u64 = 0x5555555555555555;
    let mask2: u64 = 0x3333333333333333;
    let mask3: u64 = 0x0f0f0f0f0f0f0f0f;

    b = ((b >> 1) & mask1) | ((b & mask1) << 1);
    b = ((b >> 2) & mask2) | ((b & mask2) << 2);
    b = ((b >> 4) & mask3) | ((b & mask3) << 4);

    b
}

/// Rotates the bitboard 90 degrees clockwise.
///
/// # Arguments
/// * `b` - The bitboard to rotate.
///
/// # Returns
/// A `u64` value representing the rotated bitboard.
#[inline]
pub fn rotate_90_clockwise(b: u64) -> u64 {
    flip_vertical(flip_diag_a8h1(b))
}

/// Rotates the bitboard 180 degrees clockwise.
///
/// # Arguments
/// * `b` - The bitboard to rotate.
///
/// # Returns
/// A `u64` value representing the rotated bitboard.
#[inline]
pub fn rotate_180_clockwise(b: u64) -> u64 {
    b.reverse_bits()
}

/// Rotates the bitboard 270 degrees clockwise (or 90 degrees counter-clockwise).
///
/// # Arguments
/// * `b` - The bitboard to rotate.
///
/// # Returns
/// A `u64` value representing the rotated bitboard.
#[inline]
pub fn rotate_270_clockwise(b: u64) -> u64 {
    flip_vertical(flip_diag_a1h8(b))
}

/// Flips the bitboard along the A1-H8 diagonal.
///
/// Reflects the bitboard across the main diagonal (from square A1 to H8).
/// After this transformation, the square at (rank, file) moves to (file, rank).
///
/// # Arguments
///
/// * `bits` - The bitboard to flip.
///
/// # Returns
///
/// A `u64` value representing the diagonally flipped bitboard.
#[inline]
pub fn flip_diag_a1h8(mut bits: u64) -> u64 {
    let mask1: u64 = 0x5500550055005500;
    let mask2: u64 = 0x3333000033330000;
    let mask3: u64 = 0x0f0f0f0f00000000;

    bits = delta_swap(bits, mask3, 28);
    bits = delta_swap(bits, mask2, 14);
    bits = delta_swap(bits, mask1, 7);
    bits
}

/// Flips the bitboard along the A8-H1 diagonal.
///
/// Reflects the bitboard across the anti-diagonal (from square A8 to H1).
/// This is equivalent to a 90-degree counter-clockwise rotation followed by a vertical flip.
///
/// # Arguments
///
/// * `bits` - The bitboard to flip.
///
/// # Returns
///
/// A `u64` value representing the diagonally flipped bitboard.
#[inline]
pub fn flip_diag_a8h1(mut bits: u64) -> u64 {
    let mask1: u64 = 0xaa00aa00aa00aa00;
    let mask2: u64 = 0xcccc0000cccc0000;
    let mask3: u64 = 0xf0f0f0f000000000;

    bits = delta_swap(bits, mask3, 36);
    bits = delta_swap(bits, mask2, 18);
    bits = delta_swap(bits, mask1, 9);
    bits
}

/// Delta swap - a fundamental bit manipulation operation.
///
/// Swaps bits that are `delta` positions apart, but only for positions specified by the mask.
/// This is a key building block for many bit permutation operations.
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
///
/// # Algorithm
///
/// 1. XOR bits with bits shifted by delta, masked to find differences
/// 2. XOR the original with the differences to perform the swap
#[inline]
fn delta_swap(bits: u64, mask: u64, delta: u32) -> u64 {
    let tmp = mask & (bits ^ (bits << delta));
    bits ^ tmp ^ (tmp >> delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clear_lsb_u64() {
        // Single bit
        assert_eq!(clear_lsb_u64(0b1000), 0b0000);

        // Multiple bits
        assert_eq!(clear_lsb_u64(0b11010100), 0b11010000);

        // All 1s
        assert_eq!(clear_lsb_u64(0b1111), 0b1110);

        // Zero
        assert_eq!(clear_lsb_u64(0), 0);

        // 64-bit
        assert_eq!(clear_lsb_u64(0xABCDEF0123456780), 0xABCDEF0123456700);
    }

    #[test]
    fn test_flip_vertical() {
        // Simple pattern
        let board = 0x0102030405060708u64;
        let flipped = flip_vertical(board);
        assert_eq!(flipped, 0x0807060504030201u64);

        // Symmetric pattern should be unchanged
        let symmetric = 0x1818181818181818u64;
        assert_eq!(flip_vertical(flip_vertical(symmetric)), symmetric);

        // Empty
        assert_eq!(flip_vertical(0), 0);

        // Full
        assert_eq!(flip_vertical(0xFFFFFFFFFFFFFFFF), 0xFFFFFFFFFFFFFFFF);
    }

    #[test]
    fn test_flip_horizontal() {
        // Edge columns
        assert_eq!(flip_horizontal(0x0101010101010101), 0x8080808080808080);
        assert_eq!(flip_horizontal(0x8080808080808080), 0x0101010101010101);

        // Nibble pattern
        assert_eq!(flip_horizontal(0x0F0F0F0F0F0F0F0F), 0xF0F0F0F0F0F0F0F0);

        // Double flip identity
        let original = 0x123456789ABCDEF0u64;
        assert_eq!(flip_horizontal(flip_horizontal(original)), original);

        // Single rank
        assert_eq!(flip_horizontal(0xFF), 0xFF);
    }

    #[test]
    fn test_rotate_90_clockwise() {
        // Corners
        assert_eq!(rotate_90_clockwise(0x0000000000000001), 0x0000000000000080);
        assert_eq!(rotate_90_clockwise(0x0000000000000080), 0x8000000000000000);
        assert_eq!(rotate_90_clockwise(0x8000000000000000), 0x0100000000000000);
        assert_eq!(rotate_90_clockwise(0x0100000000000000), 0x0000000000000001);

        // 4x rotation identity
        let original = 0x123456789ABCDEF0u64;
        let rotated = rotate_90_clockwise(rotate_90_clockwise(rotate_90_clockwise(
            rotate_90_clockwise(original),
        )));
        assert_eq!(rotated, original);
    }

    #[test]
    fn test_rotate_180_clockwise() {
        // Test corners
        assert_eq!(rotate_180_clockwise(0x0000000000000001), 0x8000000000000000);
        assert_eq!(rotate_180_clockwise(0x8000000000000000), 0x0000000000000001);
        assert_eq!(rotate_180_clockwise(0x0000000000000080), 0x0100000000000000);
        assert_eq!(rotate_180_clockwise(0x0100000000000000), 0x0000000000000080);

        // Test a full row
        assert_eq!(rotate_180_clockwise(0x00000000000000FF), 0xFF00000000000000);
        assert_eq!(rotate_180_clockwise(0xFF00000000000000), 0x00000000000000FF);

        // Test a pattern
        let original = 0x0F0F0F0F00000000;
        let rotated = 0x00000000F0F0F0F0;
        assert_eq!(rotate_180_clockwise(original), rotated);

        // Double rotation identity
        let test_board = 0x123456789ABCDEF0u64;
        assert_eq!(
            rotate_180_clockwise(rotate_180_clockwise(test_board)),
            test_board
        );

        // Empty and full boards
        assert_eq!(rotate_180_clockwise(0), 0);
        assert_eq!(rotate_180_clockwise(u64::MAX), u64::MAX);
    }

    #[test]
    fn test_rotate_270_clockwise() {
        // Test corners
        assert_eq!(rotate_270_clockwise(0x0000000000000001), 0x0100000000000000);
        assert_eq!(rotate_270_clockwise(0x0100000000000000), 0x8000000000000000);
        assert_eq!(rotate_270_clockwise(0x8000000000000000), 0x0000000000000080);
        assert_eq!(rotate_270_clockwise(0x0000000000000080), 0x0000000000000001);

        // 4x rotation identity
        let original = 0x123456789ABCDEF0u64;
        let rotated = rotate_270_clockwise(rotate_270_clockwise(rotate_270_clockwise(
            rotate_270_clockwise(original),
        )));
        assert_eq!(rotated, original);

        // Equivalence to 3x 90-degree rotation
        let rotated_90_3x = rotate_90_clockwise(rotate_90_clockwise(rotate_90_clockwise(original)));
        assert_eq!(rotate_270_clockwise(original), rotated_90_3x);
    }

    #[test]
    fn test_flip_diag_a1h8() {
        // Diagonal invariant
        assert_eq!(flip_diag_a1h8(0x8040201008040201), 0x8040201008040201);

        // Corners
        assert_eq!(flip_diag_a1h8(0x0000000000000001), 0x0000000000000001);
        assert_eq!(flip_diag_a1h8(0x8000000000000000), 0x8000000000000000);
        assert_eq!(flip_diag_a1h8(0x0000000000000080), 0x0100000000000000);
        assert_eq!(flip_diag_a1h8(0x0100000000000000), 0x0000000000000080);

        // Double flip identity
        let original = 0x123456789ABCDEF0u64;
        assert_eq!(flip_diag_a1h8(flip_diag_a1h8(original)), original);
    }

    #[test]
    fn test_flip_diag_a8h1() {
        // Anti-diagonal invariant
        assert_eq!(flip_diag_a8h1(0x0102040810204080), 0x0102040810204080);

        // Corners
        assert_eq!(flip_diag_a8h1(0x0100000000000000), 0x0100000000000000);
        assert_eq!(flip_diag_a8h1(0x0000000000000080), 0x0000000000000080);
        assert_eq!(flip_diag_a8h1(0x0000000000000001), 0x8000000000000000);
        assert_eq!(flip_diag_a8h1(0x8000000000000000), 0x0000000000000001);

        // Double flip identity
        let original = 0x123456789ABCDEF0u64;
        assert_eq!(flip_diag_a8h1(flip_diag_a8h1(original)), original);
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
        let test_board = 0x123456789ABCDEF0u64;

        // Identity tests
        assert_eq!(flip_vertical(flip_vertical(test_board)), test_board);
        assert_eq!(flip_horizontal(flip_horizontal(test_board)), test_board);
        assert_eq!(flip_diag_a1h8(flip_diag_a1h8(test_board)), test_board);
        assert_eq!(flip_diag_a8h1(flip_diag_a8h1(test_board)), test_board);

        let mut rotated = test_board;
        for _ in 0..4 {
            rotated = rotate_90_clockwise(rotated);
        }
        assert_eq!(rotated, test_board);

        // 180° rotation equivalence
        let rotate_180_v1 = flip_horizontal(flip_vertical(test_board));
        let rotate_180_v2 = rotate_90_clockwise(rotate_90_clockwise(test_board));
        assert_eq!(rotate_180_v1, rotate_180_v2);
    }
}

/// Parallel bits extract.
///
/// Extracts bits from `a` specified by the `mask` and compacts them right-justified in the result.
///
/// # Arguments
///
/// * `a` - The source from which bits are extracted.
/// * `mask` - Specifies which bits are to be extracted.
///
/// # Returns
///
/// A `u64` value containing the extracted bits packed right-justified.
#[inline]
pub fn pext_u64(a: u64, mask: u64) -> u64 {
    if is_x86_feature_detected!("bmi2") {
        unsafe {
            use std::arch::x86_64::_pext_u64;
            _pext_u64(a, mask)
        }
    } else {
        let mut result = 0;
        let mut bit_idx = 0;
        let mut curr_mask = mask;

        while curr_mask != 0 {
            let lsb = curr_mask & curr_mask.wrapping_neg(); // 最下位の1を取得
            if (a & lsb) != 0 {
                result |= 1u64 << bit_idx;
            }
            bit_idx += 1;
            curr_mask &= curr_mask - 1; // 最下位の1をクリア
        }
        result
    }
}

/// Parallel bits deposit.
///
/// Deposits the right-justified bits from `a` into the destination based on the `mask`.
///
/// # Arguments
///
/// * `a` - The source of bits to be deposited. The bits are assumed to be right-justified.
/// * `mask` - Specifies where the bits are to be deposited in the result.
///
/// # Returns
///
/// A `u64` value where the bits from `a` are deposited into positions specified by `mask`.
#[inline]
pub fn pdep_u64(a: u64, mask: u64) -> u64 {
    if is_x86_feature_detected!("bmi2") {
        unsafe {
            use std::arch::x86_64::_pdep_u64;
            _pdep_u64(a, mask)
        }
    } else {
        let mut result = 0;
        let mut bit_idx = 0;
        let mut curr_mask = mask;

        while curr_mask != 0 {
            let lsb = curr_mask & curr_mask.wrapping_neg(); // 最下位の1を取得
            if ((a >> bit_idx) & 1) != 0 {
                result |= lsb;
            }
            bit_idx += 1;
            curr_mask &= curr_mask - 1; // 最下位の1をクリア
        }
        result
    }
}

/// Bit extract.
///
/// Extracts a range of bits from `a` specified by `start` and `length`.
///
/// # Arguments
///
/// * `a` - The source from which bits are extracted.
/// * `start` - The starting bit position of the extraction (0-based).
/// * `length` - The number of bits to extract.
///
/// # Returns
///
/// A `u32` value containing the extracted bits.
#[inline]
pub fn bextr_u32(a: u32, start: u32, length: u32) -> u32 {
    if is_x86_feature_detected!("bmi1") {
        use std::arch::x86_64::_bextr_u32;
        unsafe { _bextr_u32(a, start, length) }
    } else {
        (a >> start) & ((1 << length) - 1)
    }
}

/// Clear least significant bit.
///
/// Clears the least significant set bit (rightmost 1) in `a`.
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
    a & (a - 1)
}

/// Flips the bitboard horizontally.
///
/// # Arguments
///
/// * `b` - The bitboard to flip.
///
/// # Returns
///
/// A `u64` value representing the flipped bitboard.
#[inline]
pub fn flip_vertical(b: u64) -> u64 {
    b.swap_bytes()
}

/// Flips the bitboard horizontally.
///
/// # Arguments
///
/// * `b` - The bitboard to flip.
///
/// # Returns
///
/// A `u64` value representing the flipped bitboard.
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
///
/// * `b` - The bitboard to rotate.
///
/// # Returns
///
/// A `u64` value representing the rotated bitboard.
#[inline]
pub fn rotate_90_clockwise(b: u64) -> u64 {
    const MASK1: u64 = 0xAA00AA00AA00AA00;
    const MASK2: u64 = 0xCCCC0000CCCC0000;
    const MASK3: u64 = 0xF0F0F0F000000000;

    let mut bits = b;
    bits = delta_swap(bits, MASK3, 36);
    bits = delta_swap(bits, MASK2, 18);
    bits = delta_swap(bits, MASK1, 9);
    flip_vertical(bits)
}

pub fn flip_diag_a1h8(mut bits: u64) -> u64 {
    let mask1: u64 = 0x5500550055005500;
    let mask2: u64 = 0x3333000033330000;
    let mask3: u64 = 0x0f0f0f0f00000000;

    bits = delta_swap(bits, mask3, 28);
    bits = delta_swap(bits, mask2, 14);
    bits = delta_swap(bits, mask1, 7);
    bits
}

pub fn flip_diag_a8h1(mut bits: u64) -> u64 {
    let mask1: u64 = 0xaa00aa00aa00aa00;
    let mask2: u64 = 0xcccc0000cccc0000;
    let mask3: u64 = 0xf0f0f0f000000000;

    bits = delta_swap(bits, mask3, 36);
    bits = delta_swap(bits, mask2, 18);
    bits = delta_swap(bits, mask1, 9);
    bits
}

#[inline]
fn delta_swap(bits: u64, mask: u64, delta: u32) -> u64 {
    let tmp = mask & (bits ^ (bits << delta));
    bits ^ tmp ^ (tmp >> delta)
}


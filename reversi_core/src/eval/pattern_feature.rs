use crate::bitboard;
use crate::bitboard::BitboardIterator;
use crate::board::Board;
use crate::move_list::Move;
use crate::square::Square;

pub const NUM_PATTERN_FEATURES: usize = 22;
type Sq = Square;

pub struct FeatureToCoordinate {
    pub n_square: usize,
    pub squares: [Square; 10],
}

macro_rules! ftc {
    ($n_square:expr, [$($square:expr),* $(,)?]) => {
        FeatureToCoordinate {
            n_square: $n_square,
            squares: [$($square),*],
        }
    };
}

#[derive(Clone, Copy)]
#[repr(align(32))]
pub union Feature {
    pub v1: [u16; 32],
    v16: [core::arch::x86_64::__m256i; 2],
}

impl Feature {
    fn new() -> Self {
        Self { v1: [0; 32] }
    }
}

#[derive(Debug, Clone, Copy)]
struct CoordinateToFeature {
    n_features: u32,
    features: [[u32; 2]; 4],
}

macro_rules! ctf {
    ($n_features:expr, [$([$f:expr, $i:expr],)*]) => {
        CoordinateToFeature {
            n_features: $n_features,
            features: [$([$f, $i],)*],
        }
    };
}

#[rustfmt::skip]
pub const EVAL_F2X: [FeatureToCoordinate; NUM_PATTERN_FEATURES] = [
    ftc!(8, [Sq::A1, Sq::B1, Sq::C1, Sq::D1, Sq::A2, Sq::A3, Sq::A4, Sq::B2, Sq::None, Sq::None]),
    ftc!(8, [Sq::H1, Sq::G1, Sq::F1, Sq::E1, Sq::H2, Sq::H3, Sq::H4, Sq::G2, Sq::None, Sq::None]),
    ftc!(8, [Sq::A8, Sq::B8, Sq::C8, Sq::D8, Sq::A7, Sq::A6, Sq::A5, Sq::B7, Sq::None, Sq::None]),
    ftc!(8, [Sq::H8, Sq::G8, Sq::F8, Sq::E8, Sq::H7, Sq::H6, Sq::H5, Sq::G7, Sq::None, Sq::None]),

    ftc!(8, [Sq::C2, Sq::D2, Sq::B3, Sq::C3, Sq::D3, Sq::B4, Sq::C4, Sq::D4, Sq::None, Sq::None]),
    ftc!(8, [Sq::F2, Sq::E2, Sq::G3, Sq::F3, Sq::E3, Sq::G4, Sq::F4, Sq::E4, Sq::None, Sq::None]),
    ftc!(8, [Sq::C7, Sq::D7, Sq::B6, Sq::C6, Sq::D6, Sq::B5, Sq::C5, Sq::D5, Sq::None, Sq::None]),
    ftc!(8, [Sq::F7, Sq::E7, Sq::G6, Sq::F6, Sq::E6, Sq::G5, Sq::F5, Sq::E5, Sq::None, Sq::None]),

    ftc!(8, [Sq::A1, Sq::B2, Sq::C3, Sq::D4, Sq::E5, Sq::F6, Sq::G7, Sq::H8, Sq::None, Sq::None]),
    ftc!(8, [Sq::H1, Sq::G2, Sq::F3, Sq::E4, Sq::D5, Sq::C6, Sq::B7, Sq::A8, Sq::None, Sq::None]),

    ftc!(8, [Sq::A1, Sq::B1, Sq::C1, Sq::D1, Sq::E1, Sq::F1, Sq::G1, Sq::H1, Sq::None, Sq::None]),
    ftc!(8, [Sq::A8, Sq::B8, Sq::C8, Sq::D8, Sq::E8, Sq::F8, Sq::G8, Sq::H8, Sq::None, Sq::None]),
    ftc!(8, [Sq::A1, Sq::A2, Sq::A3, Sq::A4, Sq::A5, Sq::A6, Sq::A7, Sq::A8, Sq::None, Sq::None]),
    ftc!(8, [Sq::H1, Sq::H2, Sq::H3, Sq::H4, Sq::H5, Sq::H6, Sq::H7, Sq::H8, Sq::None, Sq::None]),

    ftc!(8, [Sq::B1, Sq::C1, Sq::D1, Sq::E1, Sq::B2, Sq::C2, Sq::D2, Sq::E2, Sq::None, Sq::None]),
    ftc!(8, [Sq::G1, Sq::F1, Sq::E1, Sq::D1, Sq::G2, Sq::F2, Sq::E2, Sq::D2, Sq::None, Sq::None]),
    ftc!(8, [Sq::B8, Sq::C8, Sq::D8, Sq::E8, Sq::B7, Sq::C7, Sq::D7, Sq::E7, Sq::None, Sq::None]),
    ftc!(8, [Sq::G8, Sq::F8, Sq::E8, Sq::D8, Sq::G7, Sq::F7, Sq::E7, Sq::D7, Sq::None, Sq::None]),

    ftc!(8, [Sq::A2, Sq::A3, Sq::A4, Sq::A5, Sq::B2, Sq::B3, Sq::B4, Sq::B5, Sq::None, Sq::None]),
    ftc!(8, [Sq::A7, Sq::A6, Sq::A5, Sq::A4, Sq::B7, Sq::B6, Sq::B5, Sq::B4, Sq::None, Sq::None]),
    ftc!(8, [Sq::H2, Sq::H3, Sq::H4, Sq::H5, Sq::G2, Sq::G3, Sq::G4, Sq::G5, Sq::None, Sq::None]),
    ftc!(8, [Sq::H7, Sq::H6, Sq::H5, Sq::H4, Sq::G7, Sq::G6, Sq::G5, Sq::G4, Sq::None, Sq::None]),
];

#[rustfmt::skip]
const EVAL_FEATURE: [Feature; 64] = [
    Feature { v1: [2187, 0, 0, 0, 0, 0, 0, 0, 2187, 0, 2187, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 243, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 243, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 81, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 2187, 0, 0, 0, 0, 0, 0, 0, 2187, 1, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [1, 0, 0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 27, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 243, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 81, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 81, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 243, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 1, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 1, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 243, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 81, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 1, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 81, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 81, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 81, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 27, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 2187, 0, 0, 0, 0, 0, 0, 1, 0, 2187, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 243, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 243, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 81, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 2187, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
];

#[rustfmt::skip]
static EVAL_X2F: [CoordinateToFeature; 64] = [
    ctf!(4, [[0, 2187],[8, 2187],[10, 2187],[12, 2187],]),
    ctf!(3, [[0, 729],[10, 729],[14, 2187],[0, 0],]),
    ctf!(3, [[0, 243],[10, 243],[14, 729],[0, 0],]),
    ctf!(4, [[0, 81],[10, 81],[14, 243],[15, 81],]),
    ctf!(4, [[1, 81],[10, 27],[14, 81],[15, 243],]),
    ctf!(3, [[1, 243],[10, 9],[15, 729],[0, 0],]),
    ctf!(3, [[1, 729],[10, 3],[15, 2187],[0, 0],]),
    ctf!(4, [[1, 2187],[9, 2187],[10, 1],[13, 2187],]),
    ctf!(3, [[0, 27],[12, 729],[18, 2187],[0, 0],]),
    ctf!(4, [[0, 1],[8, 729],[14, 27],[18, 27],]),
    ctf!(2, [[4, 2187],[14, 9],[0, 0],[0, 0],]),
    ctf!(3, [[4, 729],[14, 3],[15, 1],[0, 0],]),
    ctf!(3, [[5, 729],[14, 1],[15, 3],[0, 0],]),
    ctf!(2, [[5, 2187],[15, 9],[0, 0],[0, 0],]),
    ctf!(4, [[1, 1],[9, 729],[15, 27],[20, 27],]),
    ctf!(3, [[1, 27],[13, 729],[20, 2187],[0, 0],]),
    ctf!(3, [[0, 9],[12, 243],[18, 729],[0, 0],]),
    ctf!(2, [[4, 243],[18, 9],[0, 0],[0, 0],]),
    ctf!(2, [[4, 81],[8, 243],[0, 0],[0, 0],]),
    ctf!(1, [[4, 27],[0, 0],[0, 0],[0, 0],]),
    ctf!(1, [[5, 27],[0, 0],[0, 0],[0, 0],]),
    ctf!(2, [[5, 81],[9, 243],[0, 0],[0, 0],]),
    ctf!(2, [[5, 243],[20, 9],[0, 0],[0, 0],]),
    ctf!(3, [[1, 9],[13, 243],[20, 729],[0, 0],]),
    ctf!(4, [[0, 3],[12, 81],[18, 243],[19, 81],]),
    ctf!(3, [[4, 9],[18, 3],[19, 1],[0, 0],]),
    ctf!(1, [[4, 3],[0, 0],[0, 0],[0, 0],]),
    ctf!(2, [[4, 1],[8, 81],[0, 0],[0, 0],]),
    ctf!(2, [[5, 1],[9, 81],[0, 0],[0, 0],]),
    ctf!(1, [[5, 3],[0, 0],[0, 0],[0, 0],]),
    ctf!(3, [[5, 9],[20, 3],[21, 1],[0, 0],]),
    ctf!(4, [[1, 3],[13, 81],[20, 243],[21, 81],]),
    ctf!(4, [[2, 3],[12, 27],[18, 81],[19, 243],]),
    ctf!(3, [[6, 9],[18, 1],[19, 3],[0, 0],]),
    ctf!(1, [[6, 3],[0, 0],[0, 0],[0, 0],]),
    ctf!(2, [[6, 1],[9, 27],[0, 0],[0, 0],]),
    ctf!(2, [[7, 1],[8, 27],[0, 0],[0, 0],]),
    ctf!(1, [[7, 3],[0, 0],[0, 0],[0, 0],]),
    ctf!(3, [[7, 9],[20, 1],[21, 3],[0, 0],]),
    ctf!(4, [[3, 3],[13, 27],[20, 81],[21, 243],]),
    ctf!(3, [[2, 9],[12, 9],[19, 729],[0, 0],]),
    ctf!(2, [[6, 243],[19, 9],[0, 0],[0, 0],]),
    ctf!(2, [[6, 81],[9, 9],[0, 0],[0, 0],]),
    ctf!(1, [[6, 27],[0, 0],[0, 0],[0, 0],]),
    ctf!(1, [[7, 27],[0, 0],[0, 0],[0, 0],]),
    ctf!(2, [[7, 81],[8, 9],[0, 0],[0, 0],]),
    ctf!(2, [[7, 243],[21, 9],[0, 0],[0, 0],]),
    ctf!(3, [[3, 9],[13, 9],[21, 729],[0, 0],]),
    ctf!(3, [[2, 27],[12, 3],[19, 2187],[0, 0],]),
    ctf!(4, [[2, 1],[9, 3],[16, 27],[19, 27],]),
    ctf!(2, [[6, 2187],[16, 9],[0, 0],[0, 0],]),
    ctf!(3, [[6, 729],[16, 3],[17, 1],[0, 0],]),
    ctf!(3, [[7, 729],[16, 1],[17, 3],[0, 0],]),
    ctf!(2, [[7, 2187],[17, 9],[0, 0],[0, 0],]),
    ctf!(4, [[3, 1],[8, 3],[17, 27],[21, 27],]),
    ctf!(3, [[3, 27],[13, 3],[21, 2187],[0, 0],]),
    ctf!(4, [[2, 2187],[9, 1],[11, 2187],[12, 1],]),
    ctf!(3, [[2, 729],[11, 729],[16, 2187],[0, 0],]),
    ctf!(3, [[2, 243],[11, 243],[16, 729],[0, 0],]),
    ctf!(4, [[2, 81],[11, 81],[16, 243],[17, 81],]),
    ctf!(4, [[3, 81],[11, 27],[16, 81],[17, 243],]),
    ctf!(3, [[3, 243],[11, 9],[17, 729],[0, 0],]),
    ctf!(3, [[3, 729],[11, 3],[17, 2187],[0, 0],]),
    ctf!(4, [[3, 2187],[8, 1],[11, 1],[13, 1],]),
];

#[derive(Clone)]
pub struct FeatureSet {
    pub p_features: [Feature; 61],
    pub o_features: [Feature; 61],
}

impl FeatureSet {
    pub fn new(board: &Board, ply: usize) -> FeatureSet {
        let mut feature_set = FeatureSet {
            p_features: [Feature::new(); 61],
            o_features: [Feature::new(); 61],
        };

        let o_board = board.switch_players();
        let p_feature = &mut feature_set.p_features[ply];
        let o_feature = &mut feature_set.o_features[ply];
        for (i, f2x) in EVAL_F2X.iter().enumerate() {
            for j in 0..f2x.n_square {
                let sq = f2x.squares[j];
                unsafe {
                    p_feature.v1[i] = p_feature.v1[i] * 3 + get_square_color(board, sq);
                    o_feature.v1[i] = o_feature.v1[i] * 3 + get_square_color(&o_board, sq);
                }
            }
        }
        feature_set
    }

    pub fn update(&mut self, mv: &Move, ply: usize, player: u8) {
        let flip = mv.flipped;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                let p_in = self.p_features[ply].v16;
                let p_out = &mut self.p_features[ply + 1].v16;
                let o_in = self.o_features[ply].v16;
                let o_out = &mut self.o_features[ply + 1].v16;

                let sq = mv.sq as usize;
                let f = &EVAL_FEATURE[sq].v16;

                let (p_scale, o_scale, p_sign, o_sign) = if player == 0 {
                    (
                        _mm256_set1_epi16(2),
                        _mm256_set1_epi16(1),
                        _mm256_set1_epi16(-1), // p: subtract sum
                        _mm256_set1_epi16(1),  // o: add sum
                    )
                } else {
                    (
                        _mm256_set1_epi16(1),
                        _mm256_set1_epi16(2),
                        _mm256_set1_epi16(1),  // p: add sum
                        _mm256_set1_epi16(-1), // o: subtract sum
                    )
                };

                p_out[0] = _mm256_sub_epi16(p_in[0], _mm256_mullo_epi16(f[0], p_scale));
                p_out[1] = _mm256_sub_epi16(p_in[1], _mm256_mullo_epi16(f[1], p_scale));
                o_out[0] = _mm256_sub_epi16(o_in[0], _mm256_mullo_epi16(f[0], o_scale));
                o_out[1] = _mm256_sub_epi16(o_in[1], _mm256_mullo_epi16(f[1], o_scale));

                let mut sum0 = _mm256_setzero_si256();
                let mut sum1 = _mm256_setzero_si256();
                for x in BitboardIterator::new(flip) {
                    let f = &EVAL_FEATURE[x as usize].v16;
                    sum0 = _mm256_add_epi16(sum0, f[0]);
                    sum1 = _mm256_add_epi16(sum1, f[1]);
                }

                p_out[0] = _mm256_add_epi16(p_out[0], _mm256_mullo_epi16(sum0, p_sign));
                p_out[1] = _mm256_add_epi16(p_out[1], _mm256_mullo_epi16(sum1, p_sign));
                o_out[0] = _mm256_add_epi16(o_out[0], _mm256_mullo_epi16(sum0, o_sign));
                o_out[1] = _mm256_add_epi16(o_out[1], _mm256_mullo_epi16(sum1, o_sign));
            }
        } else {
            self.p_features.copy_within(ply..ply + 1, ply + 1);
            self.o_features.copy_within(ply..ply + 1, ply + 1);
            let p_out = &mut self.p_features[ply + 1];
            let o_out = &mut self.o_features[ply + 1];
            let s = &EVAL_X2F[mv.sq as usize];

            if player == 0 {
                for i in 0..s.n_features {
                    let j = s.features[i as usize][0] as usize;
                    let x = s.features[i as usize][1] as usize;
                    unsafe {
                        p_out.v1[j] -= 2 * x as u16;
                        o_out.v1[j] -= x as u16;
                    }
                }

                for x in BitboardIterator::new(flip) {
                    let s_bit = &EVAL_X2F[x as usize];
                    for i in 0..s_bit.n_features {
                        let j = s_bit.features[i as usize][0] as usize;
                        let x = s_bit.features[i as usize][1] as usize;
                        unsafe {
                            p_out.v1[j] -= x as u16;
                            o_out.v1[j] += x as u16;
                        }
                    }
                }
            } else {
                for i in 0..s.n_features {
                    let j = s.features[i as usize][0] as usize;
                    let x = s.features[i as usize][1] as usize;
                    unsafe {
                        p_out.v1[j] -= x as u16;
                        o_out.v1[j] -= 2 * x as u16;
                    }
                }

                for x in BitboardIterator::new(flip) {
                    let s_bit = &EVAL_X2F[x as usize];
                    for i in 0..s_bit.n_features {
                        let j = s_bit.features[i as usize][0] as usize;
                        let x = s_bit.features[i as usize][1] as usize;
                        unsafe {
                            p_out.v1[j] += x as u16;
                            o_out.v1[j] -= x as u16;
                        }
                    }
                }
            }
        }
    }
}

pub fn set_features(board: &Board, features: &mut [u16]) {
    for i in 0..NUM_PATTERN_FEATURES {
        let f2x = &EVAL_F2X[i];
        for j in 0..f2x.n_square {
            let sq = f2x.squares[j];
            let c = get_square_color(board, sq);
            features[i] = features[i] * 3 + c;
        }
    }
}

fn get_square_color(board: &Board, sq: Square) -> u16 {
    if bitboard::is_set(board.player, sq) {
        0
    } else if bitboard::is_set(board.opponent, sq) {
        1
    } else {
        2
    }
}

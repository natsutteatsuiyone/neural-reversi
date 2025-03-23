use crate::bitboard;
use crate::bitboard::BitboardIterator;
use crate::board::Board;
use crate::move_list::Move;
use crate::square::Square;

pub const NUM_PATTERN_FEATURES: usize = 8;
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
    pub v1: [u16; 16],
    v16: [core::arch::x86_64::__m256i; 1],
}

impl Feature {
    fn new() -> Self {
        Self { v1: [0; 16] }
    }
}

#[derive(Debug, Clone, Copy)]
struct CoordinateToFeature {
    n_features: u32,
    features: [[u32; 2]; 1],
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
];

#[rustfmt::skip]
const EVAL_FEATURE: [Feature; 64] = [
    Feature { v1: [2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 0, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    Feature { v1: [0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
];

#[rustfmt::skip]
static EVAL_X2F: [CoordinateToFeature; 64] = [
    ctf!(1, [[0, 2187],]),
    ctf!(1, [[0, 729],]),
    ctf!(1, [[0, 243],]),
    ctf!(1, [[0, 81],]),
    ctf!(1, [[1, 81],]),
    ctf!(1, [[1, 243],]),
    ctf!(1, [[1, 729],]),
    ctf!(1, [[1, 2187],]),
    ctf!(1, [[0, 27],]),
    ctf!(1, [[0, 1],]),
    ctf!(1, [[4, 2187],]),
    ctf!(1, [[4, 729],]),
    ctf!(1, [[5, 729],]),
    ctf!(1, [[5, 2187],]),
    ctf!(1, [[1, 1],]),
    ctf!(1, [[1, 27],]),
    ctf!(1, [[0, 9],]),
    ctf!(1, [[4, 243],]),
    ctf!(1, [[4, 81],]),
    ctf!(1, [[4, 27],]),
    ctf!(1, [[5, 27],]),
    ctf!(1, [[5, 81],]),
    ctf!(1, [[5, 243],]),
    ctf!(1, [[1, 9],]),
    ctf!(1, [[0, 3],]),
    ctf!(1, [[4, 9],]),
    ctf!(1, [[4, 3],]),
    ctf!(1, [[4, 1],]),
    ctf!(1, [[5, 1],]),
    ctf!(1, [[5, 3],]),
    ctf!(1, [[5, 9],]),
    ctf!(1, [[1, 3],]),
    ctf!(1, [[2, 3],]),
    ctf!(1, [[6, 9],]),
    ctf!(1, [[6, 3],]),
    ctf!(1, [[6, 1],]),
    ctf!(1, [[7, 1],]),
    ctf!(1, [[7, 3],]),
    ctf!(1, [[7, 9],]),
    ctf!(1, [[3, 3],]),
    ctf!(1, [[2, 9],]),
    ctf!(1, [[6, 243],]),
    ctf!(1, [[6, 81],]),
    ctf!(1, [[6, 27],]),
    ctf!(1, [[7, 27],]),
    ctf!(1, [[7, 81],]),
    ctf!(1, [[7, 243],]),
    ctf!(1, [[3, 9],]),
    ctf!(1, [[2, 27],]),
    ctf!(1, [[2, 1],]),
    ctf!(1, [[6, 2187],]),
    ctf!(1, [[6, 729],]),
    ctf!(1, [[7, 729],]),
    ctf!(1, [[7, 2187],]),
    ctf!(1, [[3, 1],]),
    ctf!(1, [[3, 27],]),
    ctf!(1, [[2, 2187],]),
    ctf!(1, [[2, 729],]),
    ctf!(1, [[2, 243],]),
    ctf!(1, [[2, 81],]),
    ctf!(1, [[3, 81],]),
    ctf!(1, [[3, 243],]),
    ctf!(1, [[3, 729],]),
    ctf!(1, [[3, 2187],]),
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
            let offset = i as u16 * 3u16.pow(f2x.n_square as u32);
            unsafe {
                p_feature.v1[i] += offset;
                o_feature.v1[i] += offset;
            }
        }
        feature_set
    }

    pub fn update(&mut self, mv: &Move, ply: usize, player: u8) {
        let flip = mv.flipped;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                let p_feature_current = self.p_features[ply];
                let o_feature_current = self.o_features[ply];

                let p0 = _mm256_loadu_si256(p_feature_current.v16.as_ptr());
                let o0 = _mm256_loadu_si256(o_feature_current.v16.as_ptr());

                let sq = mv.sq as usize;
                let f_ptr = &EVAL_FEATURE[sq].v16;
                let f0 = _mm256_loadu_si256(f_ptr.as_ptr());

                let player_mask = if player == 0 {
                    _mm256_set1_epi16(2)
                } else {
                    _mm256_set1_epi16(1)
                };
                let other_mask = _mm256_sub_epi16(_mm256_set1_epi16(3), player_mask);

                let p_sub0 = _mm256_mullo_epi16(f0, player_mask);
                let o_sub0 = _mm256_mullo_epi16(f0, other_mask);

                let mut p_res0 = _mm256_sub_epi16(p0, p_sub0);
                let mut o_res0 = _mm256_sub_epi16(o0, o_sub0);

                let mut sum0 = _mm256_setzero_si256();
                for x in BitboardIterator::new(flip) {
                    let f_flip = &EVAL_FEATURE[x as usize].v16;
                    sum0 = _mm256_add_epi16(sum0, _mm256_loadu_si256(f_flip.as_ptr()));
                }

                if player == 0 {
                    p_res0 = _mm256_sub_epi16(p_res0, sum0);
                    o_res0 = _mm256_add_epi16(o_res0, sum0);
                } else {
                    p_res0 = _mm256_add_epi16(p_res0, sum0);
                    o_res0 = _mm256_sub_epi16(o_res0, sum0);
                }

                let p_out = &mut self.p_features[ply + 1].v16;
                let o_out = &mut self.o_features[ply + 1].v16;
                _mm256_storeu_si256(p_out.as_mut_ptr(), p_res0);
                _mm256_storeu_si256(o_out.as_mut_ptr(), o_res0);
            }
        }else{
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

use std::io::{self, Read};

use byteorder::{LittleEndian, ReadBytesExt};
use reversi_core::{
    board::Board,
    constants::{MID_SCORE_MAX, MID_SCORE_MIN},
    eval::{
        constants::{INPUT_FEATURE_DIMS, NUM_FEATURES, PATTERN_FEATURE_OFFSETS},
        pattern_feature::PatternFeature,
    },
    types::Score,
};

const NN_DIMS: usize = 256;
const NUM_OUTPUT_LAYERS: usize = 60;
const OUTPUT_WEIGHT_SCALE_BITS: u32 = 7;

#[repr(align(16))]
struct AlignedBiases([i16; NN_DIMS]);

#[repr(align(16))]
struct AlignedOutputWeights([i16; NN_DIMS]);

struct AlignedWeights {
    ptr: *mut i8,
    len: usize,
}

impl AlignedWeights {
    fn new(len: usize) -> Self {
        use std::alloc::{Layout, alloc_zeroed};
        unsafe {
            let layout = Layout::from_size_align(len, 16).unwrap();
            let ptr = alloc_zeroed(layout) as *mut i8;
            Self { ptr, len }
        }
    }

    #[allow(dead_code)]
    fn as_ptr(&self) -> *const i8 {
        self.ptr
    }

    #[allow(dead_code)]
    fn as_slice(&self) -> &[i8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [i8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedWeights {
    fn drop(&mut self) {
        use std::alloc::{Layout, dealloc};
        unsafe {
            let layout = Layout::from_size_align(self.len, 16).unwrap();
            dealloc(self.ptr as *mut u8, layout);
        }
    }
}

struct InputLayer {
    biases: AlignedBiases,
    weights: AlignedWeights,
}

impl InputLayer {
    fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases_array = [0i16; NN_DIMS];
        let mut weights = AlignedWeights::new(INPUT_FEATURE_DIMS * NN_DIMS);

        reader.read_i16_into::<LittleEndian>(&mut biases_array)?;
        reader.read_i8_into(weights.as_mut_slice())?;

        Ok(Self {
            biases: AlignedBiases(biases_array),
            weights,
        })
    }
}

struct OutputLayer {
    bias: i32,
    weights: AlignedOutputWeights,
}

impl OutputLayer {
    fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let bias = reader.read_i32::<LittleEndian>()?;

        let mut weights_array = [0i16; NN_DIMS];
        reader.read_i16_into::<LittleEndian>(&mut weights_array)?;

        Ok(Self {
            bias,
            weights: AlignedOutputWeights(weights_array),
        })
    }
}

pub struct Network {
    input_layer: InputLayer,
    output_layers: Vec<OutputLayer>,
}

impl Network {
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let cursor = io::Cursor::new(bytes);
        Self::from_reader(cursor)
    }

    fn from_reader<R: Read>(reader: R) -> io::Result<Self> {
        let mut decoder = zstd::stream::read::Decoder::new(reader)?;

        let input_layer = InputLayer::load(&mut decoder)?;

        let mut output_layers = Vec::with_capacity(NUM_OUTPUT_LAYERS);
        for _ in 0..NUM_OUTPUT_LAYERS {
            let output_layer = OutputLayer::load(&mut decoder)?;
            output_layers.push(output_layer);
        }

        Ok(Network {
            input_layer,
            output_layers,
        })
    }

    pub fn evaluate(&self, _board: &Board, pattern_feature: &PatternFeature, ply: usize) -> Score {
        let output_layer = &self.output_layers[ply];
        let score: i32;

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            score = self.forward_simd(pattern_feature, &self.input_layer, output_layer);
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            score = self.forward_scalar(pattern_feature, &self.input_layer, output_layer);
        }

        (score >> OUTPUT_WEIGHT_SCALE_BITS).clamp(MID_SCORE_MIN + 1, MID_SCORE_MAX - 1)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    fn forward_simd(
        &self,
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        output_layer: &OutputLayer,
    ) -> i32 {
        use std::arch::wasm32::*;

        const NUM_I16X8: usize = 32;

        unsafe {
            let mut acc: [v128; NUM_I16X8] = std::mem::zeroed();

            let bias_ptr = input_layer.biases.0.as_ptr() as *const v128;
            std::ptr::copy_nonoverlapping(bias_ptr, acc.as_mut_ptr(), NUM_I16X8);

            let weights = input_layer.weights.as_ptr();

            for feature_idx in 0..NUM_FEATURES {
                let offset = feature_offset(pattern_feature, feature_idx);
                let row_ptr = weights.add(offset * NN_DIMS) as *const v128;

                macro_rules! process_chunk {
                    ($chunk:expr) => {{
                        let w = v128_load(row_ptr.add($chunk));
                        let idx = $chunk * 2;
                        acc[idx] = i16x8_add(acc[idx], i16x8_extend_low_i8x16(w));
                        acc[idx + 1] = i16x8_add(acc[idx + 1], i16x8_extend_high_i8x16(w));
                    }};
                }

                process_chunk!(0);
                process_chunk!(1);
                process_chunk!(2);
                process_chunk!(3);
                process_chunk!(4);
                process_chunk!(5);
                process_chunk!(6);
                process_chunk!(7);
                process_chunk!(8);
                process_chunk!(9);
                process_chunk!(10);
                process_chunk!(11);
                process_chunk!(12);
                process_chunk!(13);
                process_chunk!(14);
                process_chunk!(15);
            }

            // 出力層
            let zero = i16x8_splat(0);
            let one = i16x8_splat(255);
            let out_w = output_layer.weights.0.as_ptr() as *const v128;

            let mut s0 = i32x4_splat(0);
            let mut s1 = i32x4_splat(0);
            let mut s2 = i32x4_splat(0);
            let mut s3 = i32x4_splat(0);

            macro_rules! process_output {
                ($idx:expr, $sum:expr) => {{
                    let w = v128_load(out_w.add($idx));
                    let relu = i16x8_max(acc[$idx], zero);
                    let clamped = i16x8_min(relu, one);
                    let sq = u16x8_mul(clamped, clamped);
                    let activation = u16x8_shr(sq, 8);
                    let low = i32x4_extmul_low_i16x8(activation, w);
                    let high = i32x4_extmul_high_i16x8(activation, w);
                    $sum = i32x4_add($sum, low);
                    $sum = i32x4_add($sum, high);
                }};
            }

            for i in (0..NUM_I16X8).step_by(4) {
                process_output!(i, s0);
                process_output!(i + 1, s1);
                process_output!(i + 2, s2);
                process_output!(i + 3, s3);
            }

            let total = i32x4_add(i32x4_add(s0, s1), i32x4_add(s2, s3));

            output_layer.bias
                + i32x4_extract_lane::<0>(total)
                + i32x4_extract_lane::<1>(total)
                + i32x4_extract_lane::<2>(total)
                + i32x4_extract_lane::<3>(total)
        }
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    fn forward_scalar(
        &self,
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        output_layer: &OutputLayer,
    ) -> i32 {
        let mut acc: [i16; NN_DIMS] = input_layer.biases.0.clone();

        // accumulate
        let weights = input_layer.weights.as_slice();
        for feature_idx in 0..NUM_FEATURES {
            let offset = feature_offset(pattern_feature, feature_idx);
            let row = &weights[offset * NN_DIMS..(offset + 1) * NN_DIMS];
            for i in 0..NN_DIMS {
                acc[i] += row[i] as i16;
            }
        }

        // activation and output
        let mut output = 0i32;
        for (value, &weight) in acc.iter().zip(output_layer.weights.0.iter()) {
            if *value > 0 {
                let clamped = (*value).min(255) as i32;
                let activation = (clamped * clamped) >> 8;
                output += activation * (weight as i32);
            }
        }

        output + output_layer.bias
    }
}

#[inline(always)]
fn feature_offset(pattern_feature: &PatternFeature, idx: usize) -> usize {
    *unsafe { PATTERN_FEATURE_OFFSETS.get_unchecked(idx) }
        + unsafe { pattern_feature.get_unchecked(idx) } as usize
}

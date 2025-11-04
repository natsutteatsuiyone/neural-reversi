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

const NN_DIMS: usize = 128;
const NUM_OUTPUT_LAYERS: usize = 60;
const OUTPUT_WEIGHT_SCALE_BITS: u32 = 8;

#[derive(Debug)]
struct InputLayer {
    biases: [i16; NN_DIMS],
    weights: Vec<i16>,
}

impl InputLayer {
    fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = [0i16; NN_DIMS];
        let mut weights = vec![0i16; INPUT_FEATURE_DIMS * NN_DIMS];

        reader.read_i16_into::<LittleEndian>(&mut biases)?;
        reader.read_i16_into::<LittleEndian>(&mut weights)?;

        Ok(Self { biases, weights })
    }
}

#[derive(Debug)]
struct OutputLayer {
    bias: i32,
    weights: [i16; NN_DIMS],
}

impl OutputLayer {
    fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let bias = reader.read_i32::<LittleEndian>()?;

        let mut weights = [0i16; NN_DIMS];
        reader.read_i16_into::<LittleEndian>(&mut weights)?;

        Ok(Self { bias, weights })
    }
}

pub struct Network {
    input_layer: InputLayer,
    output_layers: Vec<OutputLayer>,
}

impl Network {
    /// Creates a new small network from an in-memory blob
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
        const CHUNK_SIZE: usize = 8;
        const NUM_CHUNKS: usize = NN_DIMS / CHUNK_SIZE;

        unsafe {
            let mut acc: [v128; NN_DIMS / 4] = std::mem::zeroed();
            for (i, bias_chunk) in input_layer.biases.chunks_exact(4).enumerate() {
                acc[i] = i32x4(
                    bias_chunk[0] as i32,
                    bias_chunk[1] as i32,
                    bias_chunk[2] as i32,
                    bias_chunk[3] as i32,
                );
            }

            let weights = input_layer.weights.as_slice();

            for feature_idx in 0..NUM_FEATURES {
                let offset = feature_offset(pattern_feature, feature_idx);
                let row_ptr = weights.as_ptr().add(offset * NN_DIMS);

                for chunk_idx in 0..NUM_CHUNKS {
                    let weight_vec = v128_load(row_ptr.add(chunk_idx * CHUNK_SIZE) as *const v128);

                    let weight_low = i32x4_extend_low_i16x8(weight_vec);
                    let weight_high = i32x4_extend_high_i16x8(weight_vec);

                    let acc_idx = chunk_idx * 2;
                    acc[acc_idx] = i32x4_add(acc[acc_idx], weight_low);
                    acc[acc_idx + 1] = i32x4_add(acc[acc_idx + 1], weight_high);
                }
            }

            let mut output = output_layer.bias;
            let zero = i32x4_splat(0);
            let clamp_max = i32x4_splat(1023);

            for i in 0..(NN_DIMS / 4) {
                let value = acc[i];

                let relu_val = i32x4_max(value, zero);
                let clamped = i32x4_min(relu_val, clamp_max);

                let squared = i32x4_mul(clamped, clamped);
                let activated = i32x4_shr(squared, 10);

                let weight_slice = &output_layer.weights[i * 4..i * 4 + 4];
                let weight = i32x4(
                    weight_slice[0] as i32,
                    weight_slice[1] as i32,
                    weight_slice[2] as i32,
                    weight_slice[3] as i32,
                );

                let prod = i32x4_mul(activated, weight);

                output += i32x4_extract_lane::<0>(prod);
                output += i32x4_extract_lane::<1>(prod);
                output += i32x4_extract_lane::<2>(prod);
                output += i32x4_extract_lane::<3>(prod);
            }
            output
        }
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    fn forward_scalar(
        &self,
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        output_layer: &OutputLayer,
    ) -> i32 {
        let mut acc: [i32; NN_DIMS] = input_layer.biases.map(|b| b as i32);

        // accumulate
        let weights = input_layer.weights.as_slice();
        for feature_idx in 0..NUM_FEATURES {
            let offset = feature_offset(pattern_feature, feature_idx);
            let row = &weights[offset * NN_DIMS..(offset + 1) * NN_DIMS];
            for neuron_idx in 0..NN_DIMS {
                acc[neuron_idx] += row[neuron_idx] as i32;
            }
        }

        // activation and output
        let mut output = 0i32;
        for (value, &weight) in acc.iter().zip(output_layer.weights.iter()) {
            if *value > 0 {
                let clamped = (*value).min(1023);
                let activation = (clamped * clamped) >> 10;
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

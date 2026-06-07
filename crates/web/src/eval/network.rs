use std::io::{self, Read};

use byteorder::{LittleEndian, ReadBytesExt};
use reversi_core::{
    eval::pattern_feature::{
        INPUT_FEATURE_DIMS, NUM_FEATURES, PATTERN_FEATURE_OFFSETS, PatternFeature,
    },
    types::ScaledScore,
};

const NN_DIMS: usize = 256;
const HIDDEN_DIMS: usize = 8;
const OUTPUT_DIMS: usize = HIDDEN_DIMS + NN_DIMS;
const NUM_LAYER_STACKS: usize = 60;
const HIDDEN_WEIGHT_SCALE_BITS: u32 = 6;
const OUTPUT_WEIGHT_SCALE_BITS: u32 = 7;
const QUANTIZED_ONE: i32 = 255;

type L1Layer = HiddenLayer<NN_DIMS, { NN_DIMS * HIDDEN_DIMS }>;
type L2Layer = HiddenLayer<HIDDEN_DIMS, { HIDDEN_DIMS * HIDDEN_DIMS }>;

#[repr(align(16))]
#[derive(Clone, Copy)]
struct AlignedI16Array<const N: usize>([i16; N]);

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
    biases: AlignedI16Array<NN_DIMS>,
    weights: AlignedWeights,
}

impl InputLayer {
    fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases_array = [0i16; NN_DIMS];
        let mut weights = AlignedWeights::new(INPUT_FEATURE_DIMS * NN_DIMS);

        reader.read_i16_into::<LittleEndian>(&mut biases_array)?;
        reader.read_i8_into(weights.as_mut_slice())?;

        Ok(Self {
            biases: AlignedI16Array(biases_array),
            weights,
        })
    }
}

struct HiddenLayer<const IN_DIMS: usize, const WEIGHTS_LEN: usize> {
    biases: [i32; HIDDEN_DIMS],
    weights: AlignedI16Array<WEIGHTS_LEN>,
}

impl<const IN_DIMS: usize, const WEIGHTS_LEN: usize> HiddenLayer<IN_DIMS, WEIGHTS_LEN> {
    fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        debug_assert_eq!(WEIGHTS_LEN, IN_DIMS * HIDDEN_DIMS);

        let mut biases = [0i32; HIDDEN_DIMS];
        let mut weights = [0i16; WEIGHTS_LEN];

        reader.read_i32_into::<LittleEndian>(&mut biases)?;
        reader.read_i16_into::<LittleEndian>(&mut weights)?;

        Ok(Self {
            biases,
            weights: AlignedI16Array(weights),
        })
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    fn forward_scalar(&self, input: &[i16; IN_DIMS]) -> [i16; HIDDEN_DIMS] {
        let mut output = [0i16; HIDDEN_DIMS];

        for (out_idx, value) in output.iter_mut().enumerate() {
            let row_begin = out_idx * IN_DIMS;
            let row = &self.weights.0[row_begin..row_begin + IN_DIMS];
            let mut acc = self.biases[out_idx] as i64;

            for (&input_value, &weight) in input.iter().zip(row.iter()) {
                acc += input_value as i64 * weight as i64;
            }

            *value = screlu_quantized_i64(acc >> HIDDEN_WEIGHT_SCALE_BITS);
        }

        output
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    fn forward_simd(&self, input: &[i16; IN_DIMS]) -> [i16; HIDDEN_DIMS] {
        use std::arch::wasm32::*;

        debug_assert_eq!(IN_DIMS % 8, 0);

        let mut output = [0i16; HIDDEN_DIMS];
        unsafe {
            let input_ptr = input.as_ptr() as *const v128;
            let weights_ptr = self.weights.0.as_ptr() as *const v128;
            let chunks = IN_DIMS / 8;
            let mut sums: [v128; HIDDEN_DIMS] = std::mem::zeroed();

            macro_rules! process_row {
                ($out_idx:expr, $activation:expr, $chunk:expr) => {{
                    let weight = v128_load(weights_ptr.add($out_idx * chunks + $chunk));
                    sums[$out_idx] =
                        i32x4_add(sums[$out_idx], i32x4_extmul_low_i16x8($activation, weight));
                    sums[$out_idx] =
                        i32x4_add(sums[$out_idx], i32x4_extmul_high_i16x8($activation, weight));
                }};
            }

            for chunk in 0..chunks {
                let activation = v128_load(input_ptr.add(chunk));
                process_row!(0, activation, chunk);
                process_row!(1, activation, chunk);
                process_row!(2, activation, chunk);
                process_row!(3, activation, chunk);
                process_row!(4, activation, chunk);
                process_row!(5, activation, chunk);
                process_row!(6, activation, chunk);
                process_row!(7, activation, chunk);
            }

            for (out_idx, value) in output.iter_mut().enumerate() {
                let sum = sums[out_idx];
                let acc = self.biases[out_idx] as i64
                    + i32x4_extract_lane::<0>(sum) as i64
                    + i32x4_extract_lane::<1>(sum) as i64
                    + i32x4_extract_lane::<2>(sum) as i64
                    + i32x4_extract_lane::<3>(sum) as i64;
                *value = screlu_quantized_i64(acc >> HIDDEN_WEIGHT_SCALE_BITS);
            }
        }

        output
    }
}

struct OutputLayer {
    bias: i32,
    weights: AlignedI16Array<OUTPUT_DIMS>,
}

impl OutputLayer {
    fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let bias = reader.read_i32::<LittleEndian>()?;

        let mut weights_array = [0i16; OUTPUT_DIMS];
        reader.read_i16_into::<LittleEndian>(&mut weights_array)?;

        Ok(Self {
            bias,
            weights: AlignedI16Array(weights_array),
        })
    }
}

struct LayerStack {
    l1: L1Layer,
    l2: L2Layer,
    output: OutputLayer,
}

impl LayerStack {
    fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        Ok(Self {
            l1: L1Layer::load(reader)?,
            l2: L2Layer::load(reader)?,
            output: OutputLayer::load(reader)?,
        })
    }
}

/// Neural network for position evaluation.
pub struct Network {
    input_layer: InputLayer,
    layer_stacks: Vec<LayerStack>,
}

impl Network {
    /// Loads a network from zstd-compressed weight data.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if decompression or deserialization fails.
    ///
    /// [`io::Error`]: std::io::Error
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let cursor = io::Cursor::new(bytes);
        Self::from_reader(cursor)
    }

    fn from_reader<R: Read>(reader: R) -> io::Result<Self> {
        let mut decoder = zstd::stream::read::Decoder::new(reader)?;

        let input_layer = InputLayer::load(&mut decoder)?;

        let mut layer_stacks = Vec::with_capacity(NUM_LAYER_STACKS);
        for _ in 0..NUM_LAYER_STACKS {
            layer_stacks.push(LayerStack::load(&mut decoder)?);
        }

        Ok(Network {
            input_layer,
            layer_stacks,
        })
    }

    /// Evaluates a board position and returns the score for the current ply.
    ///
    /// # Panics
    ///
    /// Panics if `ply` is out of range for the layer stacks.
    pub fn evaluate(&self, pattern_feature: &PatternFeature, ply: usize) -> ScaledScore {
        let layer_stack = &self.layer_stacks[ply];
        let score: i32;

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            score = self.forward_simd(pattern_feature, &self.input_layer, layer_stack);
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            score = self.forward_scalar(pattern_feature, &self.input_layer, layer_stack);
        }

        let score = ScaledScore::from_raw(score >> OUTPUT_WEIGHT_SCALE_BITS);
        score.clamp(ScaledScore::MIN + 1, ScaledScore::MAX - 1)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    fn forward_simd(
        &self,
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        layer_stack: &LayerStack,
    ) -> i32 {
        use std::arch::wasm32::*;

        unsafe {
            let weights = input_layer.weights.as_ptr();
            let mut row_ptrs = std::mem::MaybeUninit::<[*const v128; NUM_FEATURES]>::uninit();
            let row_ptrs_ptr = row_ptrs.as_mut_ptr() as *mut *const v128;

            for feature_idx in 0..NUM_FEATURES {
                let offset = feature_offset(pattern_feature, feature_idx);
                row_ptrs_ptr
                    .add(feature_idx)
                    .write(weights.add(offset * NN_DIMS) as *const v128);
            }
            // SAFETY: the loop above writes every NUM_FEATURES pointer slot exactly once.
            let row_ptrs = row_ptrs.assume_init();

            let bias_ptr = input_layer.biases.0.as_ptr() as *const v128;
            let zero = i16x8_splat(0);
            let one = i16x8_splat(QUANTIZED_ONE as i16);
            let mut input_activation = std::mem::MaybeUninit::<AlignedI16Array<NN_DIMS>>::uninit();
            let input_activation_ptr =
                std::ptr::addr_of_mut!((*input_activation.as_mut_ptr()).0) as *mut i16;
            let out_w = layer_stack.output.weights.0.as_ptr().add(HIDDEN_DIMS) as *const v128;

            macro_rules! activate_store {
                ($idx:expr, $acc:expr) => {{
                    let relu = i16x8_max($acc, zero);
                    let clamped = i16x8_min(relu, one);
                    let sq = u16x8_mul(clamped, clamped);
                    let activation = u16x8_shr(sq, 8);
                    v128_store(input_activation_ptr.add($idx * 8) as *mut v128, activation);
                }};
            }

            macro_rules! process_block {
                ($block:expr) => {{
                    let c = $block * 4;
                    let mut a0 = v128_load(bias_ptr.add(c * 2));
                    let mut a1 = v128_load(bias_ptr.add(c * 2 + 1));
                    let mut a2 = v128_load(bias_ptr.add(c * 2 + 2));
                    let mut a3 = v128_load(bias_ptr.add(c * 2 + 3));
                    let mut a4 = v128_load(bias_ptr.add(c * 2 + 4));
                    let mut a5 = v128_load(bias_ptr.add(c * 2 + 5));
                    let mut a6 = v128_load(bias_ptr.add(c * 2 + 6));
                    let mut a7 = v128_load(bias_ptr.add(c * 2 + 7));

                    for &row_ptr in &row_ptrs {
                        let w0 = v128_load(row_ptr.add(c));
                        let w1 = v128_load(row_ptr.add(c + 1));
                        let w2 = v128_load(row_ptr.add(c + 2));
                        let w3 = v128_load(row_ptr.add(c + 3));
                        a0 = i16x8_add(a0, i16x8_extend_low_i8x16(w0));
                        a1 = i16x8_add(a1, i16x8_extend_high_i8x16(w0));
                        a2 = i16x8_add(a2, i16x8_extend_low_i8x16(w1));
                        a3 = i16x8_add(a3, i16x8_extend_high_i8x16(w1));
                        a4 = i16x8_add(a4, i16x8_extend_low_i8x16(w2));
                        a5 = i16x8_add(a5, i16x8_extend_high_i8x16(w2));
                        a6 = i16x8_add(a6, i16x8_extend_low_i8x16(w3));
                        a7 = i16x8_add(a7, i16x8_extend_high_i8x16(w3));
                    }

                    activate_store!(c * 2, a0);
                    activate_store!(c * 2 + 1, a1);
                    activate_store!(c * 2 + 2, a2);
                    activate_store!(c * 2 + 3, a3);
                    activate_store!(c * 2 + 4, a4);
                    activate_store!(c * 2 + 5, a5);
                    activate_store!(c * 2 + 6, a6);
                    activate_store!(c * 2 + 7, a7);
                }};
            }

            process_block!(0);
            process_block!(1);
            process_block!(2);
            process_block!(3);

            let act_ptr = input_activation_ptr as *const v128;
            let mut s0 = i32x4_splat(0);
            let mut s1 = i32x4_splat(0);
            let mut s2 = i32x4_splat(0);
            let mut s3 = i32x4_splat(0);

            macro_rules! skip_dot {
                ($idx:expr, $sum:expr) => {{
                    let a = v128_load(act_ptr.add($idx));
                    let w = v128_load(out_w.add($idx));
                    $sum = i32x4_add($sum, i32x4_extmul_low_i16x8(a, w));
                    $sum = i32x4_add($sum, i32x4_extmul_high_i16x8(a, w));
                }};
            }

            let mut idx = 0;
            while idx < NN_DIMS / 8 {
                skip_dot!(idx, s0);
                skip_dot!(idx + 1, s1);
                skip_dot!(idx + 2, s2);
                skip_dot!(idx + 3, s3);
                idx += 4;
            }

            let total = i32x4_add(i32x4_add(s0, s1), i32x4_add(s2, s3));
            let mut output = layer_stack.output.bias
                + i32x4_extract_lane::<0>(total)
                + i32x4_extract_lane::<1>(total)
                + i32x4_extract_lane::<2>(total)
                + i32x4_extract_lane::<3>(total);

            // SAFETY: the 4 blocks above write all 32 i16x8 lanes in NN_DIMS.
            let input_activation = input_activation.assume_init();
            let l1 = layer_stack.l1.forward_simd(&input_activation.0);
            let l2 = layer_stack.l2.forward_simd(&l1);

            for (idx, &value) in l2.iter().enumerate() {
                output += value as i32 * layer_stack.output.weights.0[idx] as i32;
            }

            output
        }
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    fn forward_scalar(
        &self,
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        layer_stack: &LayerStack,
    ) -> i32 {
        let input_activation = input_activation_scalar(pattern_feature, input_layer);
        let l1 = layer_stack.l1.forward_scalar(&input_activation.0);
        let l2 = layer_stack.l2.forward_scalar(&l1);

        let mut output = layer_stack.output.bias;

        for (idx, &value) in l2.iter().enumerate() {
            output += value as i32 * layer_stack.output.weights.0[idx] as i32;
        }

        for (idx, &value) in input_activation.0.iter().enumerate() {
            output += value as i32 * layer_stack.output.weights.0[HIDDEN_DIMS + idx] as i32;
        }

        output
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn input_activation_scalar(
    pattern_feature: &PatternFeature,
    input_layer: &InputLayer,
) -> AlignedI16Array<NN_DIMS> {
    let mut acc: [i16; NN_DIMS] = input_layer.biases.0;

    let weights = input_layer.weights.as_slice();
    for feature_idx in 0..NUM_FEATURES {
        let offset = feature_offset(pattern_feature, feature_idx);
        let row = &weights[offset * NN_DIMS..(offset + 1) * NN_DIMS];
        for i in 0..NN_DIMS {
            acc[i] += row[i] as i16;
        }
    }

    for value in &mut acc {
        *value = screlu_quantized(*value as i32);
    }

    AlignedI16Array(acc)
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline(always)]
fn screlu_quantized(value: i32) -> i16 {
    let clamped = value.clamp(0, QUANTIZED_ONE);
    ((clamped * clamped) >> 8) as i16
}

#[inline(always)]
fn screlu_quantized_i64(value: i64) -> i16 {
    let clamped = value.clamp(0, QUANTIZED_ONE as i64);
    ((clamped * clamped) >> 8) as i16
}

#[inline(always)]
fn feature_offset(pattern_feature: &PatternFeature, idx: usize) -> usize {
    *unsafe { PATTERN_FEATURE_OFFSETS.get_unchecked(idx) }
        + unsafe { pattern_feature.get_unchecked(idx) } as usize
}

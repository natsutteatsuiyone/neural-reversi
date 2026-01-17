//! Small neural network for endgame evaluation.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::constants::CACHE_LINE_SIZE;
use crate::eval::pattern_feature::{INPUT_FEATURE_DIMS, NUM_FEATURES, PatternFeature};
use crate::eval::util::feature_offset;
use crate::types::ScaledScore;
use crate::util::align::Align64;

const PA_OUTPUT_DIMS: usize = 128;

const OUTPUT_WEIGHT_SCALE_BITS: u32 = 8;
const NUM_INPUT_LAYERS: usize = 3;
const NUM_OUTPUT_LAYERS: usize = 30;
const ENDGAME_START_PLY: usize = 30;
const INPUT_LAYER_SEGMENT_SIZE: usize = NUM_OUTPUT_LAYERS / NUM_INPUT_LAYERS;

/// Input layer for the small network.
#[derive(Debug)]
struct InputLayer {
    biases: Align64<[i16; PA_OUTPUT_DIMS]>,
    weights: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
}

impl InputLayer {
    fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = Align64([0i16; PA_OUTPUT_DIMS]);
        let mut weights = avec![[CACHE_LINE_SIZE]|0i16; INPUT_FEATURE_DIMS * PA_OUTPUT_DIMS];

        reader.read_i16_into::<LittleEndian>(biases.as_mut_slice())?;
        reader.read_i16_into::<LittleEndian>(&mut weights)?;

        Ok(Self { biases, weights })
    }
}

/// Output layer for the small network.
#[derive(Debug)]
struct OutputLayer {
    bias: i32,
    weights: Align64<[i16; PA_OUTPUT_DIMS]>,
}

impl OutputLayer {
    fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let bias = reader.read_i32::<LittleEndian>()?;

        let mut weights = Align64([0i16; PA_OUTPUT_DIMS]);
        reader.read_i16_into::<LittleEndian>(weights.as_mut_slice())?;

        Ok(Self { bias, weights })
    }
}

/// Small neural network optimized for endgame positions.
pub struct NetworkSmall {
    input_layers: Vec<InputLayer>,
    output_layers: Vec<OutputLayer>,
    forward_fn: unsafe fn(&PatternFeature, &InputLayer, &OutputLayer) -> i32,
}

impl NetworkSmall {
    /// Creates a new small network from compressed weights file.
    pub fn new(file_path: &Path) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        Self::from_reader(reader)
    }

    /// Creates a new small network from an in-memory blob.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let cursor = io::Cursor::new(bytes);
        Self::from_reader(cursor)
    }

    fn from_reader<R: Read>(reader: R) -> io::Result<Self> {
        let mut decoder = zstd::stream::read::Decoder::new(reader)?;

        let mut input_layers = Vec::with_capacity(NUM_INPUT_LAYERS);
        for _ in 0..NUM_INPUT_LAYERS {
            let input_layer = InputLayer::load(&mut decoder)?;
            input_layers.push(input_layer);
        }

        let mut output_layers = Vec::with_capacity(NUM_OUTPUT_LAYERS);
        for _ in 0..NUM_OUTPUT_LAYERS {
            let output_layer = OutputLayer::load(&mut decoder)?;
            output_layers.push(output_layer);
        }

        // Select the optimal forward implementation at load time
        let forward_fn = Self::select_forward_fn();

        Ok(NetworkSmall {
            input_layers,
            output_layers,
            forward_fn,
        })
    }

    /// Selects the optimal forward implementation based on CPU features.
    fn select_forward_fn() -> unsafe fn(&PatternFeature, &InputLayer, &OutputLayer) -> i32 {
        use cfg_if::cfg_if;
        cfg_if! {
            if #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))] {
                if is_x86_feature_detected!("avx512vnni") {
                    Self::forward_avx512_vnni
                } else {
                    Self::forward_avx512_no_vnni
                }
            } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
                if is_x86_feature_detected!("avxvnni") {
                    Self::forward_avx2_vnni
                } else {
                    Self::forward_avx2_no_vnni
                }
            } else {
                Self::forward_scalar_wrapper
            }
        }
    }

    /// Evaluates a position using the small network.
    ///
    /// Faster but less accurate than the main network.
    ///
    /// # Arguments
    ///
    /// * `pattern_feature` - Pattern features from the board.
    /// * `ply` - Current game ply.
    pub fn evaluate(&self, pattern_feature: &PatternFeature, ply: usize) -> ScaledScore {
        debug_assert!(ply >= ENDGAME_START_PLY);
        debug_assert_eq!(NUM_OUTPUT_LAYERS % NUM_INPUT_LAYERS, 0);

        let ply_offset = ply - ENDGAME_START_PLY;
        let input_layer_index = ply_offset / INPUT_LAYER_SEGMENT_SIZE;
        let input_layer = &self.input_layers[input_layer_index];

        let output_layer = &self.output_layers[ply_offset];

        let sum = unsafe { (self.forward_fn)(pattern_feature, input_layer, output_layer) };
        let total = sum + output_layer.bias;
        let score = ScaledScore::from_raw(total >> OUTPUT_WEIGHT_SCALE_BITS);

        score.clamp(ScaledScore::MIN + 1, ScaledScore::MAX - 1)
    }

    /// AVX-512 accelerated forward pass optionally using VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw,avx512vnni")]
    fn forward_avx512_vnni(
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        output_layer: &OutputLayer,
    ) -> i32 {
        Self::forward_avx512::<true>(pattern_feature, input_layer, output_layer)
    }

    // AVX-512 accelerated forward pass without VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
    fn forward_avx512_no_vnni(
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        output_layer: &OutputLayer,
    ) -> i32 {
        Self::forward_avx512::<false>(pattern_feature, input_layer, output_layer)
    }

    // AVX2 accelerated forward pass optionally using VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2,avxvnni")]
    #[allow(dead_code)]
    fn forward_avx2_vnni(
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        output_layer: &OutputLayer,
    ) -> i32 {
        Self::forward_avx2::<true>(pattern_feature, input_layer, output_layer)
    }

    // AVX2 accelerated forward pass without VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    #[allow(dead_code)]
    fn forward_avx2_no_vnni(
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        output_layer: &OutputLayer,
    ) -> i32 {
        Self::forward_avx2::<false>(pattern_feature, input_layer, output_layer)
    }

    /// Wrapper for forward_scalar to match the unsafe fn signature.
    #[allow(dead_code)]
    unsafe fn forward_scalar_wrapper(
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        output_layer: &OutputLayer,
    ) -> i32 {
        Self::forward_scalar(pattern_feature, input_layer, output_layer)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
    #[inline]
    fn forward_avx512<const USE_VNNI: bool>(
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        output_layer: &OutputLayer,
    ) -> i32 {
        use crate::eval::util::mm512_dpwssd_epi32;
        use std::arch::x86_64::*;

        const NUM_REGS: usize = PA_OUTPUT_DIMS / 32;
        debug_assert_eq!(NUM_REGS, 4);

        let weights_ptr = input_layer.weights.as_ptr() as *const __m512i;
        let bias_ptr = input_layer.biases.as_ptr() as *const __m512i;

        unsafe {
            let mut acc0 = _mm512_load_si512(bias_ptr);
            let mut acc1 = _mm512_load_si512(bias_ptr.add(1));
            let mut acc2 = _mm512_load_si512(bias_ptr.add(2));
            let mut acc3 = _mm512_load_si512(bias_ptr.add(3));

            macro_rules! accumulate_4 {
                ($base:expr) => {{
                    let idx0 = feature_offset(pattern_feature, $base) * NUM_REGS;
                    let idx1 = feature_offset(pattern_feature, $base + 1) * NUM_REGS;
                    let idx2 = feature_offset(pattern_feature, $base + 2) * NUM_REGS;
                    let idx3 = feature_offset(pattern_feature, $base + 3) * NUM_REGS;

                    let w0 = _mm512_load_si512(weights_ptr.add(idx0));
                    let w1 = _mm512_load_si512(weights_ptr.add(idx1));
                    let w2 = _mm512_load_si512(weights_ptr.add(idx2));
                    let w3 = _mm512_load_si512(weights_ptr.add(idx3));
                    let sum01 = _mm512_add_epi16(w0, w1);
                    let sum23 = _mm512_add_epi16(w2, w3);
                    let sum = _mm512_add_epi16(sum01, sum23);
                    acc0 = _mm512_add_epi16(acc0, sum);

                    let w0 = _mm512_load_si512(weights_ptr.add(idx0 + 1));
                    let w1 = _mm512_load_si512(weights_ptr.add(idx1 + 1));
                    let w2 = _mm512_load_si512(weights_ptr.add(idx2 + 1));
                    let w3 = _mm512_load_si512(weights_ptr.add(idx3 + 1));
                    let sum01 = _mm512_add_epi16(w0, w1);
                    let sum23 = _mm512_add_epi16(w2, w3);
                    let sum = _mm512_add_epi16(sum01, sum23);
                    acc1 = _mm512_add_epi16(acc1, sum);

                    let w0 = _mm512_load_si512(weights_ptr.add(idx0 + 2));
                    let w1 = _mm512_load_si512(weights_ptr.add(idx1 + 2));
                    let w2 = _mm512_load_si512(weights_ptr.add(idx2 + 2));
                    let w3 = _mm512_load_si512(weights_ptr.add(idx3 + 2));
                    let sum01 = _mm512_add_epi16(w0, w1);
                    let sum23 = _mm512_add_epi16(w2, w3);
                    let sum = _mm512_add_epi16(sum01, sum23);
                    acc2 = _mm512_add_epi16(acc2, sum);

                    let w0 = _mm512_load_si512(weights_ptr.add(idx0 + 3));
                    let w1 = _mm512_load_si512(weights_ptr.add(idx1 + 3));
                    let w2 = _mm512_load_si512(weights_ptr.add(idx2 + 3));
                    let w3 = _mm512_load_si512(weights_ptr.add(idx3 + 3));
                    let sum01 = _mm512_add_epi16(w0, w1);
                    let sum23 = _mm512_add_epi16(w2, w3);
                    let sum = _mm512_add_epi16(sum01, sum23);
                    acc3 = _mm512_add_epi16(acc3, sum);
                }};
            }

            accumulate_4!(0);
            accumulate_4!(4);
            accumulate_4!(8);
            accumulate_4!(12);
            accumulate_4!(16);
            accumulate_4!(20);

            let zero = _mm512_setzero_si512();
            let one = _mm512_set1_epi16(1023);

            let v0 = _mm512_min_epi16(_mm512_max_epi16(acc0, zero), one);
            let v1 = _mm512_min_epi16(_mm512_max_epi16(acc1, zero), one);
            let v2 = _mm512_min_epi16(_mm512_max_epi16(acc2, zero), one);
            let v3 = _mm512_min_epi16(_mm512_max_epi16(acc3, zero), one);

            const SHIFT: u32 = 6;
            let act0 = _mm512_mulhi_epu16(_mm512_slli_epi16(v0, SHIFT), v0);
            let act1 = _mm512_mulhi_epu16(_mm512_slli_epi16(v1, SHIFT), v1);
            let act2 = _mm512_mulhi_epu16(_mm512_slli_epi16(v2, SHIFT), v2);
            let act3 = _mm512_mulhi_epu16(_mm512_slli_epi16(v3, SHIFT), v3);

            let out_w_ptr = output_layer.weights.as_ptr() as *const __m512i;
            let ow0 = _mm512_load_si512(out_w_ptr);
            let ow1 = _mm512_load_si512(out_w_ptr.add(1));
            let ow2 = _mm512_load_si512(out_w_ptr.add(2));
            let ow3 = _mm512_load_si512(out_w_ptr.add(3));

            let out0 = mm512_dpwssd_epi32::<USE_VNNI>(_mm512_setzero_si512(), act0, ow0);
            let out1 = mm512_dpwssd_epi32::<USE_VNNI>(_mm512_setzero_si512(), act1, ow1);
            let out2 = mm512_dpwssd_epi32::<USE_VNNI>(_mm512_setzero_si512(), act2, ow2);
            let out3 = mm512_dpwssd_epi32::<USE_VNNI>(_mm512_setzero_si512(), act3, ow3);

            let combined =
                _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(out0, out1), out2), out3);
            _mm512_reduce_add_epi32(combined)
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    #[inline]
    fn forward_avx2<const USE_VNNI: bool>(
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        output_layer: &OutputLayer,
    ) -> i32 {
        use crate::eval::util::mm256_dpwssd_epi32;
        use std::arch::x86_64::*;

        const NUM_REGS: usize = PA_OUTPUT_DIMS / 16;
        debug_assert_eq!(NUM_REGS, 8);

        let weights_ptr = input_layer.weights.as_ptr() as *const __m256i;
        let bias_ptr = input_layer.biases.as_ptr() as *const __m256i;

        unsafe {
            use crate::eval::util::m256_hadd;

            let mut acc0 = _mm256_load_si256(bias_ptr);
            let mut acc1 = _mm256_load_si256(bias_ptr.add(1));
            let mut acc2 = _mm256_load_si256(bias_ptr.add(2));
            let mut acc3 = _mm256_load_si256(bias_ptr.add(3));
            let mut acc4 = _mm256_load_si256(bias_ptr.add(4));
            let mut acc5 = _mm256_load_si256(bias_ptr.add(5));
            let mut acc6 = _mm256_load_si256(bias_ptr.add(6));
            let mut acc7 = _mm256_load_si256(bias_ptr.add(7));

            macro_rules! accumulate_4 {
                ($base:expr) => {{
                    let idx0 = feature_offset(pattern_feature, $base) * NUM_REGS;
                    let idx1 = feature_offset(pattern_feature, $base + 1) * NUM_REGS;
                    let idx2 = feature_offset(pattern_feature, $base + 2) * NUM_REGS;
                    let idx3 = feature_offset(pattern_feature, $base + 3) * NUM_REGS;

                    macro_rules! accumulate_reg {
                        ($j:expr, $acc:ident) => {{
                            let w0 = _mm256_load_si256(weights_ptr.add(idx0 + $j));
                            let w1 = _mm256_load_si256(weights_ptr.add(idx1 + $j));
                            let w2 = _mm256_load_si256(weights_ptr.add(idx2 + $j));
                            let w3 = _mm256_load_si256(weights_ptr.add(idx3 + $j));
                            let sum01 = _mm256_add_epi16(w0, w1);
                            let sum23 = _mm256_add_epi16(w2, w3);
                            let sum = _mm256_add_epi16(sum01, sum23);
                            $acc = _mm256_add_epi16($acc, sum);
                        }};
                    }

                    accumulate_reg!(0, acc0);
                    accumulate_reg!(1, acc1);
                    accumulate_reg!(2, acc2);
                    accumulate_reg!(3, acc3);
                    accumulate_reg!(4, acc4);
                    accumulate_reg!(5, acc5);
                    accumulate_reg!(6, acc6);
                    accumulate_reg!(7, acc7);
                }};
            }

            accumulate_4!(0);
            accumulate_4!(4);
            accumulate_4!(8);
            accumulate_4!(12);
            accumulate_4!(16);
            accumulate_4!(20);

            let zero = _mm256_setzero_si256();
            let one = _mm256_set1_epi16(1023);

            let v0 = _mm256_min_epi16(_mm256_max_epi16(acc0, zero), one);
            let v1 = _mm256_min_epi16(_mm256_max_epi16(acc1, zero), one);
            let v2 = _mm256_min_epi16(_mm256_max_epi16(acc2, zero), one);
            let v3 = _mm256_min_epi16(_mm256_max_epi16(acc3, zero), one);
            let v4 = _mm256_min_epi16(_mm256_max_epi16(acc4, zero), one);
            let v5 = _mm256_min_epi16(_mm256_max_epi16(acc5, zero), one);
            let v6 = _mm256_min_epi16(_mm256_max_epi16(acc6, zero), one);
            let v7 = _mm256_min_epi16(_mm256_max_epi16(acc7, zero), one);

            const SHIFT: i32 = 6;
            let act0 = _mm256_mulhi_epu16(_mm256_slli_epi16(v0, SHIFT), v0);
            let act1 = _mm256_mulhi_epu16(_mm256_slli_epi16(v1, SHIFT), v1);
            let act2 = _mm256_mulhi_epu16(_mm256_slli_epi16(v2, SHIFT), v2);
            let act3 = _mm256_mulhi_epu16(_mm256_slli_epi16(v3, SHIFT), v3);
            let act4 = _mm256_mulhi_epu16(_mm256_slli_epi16(v4, SHIFT), v4);
            let act5 = _mm256_mulhi_epu16(_mm256_slli_epi16(v5, SHIFT), v5);
            let act6 = _mm256_mulhi_epu16(_mm256_slli_epi16(v6, SHIFT), v6);
            let act7 = _mm256_mulhi_epu16(_mm256_slli_epi16(v7, SHIFT), v7);

            let out_w_ptr = output_layer.weights.as_ptr() as *const __m256i;

            let out0 = mm256_dpwssd_epi32::<USE_VNNI>(
                _mm256_setzero_si256(),
                act0,
                _mm256_load_si256(out_w_ptr),
            );
            let out1 = mm256_dpwssd_epi32::<USE_VNNI>(
                _mm256_setzero_si256(),
                act1,
                _mm256_load_si256(out_w_ptr.add(1)),
            );
            let out2 = mm256_dpwssd_epi32::<USE_VNNI>(
                _mm256_setzero_si256(),
                act2,
                _mm256_load_si256(out_w_ptr.add(2)),
            );
            let out3 = mm256_dpwssd_epi32::<USE_VNNI>(
                _mm256_setzero_si256(),
                act3,
                _mm256_load_si256(out_w_ptr.add(3)),
            );
            let out4 = mm256_dpwssd_epi32::<USE_VNNI>(
                _mm256_setzero_si256(),
                act4,
                _mm256_load_si256(out_w_ptr.add(4)),
            );
            let out5 = mm256_dpwssd_epi32::<USE_VNNI>(
                _mm256_setzero_si256(),
                act5,
                _mm256_load_si256(out_w_ptr.add(5)),
            );
            let out6 = mm256_dpwssd_epi32::<USE_VNNI>(
                _mm256_setzero_si256(),
                act6,
                _mm256_load_si256(out_w_ptr.add(6)),
            );
            let out7 = mm256_dpwssd_epi32::<USE_VNNI>(
                _mm256_setzero_si256(),
                act7,
                _mm256_load_si256(out_w_ptr.add(7)),
            );

            let sum01 = _mm256_add_epi32(out0, out1);
            let sum23 = _mm256_add_epi32(out2, out3);
            let sum45 = _mm256_add_epi32(out4, out5);
            let sum67 = _mm256_add_epi32(out6, out7);
            let combined = _mm256_add_epi32(
                _mm256_add_epi32(sum01, sum23),
                _mm256_add_epi32(sum45, sum67),
            );

            m256_hadd(combined)
        }
    }

    fn forward_scalar(
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        output_layer: &OutputLayer,
    ) -> i32 {
        let mut acc = [0i32; PA_OUTPUT_DIMS];

        for (dst, &bias) in acc.iter_mut().zip(input_layer.biases.iter()) {
            *dst = bias as i32;
        }

        let weights = input_layer.weights.as_slice();
        for feature_idx in 0..NUM_FEATURES {
            let offset = feature_offset(pattern_feature, feature_idx);
            let row = &weights[offset * PA_OUTPUT_DIMS..(offset + 1) * PA_OUTPUT_DIMS];
            for (dst, &w) in acc.iter_mut().zip(row.iter()) {
                *dst += w as i32;
            }
        }

        let mut output = 0i32;
        for (value, &weight) in acc.iter().zip(output_layer.weights.iter()) {
            if *value > 0 {
                let clamped = (*value).min(1023);
                let activation = (clamped * clamped) >> 10;
                output += activation * (weight as i32);
            }
        }

        output
    }
}

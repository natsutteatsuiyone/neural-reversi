use std::io::{self, Read};

use byteorder::{LittleEndian, ReadBytesExt};
use reversi_core::{
    eval::pattern_feature::{
        INPUT_FEATURE_DIMS, NUM_FEATURES, PATTERN_FEATURE_OFFSETS, PatternFeature,
    },
    types::ScaledScore,
};

const NN_DIMS: usize = 256;
const HIDDEN_DIMS: usize = 16;
const OUTPUT_DIMS: usize = HIDDEN_DIMS + NN_DIMS;
const NUM_LAYER_STACKS: usize = 60;
const HIDDEN_WEIGHT_SCALE_BITS: u32 = 10;
const OUTPUT_WEIGHT_SCALE_BITS: u32 = 7;
const INPUT_QUANTIZED_ONE: i32 = 255;
const HIDDEN_QUANTIZED_ONE: i32 = 1023;
const INPUT_ACTIVATION_SCALE_BITS: u32 = 8;
const HIDDEN_ACTIVATION_SCALE_BITS: u32 = 10;

type L1Layer = HiddenLayer<NN_DIMS, { NN_DIMS * HIDDEN_DIMS }>;
type L2Layer = HiddenLayer<HIDDEN_DIMS, { HIDDEN_DIMS * HIDDEN_DIMS }>;

/// True when hidden-layer weights are stored in the blocked layout produced by
/// `HiddenLayer::interleave_blocked`. The load-time permute and the
/// `forward_simd` access pattern both key on this one constant so the two
/// sides cannot drift apart.
const BLOCKED_WEIGHT_LAYOUT: bool = cfg!(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    not(target_feature = "relaxed-simd")
));

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

        let weights = if BLOCKED_WEIGHT_LAYOUT {
            Self::interleave_blocked(&weights)
        } else {
            weights
        };

        Ok(Self {
            biases,
            weights: AlignedI16Array(weights),
        })
    }

    /// Permutes row-major weights into a blocked layout: for each group of 8
    /// output rows, the 8 rows' v128 chunks are interleaved chunk-major, so
    /// `forward_simd` reads one contiguous stream with a single induction
    /// pointer instead of 8 strided row pointers. Applied only when
    /// [`BLOCKED_WEIGHT_LAYOUT`] is true; see `forward_simd`.
    fn interleave_blocked(weights: &[i16; WEIGHTS_LEN]) -> [i16; WEIGHTS_LEN] {
        let chunks = IN_DIMS / 8;
        let mut out = [0i16; WEIGHTS_LEN];
        for block in 0..HIDDEN_DIMS / 8 {
            for chunk in 0..chunks {
                for r in 0..8 {
                    let src = (block * 8 + r) * IN_DIMS + chunk * 8;
                    let dst = ((block * chunks + chunk) * 8 + r) * 8;
                    out[dst..dst + 8].copy_from_slice(&weights[src..src + 8]);
                }
            }
        }
        out
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

            let shifted = acc >> HIDDEN_WEIGHT_SCALE_BITS;
            let clamped = shifted.clamp(0, HIDDEN_QUANTIZED_ONE as i64);
            *value = ((clamped * clamped) >> HIDDEN_ACTIVATION_SCALE_BITS) as i16;
        }

        output
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    fn forward_simd(&self, input: &[i16; IN_DIMS]) -> [i16; HIDDEN_DIMS] {
        use std::arch::wasm32::*;

        debug_assert_eq!(IN_DIMS % 8, 0);
        debug_assert_eq!(HIDDEN_DIMS % 8, 0);

        let mut output = [0i16; HIDDEN_DIMS];
        unsafe {
            let input_ptr = input.as_ptr() as *const v128;
            let weights_ptr = self.weights.0.as_ptr() as *const v128;
            let chunks = IN_DIMS / 8;

            let zero = i32x4_splat(0);
            let qone = i32x4_splat(HIDDEN_QUANTIZED_ONE);
            let out_ptr = output.as_mut_ptr() as *mut v128;

            // Four outputs share each activation load and keep their dot
            // products in i32x4 accumulators; a shuffle transpose then yields
            // all four horizontal sums at once. Lane sums stay within i32:
            // |dot| <= IN_DIMS * 1022 * 32767 < i32::MAX for both layer shapes
            // (1022 = max screlu output, HIDDEN_QUANTIZED_ONE^2 >>
            // HIDDEN_ACTIVATION_SCALE_BITS), and trained biases are orders of
            // magnitude below the remaining headroom.
            for half in 0..HIDDEN_DIMS / 8 {
                let out_base = half * 8;

                let mut a0 = zero;
                let mut a1 = zero;
                let mut a2 = zero;
                let mut a3 = zero;
                let mut a4 = zero;
                let mut a5 = zero;
                let mut a6 = zero;
                let mut a7 = zero;

                // Engine-tuned loop shape (see the input-layer accumulate loop):
                // V8 (relaxed-simd build) is fastest with row-major weights and a
                // plain loop; JSC (simd128 build) gains ~3% from the blocked
                // weight layout combined with a 2x unroll. The branch is folded
                // at compile time.
                if !BLOCKED_WEIGHT_LAYOUT {
                    let row0 = weights_ptr.add(out_base * chunks);
                    let row1 = weights_ptr.add((out_base + 1) * chunks);
                    let row2 = weights_ptr.add((out_base + 2) * chunks);
                    let row3 = weights_ptr.add((out_base + 3) * chunks);
                    let row4 = weights_ptr.add((out_base + 4) * chunks);
                    let row5 = weights_ptr.add((out_base + 5) * chunks);
                    let row6 = weights_ptr.add((out_base + 6) * chunks);
                    let row7 = weights_ptr.add((out_base + 7) * chunks);

                    for chunk in 0..chunks {
                        let act = v128_load(input_ptr.add(chunk));
                        a0 = i32x4_add(a0, i32x4_dot_i16x8(act, v128_load(row0.add(chunk))));
                        a1 = i32x4_add(a1, i32x4_dot_i16x8(act, v128_load(row1.add(chunk))));
                        a2 = i32x4_add(a2, i32x4_dot_i16x8(act, v128_load(row2.add(chunk))));
                        a3 = i32x4_add(a3, i32x4_dot_i16x8(act, v128_load(row3.add(chunk))));
                        a4 = i32x4_add(a4, i32x4_dot_i16x8(act, v128_load(row4.add(chunk))));
                        a5 = i32x4_add(a5, i32x4_dot_i16x8(act, v128_load(row5.add(chunk))));
                        a6 = i32x4_add(a6, i32x4_dot_i16x8(act, v128_load(row6.add(chunk))));
                        a7 = i32x4_add(a7, i32x4_dot_i16x8(act, v128_load(row7.add(chunk))));
                    }
                } else {
                    debug_assert_eq!(chunks % 2, 0);
                    let block_ptr = weights_ptr.add(out_base * chunks);
                    let mut chunk = 0;
                    while chunk < chunks {
                        let act0 = v128_load(input_ptr.add(chunk));
                        let act1 = v128_load(input_ptr.add(chunk + 1));
                        let w = block_ptr.add(chunk * 8);
                        a0 = i32x4_add(a0, i32x4_dot_i16x8(act0, v128_load(w)));
                        a1 = i32x4_add(a1, i32x4_dot_i16x8(act0, v128_load(w.add(1))));
                        a2 = i32x4_add(a2, i32x4_dot_i16x8(act0, v128_load(w.add(2))));
                        a3 = i32x4_add(a3, i32x4_dot_i16x8(act0, v128_load(w.add(3))));
                        a4 = i32x4_add(a4, i32x4_dot_i16x8(act0, v128_load(w.add(4))));
                        a5 = i32x4_add(a5, i32x4_dot_i16x8(act0, v128_load(w.add(5))));
                        a6 = i32x4_add(a6, i32x4_dot_i16x8(act0, v128_load(w.add(6))));
                        a7 = i32x4_add(a7, i32x4_dot_i16x8(act0, v128_load(w.add(7))));
                        a0 = i32x4_add(a0, i32x4_dot_i16x8(act1, v128_load(w.add(8))));
                        a1 = i32x4_add(a1, i32x4_dot_i16x8(act1, v128_load(w.add(9))));
                        a2 = i32x4_add(a2, i32x4_dot_i16x8(act1, v128_load(w.add(10))));
                        a3 = i32x4_add(a3, i32x4_dot_i16x8(act1, v128_load(w.add(11))));
                        a4 = i32x4_add(a4, i32x4_dot_i16x8(act1, v128_load(w.add(12))));
                        a5 = i32x4_add(a5, i32x4_dot_i16x8(act1, v128_load(w.add(13))));
                        a6 = i32x4_add(a6, i32x4_dot_i16x8(act1, v128_load(w.add(14))));
                        a7 = i32x4_add(a7, i32x4_dot_i16x8(act1, v128_load(w.add(15))));
                        chunk += 2;
                    }
                }

                macro_rules! hsum_transpose {
                    ($b0:expr, $b1:expr, $b2:expr, $b3:expr) => {{
                        let p = i32x4_add(
                            i32x4_shuffle::<0, 2, 4, 6>($b0, $b1),
                            i32x4_shuffle::<1, 3, 5, 7>($b0, $b1),
                        );
                        let r = i32x4_add(
                            i32x4_shuffle::<0, 2, 4, 6>($b2, $b3),
                            i32x4_shuffle::<1, 3, 5, 7>($b2, $b3),
                        );
                        i32x4_add(
                            i32x4_shuffle::<0, 2, 4, 6>(p, r),
                            i32x4_shuffle::<1, 3, 5, 7>(p, r),
                        )
                    }};
                }

                macro_rules! activate {
                    ($sums:expr, $bias_idx:expr) => {{
                        let bias = v128_load(self.biases.as_ptr().add($bias_idx) as *const v128);
                        let shifted = i32x4_shr(i32x4_add($sums, bias), HIDDEN_WEIGHT_SCALE_BITS);
                        let clamped = i32x4_min(i32x4_max(shifted, zero), qone);
                        let sq = i32x4_mul(clamped, clamped);
                        i32x4_shr(sq, HIDDEN_ACTIVATION_SCALE_BITS)
                    }};
                }

                let lo = activate!(hsum_transpose!(a0, a1, a2, a3), out_base);
                let hi = activate!(hsum_transpose!(a4, a5, a6, a7), out_base + 4);
                v128_store(out_ptr.add(half), i16x8_narrow_i32x4(lo, hi));
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
        let score: i64;

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            score = self.forward_simd(pattern_feature, &self.input_layer, layer_stack);
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            score = self.forward_scalar(pattern_feature, &self.input_layer, layer_stack);
        }

        let raw_score =
            (score >> OUTPUT_WEIGHT_SCALE_BITS).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        let score = ScaledScore::from_raw(raw_score);
        score.clamp(ScaledScore::MIN + 1, ScaledScore::MAX - 1)
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    fn forward_simd(
        &self,
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        layer_stack: &LayerStack,
    ) -> i64 {
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
            let one = i16x8_splat(INPUT_QUANTIZED_ONE as i16);
            let mut input_activation = std::mem::MaybeUninit::<AlignedI16Array<NN_DIMS>>::uninit();
            let input_activation_ptr =
                std::ptr::addr_of_mut!((*input_activation.as_mut_ptr()).0) as *mut i16;
            let out_w = layer_stack.output.weights.0.as_ptr().add(HIDDEN_DIMS) as *const v128;
            let mut skip0 = i32x4_splat(0);
            let mut skip1 = i32x4_splat(0);
            let mut skip2 = i32x4_splat(0);
            let mut skip3 = i32x4_splat(0);

            macro_rules! activate_store {
                ($idx:expr, $acc:expr, $skip_sum:expr) => {{
                    let relu = i16x8_max($acc, zero);
                    let clamped = i16x8_min(relu, one);
                    let sq = u16x8_mul(clamped, clamped);
                    let activation = u16x8_shr(sq, INPUT_ACTIVATION_SCALE_BITS);
                    v128_store(input_activation_ptr.add($idx * 8) as *mut v128, activation);

                    let skip_weight = v128_load(out_w.add($idx));
                    $skip_sum = i32x4_add($skip_sum, i32x4_dot_i16x8(activation, skip_weight));
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

                    // Each feature pair's i8->i16 widening and addition is fused via byte
                    // interleave + pairwise widening add: the pairwise horizontal sum
                    // over the interleaved rows yields `w_a[col] + w_b[col]`. Partial
                    // sums (|w| <= 127) fit i16 exactly, so the result is bit-identical
                    // to a plain extend+add path. The loop shape is engine-tuned: V8
                    // (which loads the relaxed-simd build) is fastest accumulating one
                    // pair at a time, while JSC (which falls back to the simd128 build)
                    // gains ~5% from combining two pairs before each accumulator add.
                    #[cfg(target_feature = "relaxed-simd")]
                    {
                        let mut pair = 0;
                        while pair < NUM_FEATURES {
                            let row_a = row_ptrs[pair];
                            let row_b = row_ptrs[pair + 1];

                            macro_rules! pair_acc {
                                ($k:expr, $lo:ident, $hi:ident) => {{
                                    let wa = v128_load(row_a.add(c + $k));
                                    let wb = v128_load(row_b.add(c + $k));
                                    let lo = interleave_lo_i8(wa, wb);
                                    let hi = interleave_hi_i8(wa, wb);
                                    $lo = i16x8_add($lo, pairwise_widen_add(lo));
                                    $hi = i16x8_add($hi, pairwise_widen_add(hi));
                                }};
                            }

                            pair_acc!(0, a0, a1);
                            pair_acc!(1, a2, a3);
                            pair_acc!(2, a4, a5);
                            pair_acc!(3, a6, a7);
                            pair += 2;
                        }
                    }

                    #[cfg(not(target_feature = "relaxed-simd"))]
                    {
                        let mut quad = 0;
                        while quad < NUM_FEATURES {
                            let row_a = row_ptrs[quad];
                            let row_b = row_ptrs[quad + 1];
                            let row_c = row_ptrs[quad + 2];
                            let row_d = row_ptrs[quad + 3];

                            macro_rules! quad_acc {
                                ($k:expr, $lo:ident, $hi:ident) => {{
                                    let wa = v128_load(row_a.add(c + $k));
                                    let wb = v128_load(row_b.add(c + $k));
                                    let wc = v128_load(row_c.add(c + $k));
                                    let wd = v128_load(row_d.add(c + $k));
                                    let lo = i16x8_add(
                                        pairwise_widen_add(interleave_lo_i8(wa, wb)),
                                        pairwise_widen_add(interleave_lo_i8(wc, wd)),
                                    );
                                    let hi = i16x8_add(
                                        pairwise_widen_add(interleave_hi_i8(wa, wb)),
                                        pairwise_widen_add(interleave_hi_i8(wc, wd)),
                                    );
                                    $lo = i16x8_add($lo, lo);
                                    $hi = i16x8_add($hi, hi);
                                }};
                            }

                            quad_acc!(0, a0, a1);
                            quad_acc!(1, a2, a3);
                            quad_acc!(2, a4, a5);
                            quad_acc!(3, a6, a7);
                            quad += 4;
                        }
                    }

                    activate_store!(c * 2, a0, skip0);
                    activate_store!(c * 2 + 1, a1, skip1);
                    activate_store!(c * 2 + 2, a2, skip2);
                    activate_store!(c * 2 + 3, a3, skip3);
                    activate_store!(c * 2 + 4, a4, skip0);
                    activate_store!(c * 2 + 5, a5, skip1);
                    activate_store!(c * 2 + 6, a6, skip2);
                    activate_store!(c * 2 + 7, a7, skip3);
                }};
            }

            process_block!(0);
            process_block!(1);
            process_block!(2);
            process_block!(3);

            let skip_sum = horizontal_sum_i32x4(skip0) as i64
                + horizontal_sum_i32x4(skip1) as i64
                + horizontal_sum_i32x4(skip2) as i64
                + horizontal_sum_i32x4(skip3) as i64;
            let mut output = layer_stack.output.bias as i64 + skip_sum;

            // SAFETY: the 4 blocks above write all 32 i16x8 lanes in NN_DIMS.
            let input_activation = input_activation.assume_init();
            let l1 = layer_stack.l1.forward_simd(&input_activation.0);
            let l2 = layer_stack.l2.forward_simd(&l1);

            debug_assert_eq!(HIDDEN_DIMS % 8, 0);
            let l2_ptr = l2.as_ptr() as *const v128;
            let l2_weights_ptr = layer_stack.output.weights.0.as_ptr() as *const v128;
            let mut hidden_acc = i32x4_splat(0);
            for chunk in 0..HIDDEN_DIMS / 8 {
                let l2_vec = v128_load(l2_ptr.add(chunk));
                let l2_weights = v128_load(l2_weights_ptr.add(chunk));
                hidden_acc = i32x4_add(hidden_acc, i32x4_dot_i16x8(l2_vec, l2_weights));
            }
            output += horizontal_sum_i32x4(hidden_acc) as i64;

            output
        }
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    fn forward_scalar(
        &self,
        pattern_feature: &PatternFeature,
        input_layer: &InputLayer,
        layer_stack: &LayerStack,
    ) -> i64 {
        let mut input_activation = input_layer.biases.0;

        // SAFETY: `input_layer.weights` owns `len` initialized bytes at `ptr`
        // for the lifetime of the layer, and scalar forward only reads them.
        let weights =
            unsafe { std::slice::from_raw_parts(input_layer.weights.ptr, input_layer.weights.len) };
        for feature_idx in 0..NUM_FEATURES {
            let offset = feature_offset(pattern_feature, feature_idx);
            let row = &weights[offset * NN_DIMS..(offset + 1) * NN_DIMS];
            for i in 0..NN_DIMS {
                input_activation[i] += row[i] as i16;
            }
        }

        for value in &mut input_activation {
            let clamped = (*value as i32).clamp(0, INPUT_QUANTIZED_ONE);
            *value = ((clamped * clamped) >> INPUT_ACTIVATION_SCALE_BITS) as i16;
        }

        let input_activation = AlignedI16Array(input_activation);
        let l1 = layer_stack.l1.forward_scalar(&input_activation.0);
        let l2 = layer_stack.l2.forward_scalar(&l1);

        let mut output = layer_stack.output.bias as i64;

        for (idx, &value) in l2.iter().enumerate() {
            output += value as i64 * layer_stack.output.weights.0[idx] as i64;
        }

        for (idx, &value) in input_activation.0.iter().enumerate() {
            output += value as i64 * layer_stack.output.weights.0[HIDDEN_DIMS + idx] as i64;
        }

        output
    }
}

#[inline(always)]
fn feature_offset(pattern_feature: &PatternFeature, idx: usize) -> usize {
    *unsafe { PATTERN_FEATURE_OFFSETS.get_unchecked(idx) }
        + unsafe { pattern_feature.get_unchecked(idx) } as usize
}

/// Interleaves the low 8 bytes of `a` and `b`: `[a0, b0, a1, b1, ..., a7, b7]`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn interleave_lo_i8(
    a: std::arch::wasm32::v128,
    b: std::arch::wasm32::v128,
) -> std::arch::wasm32::v128 {
    use std::arch::wasm32::*;
    i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b)
}

/// Interleaves the high 8 bytes of `a` and `b`: `[a8, b8, a9, b9, ..., a15, b15]`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn interleave_hi_i8(
    a: std::arch::wasm32::v128,
    b: std::arch::wasm32::v128,
) -> std::arch::wasm32::v128 {
    use std::arch::wasm32::*;
    i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(a, b)
}

/// Widens adjacent i8 pairs and adds them: `out[i] = in[2i] + in[2i+1]` as i16.
///
/// The relaxed-simd form is bit-identical here: with `ones` as the i7 operand
/// the dot's partial sums are exactly the pairwise sums, and |w| <= 127 keeps
/// them far from i16 saturation.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn pairwise_widen_add(x: std::arch::wasm32::v128) -> std::arch::wasm32::v128 {
    use std::arch::wasm32::*;

    #[cfg(target_feature = "relaxed-simd")]
    {
        i16x8_relaxed_dot_i8x16_i7x16(x, i8x16_splat(1))
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        i16x8_extadd_pairwise_i8x16(x)
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn horizontal_sum_i32x4(sum: std::arch::wasm32::v128) -> i32 {
    use std::arch::wasm32::*;

    let pair_sums = i32x4_add(sum, i32x4_shuffle::<2, 3, 0, 1>(sum, sum));
    let total = i32x4_add(pair_sums, i32x4_shuffle::<1, 0, 3, 2>(pair_sums, pair_sums));
    i32x4_extract_lane::<0>(total)
}

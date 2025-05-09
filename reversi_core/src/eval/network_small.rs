use std::fs::File;
use std::io::{self, BufReader};

use aligned::{Aligned, A64};

use crate::board::Board;
use crate::constants::{MID_SCORE_MAX, MID_SCORE_MIN};
use crate::eval::constants::HIDDEN_WEIGHT_SCALE_BITS;
use crate::eval::linear_layer::LinearLayer;
use crate::eval::pattern_feature::NUM_PATTERN_FEATURES;
use crate::eval::phase_adaptive_input::PhaseAdaptiveInput;
use crate::eval::relu::clipped_relu;
use crate::misc::ceil_to_multiple;
use crate::search::search_context::SearchContext;
use crate::types::Score;

use super::constants::{INPUT_FEATURE_DIMS, NUM_FEATURES, OUTPUT_WEIGHT_SCALE_BITS, PATTERN_FEATURE_OFFSETS};

const L1_PA_INPUT_DIMS: usize = 64 + 1;
const L1_PA_PADDED_INPUT_DIMS: usize = ceil_to_multiple(L1_PA_INPUT_DIMS, 32);
const L1_PA_OUTPUT_DIMS: usize = 8;
const L1_PA_PADDED_OUTPUT_DIMS: usize = ceil_to_multiple(L1_PA_OUTPUT_DIMS, 32);

const L2_INPUT_DIMS: usize = L1_PA_OUTPUT_DIMS;
const L2_PADDED_INPUT_DIMS: usize = ceil_to_multiple(L2_INPUT_DIMS, 32);
const L2_OUTPUT_DIMS: usize = 32;
const L2_PADDED_OUTPUT_DIMS: usize = ceil_to_multiple(L2_OUTPUT_DIMS, 32);

const LO_INPUT_DIMS: usize = L2_OUTPUT_DIMS;

const NUM_PHASE_ADAPTIVE_INPUT: usize = 6;
const NUM_LAYER_STACKS: usize = 60;

struct LayerStack {
    pub l1_pa: LinearLayer<
        L1_PA_INPUT_DIMS,
        L1_PA_OUTPUT_DIMS,
        L1_PA_PADDED_INPUT_DIMS,
        L1_PA_PADDED_OUTPUT_DIMS,
    >,
    pub l2: LinearLayer<
        L2_INPUT_DIMS,
        L2_OUTPUT_DIMS,
        L2_PADDED_INPUT_DIMS,
        L2_PADDED_OUTPUT_DIMS,
    >,
    pub lo: LinearLayer<
        LO_INPUT_DIMS,
        1,
        { ceil_to_multiple(LO_INPUT_DIMS, 32) },
        { ceil_to_multiple(1, 32) },
    >,
}

pub struct NetworkSmall {
    pa_inputs: Vec<PhaseAdaptiveInput<
        INPUT_FEATURE_DIMS,
        { L1_PA_INPUT_DIMS - 1 },
    > >,
    layer_stacks: Vec<LayerStack>,
}

impl NetworkSmall {
    pub fn new(file_path: &str) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut decoder = zstd::stream::read::Decoder::new(reader)?;

        let mut pa_inputs = Vec::with_capacity(NUM_PHASE_ADAPTIVE_INPUT);
        for _ in 0..NUM_PHASE_ADAPTIVE_INPUT {
            let pa_input = PhaseAdaptiveInput::<
                INPUT_FEATURE_DIMS,
                { L1_PA_INPUT_DIMS - 1 },
            >::load(&mut decoder)?;
            pa_inputs.push(pa_input);
        }
        let mut layer_stacks = Vec::with_capacity(NUM_LAYER_STACKS);
        for _ in 0..NUM_LAYER_STACKS {
            let l1_pa = LinearLayer::load(&mut decoder)?;
            let l2 = LinearLayer::load(&mut decoder)?;
            let lo = LinearLayer::load(&mut decoder)?;
            layer_stacks.push(LayerStack {
                l1_pa,
                l2,
                lo,
            });
        }

        Ok(NetworkSmall {
            pa_inputs,
            layer_stacks,
        })
    }

    pub fn evaluate(&self, ctx: &SearchContext, board: &Board) -> Score {
        let ply = ctx.ply();
        let feature_indices = if ctx.player == 0 {
            &ctx.feature_set.p_features[ply]
        } else {
            &ctx.feature_set.o_features[ply]
        };
        let mobility = board.get_moves().count_ones();
        let mut indicies = [0usize; NUM_FEATURES];
        for i in 0..NUM_PATTERN_FEATURES {
            indicies[i] = unsafe { feature_indices.v1 }[i] as usize + PATTERN_FEATURE_OFFSETS[i];
        }

        let score = self.forward(&indicies, mobility as u8, ply);
        score.clamp(MID_SCORE_MIN + 1, MID_SCORE_MAX - 1)
    }

    fn forward(&self, feature_indices: &[usize], mobility: u8, ply: usize) -> Score {
        let li_pa_out = self.forward_input_pa(feature_indices, mobility, ply);

        let ls = &self.layer_stacks[ply / (60 / NUM_LAYER_STACKS)];
        let l1_out = self.forward_l1(ls, li_pa_out.as_slice());
        let l2_out = self.forward_l2(ls, l1_out.as_slice());
        self.forward_output(ls, l2_out.as_slice())
    }

    #[inline]
    fn forward_input_pa(
        &self,
        feature_indices: &[usize],
        mobility: u8,
        ply: usize,
    ) -> Aligned<A64, [u8; L1_PA_PADDED_INPUT_DIMS]> {
        let mut out = Aligned([0; L1_PA_PADDED_INPUT_DIMS]);
        let pa_input = &self.pa_inputs[ply / (60 / NUM_PHASE_ADAPTIVE_INPUT)];
        pa_input.forward_leaky_relu(feature_indices, out.as_mut_slice());
        out[L1_PA_INPUT_DIMS - 1] = mobility * 3;
        out
    }

    #[inline]
    fn forward_l1(
        &self,
        ls: &LayerStack,
        input_pa: &[u8],
    ) -> Aligned<A64, [u8; L2_PADDED_INPUT_DIMS]> {
        let mut l1_out: Aligned<A64, [i32; L1_PA_PADDED_OUTPUT_DIMS]> =
            Aligned([0; L1_PA_PADDED_OUTPUT_DIMS]);
        ls.l1_pa.forward(input_pa, l1_out.as_mut_slice());

        const L1_OUTPUT_DIMS: usize = ceil_to_multiple(L1_PA_OUTPUT_DIMS, 32);
        let mut l1_relu_out: Aligned<A64, [u8; L1_OUTPUT_DIMS]> = Aligned([0; L1_OUTPUT_DIMS]);
        clipped_relu::<HIDDEN_WEIGHT_SCALE_BITS>(l1_out.as_slice(), l1_relu_out.as_mut_slice());

        l1_relu_out
    }

    #[inline]
    fn forward_l2(
        &self,
        ls: &LayerStack,
        input: &[u8],
    ) -> Aligned<A64, [u8; L2_PADDED_OUTPUT_DIMS]> {
        let mut l2_out: Aligned<A64, [i32; L2_PADDED_OUTPUT_DIMS]> =
            Aligned([0; L2_PADDED_OUTPUT_DIMS]);
        ls.l2.forward(input, l2_out.as_mut_slice());

        let mut l2_act: Aligned<A64, [u8; L2_PADDED_OUTPUT_DIMS]> =
            Aligned([0; L2_PADDED_OUTPUT_DIMS]);
        clipped_relu::<HIDDEN_WEIGHT_SCALE_BITS>(l2_out.as_slice(), l2_act.as_mut_slice());

        l2_act
    }

    #[inline]
    fn forward_output(&self, ls: &LayerStack, input: &[u8]) -> Score {
        let mut out: Aligned<A64, [i32; 1]> = Aligned([0; 1]);
        ls.lo.forward(input, out.as_mut_slice());

        out[0] >> OUTPUT_WEIGHT_SCALE_BITS
    }
}

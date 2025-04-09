pub mod constants;
mod eval_cache;
mod layer_stack;
mod linear_layer;
pub mod pattern_feature;
mod phase_adaptive_input;
mod relu;
mod base_input;

use aligned::{Aligned, A64};
use pattern_feature::NUM_PATTERN_FEATURES;
use std::fs::File;
use std::io::{self, BufReader};

use constants::*;
use eval_cache::EvalCache;
use layer_stack::LayerStack;
use linear_layer::LinearLayer;
use phase_adaptive_input::PhaseAdaptiveInput;
use relu::{clipped_relu, sqr_clipped_relu};
use base_input::BaseInput;

use crate::board::Board;
use crate::constants::{MID_SCORE_MAX, MID_SCORE_MIN};
use crate::misc::ceil_to_multiple;
use crate::search::search_context::SearchContext;
use crate::types::Score;

pub struct Eval {
    base_input: BaseInput,
    pa_inputs: Vec<PhaseAdaptiveInput>,
    layer_stacks: Vec<LayerStack>,
    pub cache: EvalCache,
}

impl Eval {
    pub fn new(file_path: &str) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut decoder = zstd::stream::read::Decoder::new(reader)?;

        let base_input = BaseInput::load(&mut decoder)?;
        let mut pa_inputs = Vec::with_capacity(NUM_PHASE_ADAPTIVE_INPUT);
        for _ in 0..NUM_PHASE_ADAPTIVE_INPUT {
            let pa_input = PhaseAdaptiveInput::load(&mut decoder)?;
            pa_inputs.push(pa_input);
        }
        let mut layer_stacks = Vec::with_capacity(NUM_LAYER_STACKS);
        for _ in 0..NUM_LAYER_STACKS {
            let l1_base = LinearLayer::load(&mut decoder)?;
            let l1_pa = LinearLayer::load(&mut decoder)?;
            let l2 = LinearLayer::load(&mut decoder)?;
            let lo = LinearLayer::load(&mut decoder)?;
            layer_stacks.push(LayerStack {
                l1_base,
                l1_pa,
                l2,
                lo,
            });
        }

        Ok(Eval {
            base_input,
            pa_inputs,
            layer_stacks,
            cache: EvalCache::new(17),
        })
    }

    /// Evaluate the current position.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The search context.
    /// * `board` - The current board.
    ///
    /// # Returns
    ///
    /// The evaluation score of the current position.
    pub fn evaluate(&self, ctx: &SearchContext, board: &Board) -> Score {
        let key = board.hash();
        if let Some(score_cache) = self.cache.probe(key) {
            return score_cache;
        }

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

        let mut score = self.forward(
            &indicies,
            mobility as u8,
            ply,
        );

        score = score.clamp(MID_SCORE_MIN + 1, MID_SCORE_MAX - 1);

        self.cache.store(key, score);
        score
    }

    fn forward(&self, feature_indices: &[usize], mobility: u8, ply: usize) -> Score {
        let li_base_out = self.forward_input_base(feature_indices, mobility);
        let li_pa_out = self.forward_input_pa(feature_indices, mobility, ply);

        let ls = &self.layer_stacks[ply / (60 / NUM_LAYER_STACKS)];
        let l1_out = self.forward_l1(ls, li_base_out.as_slice(), li_pa_out.as_slice());
        let l2_out = self.forward_l2(ls, l1_out.as_slice());
        self.forward_output(ls, l2_out.as_slice())
    }

    #[inline]
    fn forward_input_base(
        &self,
        feature_indices:&[usize],
        mobility: u8,
    ) -> Aligned<A64, [u8; L1_BASE_PADDED_INPUT_DIMS]> {
        let mut out = Aligned([0; L1_BASE_PADDED_INPUT_DIMS]);
        self.base_input.forward(feature_indices, &mut out[0..BASE_INPUT_OUTPUT_DIMS]);
        out[L1_BASE_INPUT_DIMS - 1] = mobility * 3;
        out
    }

    #[inline]
    fn forward_input_pa(
        &self,
        feature_indices: &[usize],
        mobility: u8,
        ply: usize,
    ) -> Aligned<A64, [u8; L1_PA_PADDED_INPUT_DIMS]> {
        let mut out = Aligned([0; L1_PA_PADDED_INPUT_DIMS]);
        self.pa_inputs[ply / (60 / NUM_PHASE_ADAPTIVE_INPUT)]
            .forward(feature_indices, out.as_mut_slice());
        out[L1_PA_INPUT_DIMS - 1] = mobility * 3;
        out
    }

    #[inline]
    fn forward_l1(
        &self,
        ls: &LayerStack,
        input_base: &[u8],
        input_pa: &[u8],
    ) -> Aligned<A64, [u8; L2_PADDED_INPUT_DIMS]> {
        let mut l1_base_out: Aligned<A64, [i32; L1_BASE_PADDED_OUTPUT_DIMS]> = Aligned([0; L1_BASE_PADDED_OUTPUT_DIMS]);
        ls.l1_base.forward(input_base, l1_base_out.as_mut_slice());

        let mut l1_pa_out: Aligned<A64, [i32; L1_PA_PADDED_OUTPUT_DIMS]> = Aligned([0; L1_PA_PADDED_OUTPUT_DIMS]);
        ls.l1_pa.forward(input_pa, l1_pa_out.as_mut_slice());

        let mut l1_out: Aligned<A64, [i32; L2_INPUT_DIMS]> = Aligned([0; L2_INPUT_DIMS]);
        l1_out[0..L1_BASE_OUTPUT_DIMS]
            .copy_from_slice(&l1_base_out[0..L1_BASE_OUTPUT_DIMS]);
        l1_out[L1_BASE_OUTPUT_DIMS..(L1_BASE_OUTPUT_DIMS + L1_PA_OUTPUT_DIMS)]
            .copy_from_slice(&l1_pa_out[0..L1_PA_OUTPUT_DIMS]);

        const L1_OUTPUT_DIMS: usize = ceil_to_multiple(L1_BASE_OUTPUT_DIMS + L1_PA_OUTPUT_DIMS, 32);

        let mut l1_sqr_relu_out: Aligned<A64, [u8; L1_OUTPUT_DIMS]> = Aligned([0; L1_OUTPUT_DIMS]);
        sqr_clipped_relu::<HIDDEN_WEIGHT_SCALE_BITS>(
            l1_out.as_slice(),
            l1_sqr_relu_out.as_mut_slice()
        );

        let mut l1_relu_out: Aligned<A64, [u8; L1_OUTPUT_DIMS]> = Aligned([0; L1_OUTPUT_DIMS]);
        clipped_relu::<HIDDEN_WEIGHT_SCALE_BITS>(
            l1_out.as_slice(),
            l1_relu_out.as_mut_slice()
        );

        let mut out: Aligned<A64, [u8; L2_PADDED_INPUT_DIMS]> = Aligned([0; L2_PADDED_INPUT_DIMS]);
        out[0..(L2_INPUT_DIMS / 2)].copy_from_slice(&l1_sqr_relu_out[0..(L2_INPUT_DIMS / 2)]);
        out[(L2_INPUT_DIMS / 2)..L2_INPUT_DIMS].copy_from_slice(&l1_relu_out[0..(L2_INPUT_DIMS / 2)]);

        out
    }

    #[inline]
    fn forward_l2(
        &self,
        ls: &LayerStack,
        input: &[u8]
    ) -> Aligned<A64, [u8; L2_PADDED_OUTPUT_DIMS]> {
        let mut l2_out: Aligned<A64, [i32; L2_PADDED_OUTPUT_DIMS]> = Aligned([0; L2_PADDED_OUTPUT_DIMS]);
        ls.l2.forward(input, l2_out.as_mut_slice());

        let mut l2_act: Aligned<A64, [u8; L2_PADDED_OUTPUT_DIMS]> = Aligned([0; L2_PADDED_OUTPUT_DIMS]);
        clipped_relu::<HIDDEN_WEIGHT_SCALE_BITS>(
            l2_out.as_slice(),
            l2_act.as_mut_slice()
        );

        l2_act
    }

    #[inline]
    fn forward_output(&self, ls: &LayerStack, input: &[u8]) -> Score {
        let mut out: Aligned<A64, [i32; 1]> = Aligned([0; 1]);
        ls.lo.forward(input, out.as_mut_slice());

        out[0] >> OUTPUT_WEIGHT_SCALE_BITS
    }
}

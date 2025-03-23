pub mod constants;
mod eval_cache;
mod layer_stack;
mod linear_layer;
pub mod pattern_feature;
mod phase_adaptive_input;
mod relu;
mod universal_input;

use aligned::{Aligned, A64};
use std::fs::File;
use std::io::{self, BufReader};

use constants::*;
use eval_cache::EvalCache;
use layer_stack::LayerStack;
use linear_layer::LinearLayer;
use phase_adaptive_input::PhaseAdaptiveInput;
use relu::{clipped_relu, sqr_clipped_relu};
use universal_input::UniversalInput;

use crate::board::Board;
use crate::constants::{MID_SCORE_MAX, MID_SCORE_MIN};
use crate::misc::ceil_to_multiple;
use crate::search::search_context::SearchContext;
use crate::types::Score;

pub struct Eval {
    univ_input: UniversalInput,
    pa_inputs: Vec<PhaseAdaptiveInput>,
    layer_stacks: Vec<LayerStack>,
    pub cache: EvalCache,
}

impl Eval {
    pub fn new(file_path: &str) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut decoder = zstd::stream::read::Decoder::new(reader)?;

        let univ_input = UniversalInput::load(&mut decoder)?;
        let mut pa_inputs = Vec::with_capacity(NUM_PHASE_ADAPTIVE_INPUT);
        for _ in 0..NUM_PHASE_ADAPTIVE_INPUT {
            let pa_input = PhaseAdaptiveInput::load(&mut decoder)?;
            pa_inputs.push(pa_input);
        }
        let mut layer_stacks = Vec::with_capacity(NUM_LAYER_STACKS);
        for _ in 0..NUM_LAYER_STACKS {
            let l1_univ = LinearLayer::load(&mut decoder)?;
            let l1_pa = LinearLayer::load(&mut decoder)?;
            let l2 = LinearLayer::load(&mut decoder)?;
            let lo = LinearLayer::load(&mut decoder)?;
            layer_stacks.push(LayerStack {
                l1_univ,
                l1_pa,
                l2,
                lo,
            });
        }

        Ok(Eval {
            univ_input,
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

        let mut score = self.forward(
            unsafe { &feature_indices.v1[..NUM_FEATURES] },
            mobility as u8,
            ply,
        );

        score = score.clamp(MID_SCORE_MIN + 1, MID_SCORE_MAX - 1);

        self.cache.store(key, score);
        score
    }

    fn forward(&self, feature_indices: &[u16], mobility: u8, ply: usize) -> Score {
        let li_univ_out = self.forward_input_base(feature_indices, mobility);
        let li_pa_out = self.forward_input_ps(feature_indices, ply);

        let ls = &self.layer_stacks[ply / (60 / NUM_LAYER_STACKS)];
        let l1_out = self.forward_l1(ls, li_univ_out.as_slice(), li_pa_out.as_slice());
        let l2_out = self.forward_l2(ls, l1_out.as_slice());
        self.forward_output(ls, l2_out.as_slice())
    }

    #[inline]
    fn forward_input_base(
        &self,
        feature_indices: &[u16],
        mobility: u8,
    ) -> Aligned<A64, [u8; L1_UNIV_PADDED_INPUT_DIMS]> {
        let mut out = Aligned([0; L1_UNIV_PADDED_INPUT_DIMS]);
        self.univ_input.forward(feature_indices, &mut out[0..UNIV_INPUT_OUTPUT_DIMS]);
        out[L1_UNIV_INPUT_DIMS - 1] = mobility * 3;
        out
    }

    #[inline]
    fn forward_input_ps(
        &self,
        feature_indices: &[u16],
        ply: usize,
    ) -> Aligned<A64, [u8; L1_PS_PADDED_INPUT_DIMS]> {
        let mut out = Aligned([0; L1_PS_PADDED_INPUT_DIMS]);
        self.pa_inputs[ply / (60 / NUM_PHASE_ADAPTIVE_INPUT)]
            .forward(feature_indices, out.as_mut_slice());
        out
    }

    #[inline]
    fn forward_l1(
        &self,
        ls: &LayerStack,
        input_univ: &[u8],
        input_pa: &[u8],
    ) -> Aligned<A64, [u8; L2_PADDED_INPUT_DIMS]> {
        let mut l1_univ_out: Aligned<A64, [i32; L1_UNIV_PADDED_OUTPUT_DIMS]> = Aligned([0; L1_UNIV_PADDED_OUTPUT_DIMS]);
        ls.l1_univ.forward(input_univ, l1_univ_out.as_mut_slice());

        let mut l1_pa_out: Aligned<A64, [i32; L1_PS_PADDED_OUTPUT_DIMS]> = Aligned([0; L1_PS_PADDED_OUTPUT_DIMS]);
        ls.l1_pa.forward(input_pa, l1_pa_out.as_mut_slice());

        let mut l1_out: Aligned<A64, [i32; L2_INPUT_DIMS]> = Aligned([0; L2_INPUT_DIMS]);
        l1_out[0..L1_UNIV_OUTPUT_DIMS]
            .copy_from_slice(&l1_univ_out[0..L1_UNIV_OUTPUT_DIMS]);
        l1_out[L1_UNIV_OUTPUT_DIMS..(L1_UNIV_OUTPUT_DIMS + L1_PS_OUTPUT_DIMS)]
            .copy_from_slice(&l1_pa_out[0..L1_PS_OUTPUT_DIMS]);

        const L1_OUTPUT_DIMS: usize = ceil_to_multiple(L1_UNIV_OUTPUT_DIMS + L1_PS_OUTPUT_DIMS, 32);

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

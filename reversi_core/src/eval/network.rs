use std::fs::File;
use std::io::{self, BufReader};

use aligned::{Aligned, A64};

use crate::board::Board;
use crate::constants::{MID_SCORE_MAX, MID_SCORE_MIN};
use crate::eval::base_input::BaseInput;
use crate::eval::constants::*;
use crate::eval::linear_layer::LinearLayer;
use crate::eval::pattern_feature::NUM_PATTERN_FEATURES;
use crate::eval::phase_adaptive_input::PhaseAdaptiveInput;
use crate::eval::relu::{clipped_relu, sqr_clipped_relu};
use crate::misc::ceil_to_multiple;
use crate::search::search_context::SearchContext;
use crate::types::Score;

const BASE_OUTPUT_DIMS: usize = 96;
const PA_OUTPUT_DIMS: usize = 96;

const L1_BASE_INPUT_DIMS: usize = BASE_OUTPUT_DIMS + 1;
const L1_BASE_PADDED_INPUT_DIMS: usize = ceil_to_multiple(L1_BASE_INPUT_DIMS, 32);
const L1_BASE_OUTPUT_DIMS: usize = 8;
const L1_BASE_PADDED_OUTPUT_DIMS: usize = ceil_to_multiple(L1_BASE_OUTPUT_DIMS, 32);

const L1_PA_INPUT_DIMS: usize = PA_OUTPUT_DIMS + 1;
const L1_PA_PADDED_INPUT_DIMS: usize = ceil_to_multiple(L1_PA_INPUT_DIMS, 32);
const L1_PA_OUTPUT_DIMS: usize = 8;
const L1_PA_PADDED_OUTPUT_DIMS: usize = ceil_to_multiple(L1_PA_OUTPUT_DIMS, 32);
const L1_OUTPUT_DIMS: usize = L1_BASE_OUTPUT_DIMS + L1_PA_OUTPUT_DIMS;

const L2_INPUT_DIMS: usize = L1_OUTPUT_DIMS * 2;
const L2_PADDED_INPUT_DIMS: usize = ceil_to_multiple(L2_INPUT_DIMS, 32);
const L2_OUTPUT_DIMS: usize = 64;
const L2_PADDED_OUTPUT_DIMS: usize = ceil_to_multiple(L2_OUTPUT_DIMS, 32);

const LO_INPUT_DIMS: usize = L2_OUTPUT_DIMS;

const NUM_LAYER_STACKS: usize = 60;
const NUM_PHASE_ADAPTIVE_INPUT: usize = 6;

struct LayerStack {
    pub l1_base: LinearLayer<
        L1_BASE_INPUT_DIMS,
        L1_BASE_OUTPUT_DIMS,
        L1_BASE_PADDED_INPUT_DIMS,
        L1_BASE_PADDED_OUTPUT_DIMS,
    >,
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
        L2_PADDED_OUTPUT_DIMS
    >,
    pub lo: LinearLayer<
        LO_INPUT_DIMS,
        1,
        { ceil_to_multiple(LO_INPUT_DIMS, 32) },
        { ceil_to_multiple(1, 32) },
    >,
}

// Thread-local working buffers for network computation
struct NetworkBuffers {
    base_out: Aligned<A64, [u8; L1_BASE_PADDED_INPUT_DIMS]>,
    pa_out: Aligned<A64, [u8; L1_PA_PADDED_INPUT_DIMS]>,
    l1_base_out: Aligned<A64, [i32; L1_BASE_PADDED_OUTPUT_DIMS]>,
    l1_pa_out: Aligned<A64, [i32; L1_PA_PADDED_OUTPUT_DIMS]>,
    l1_li_out: Aligned<A64, [i32; L1_OUTPUT_DIMS]>,
    l1_sqr_relu: Aligned<A64, [u8; L1_OUTPUT_DIMS]>,
    l1_relu: Aligned<A64, [u8; L1_OUTPUT_DIMS]>,
    l1_out: Aligned<A64, [u8; L2_PADDED_INPUT_DIMS]>,
    l2_li_out: Aligned<A64, [i32; L2_PADDED_OUTPUT_DIMS]>,
    l2_out: Aligned<A64, [u8; L2_PADDED_OUTPUT_DIMS]>,
    feature_indices: [usize; NUM_FEATURES],
}

impl NetworkBuffers {
    #[inline]
    fn new() -> Self {
        Self {
            base_out: Aligned([0; L1_BASE_PADDED_INPUT_DIMS]),
            pa_out: Aligned([0; L1_PA_PADDED_INPUT_DIMS]),
            l1_base_out: Aligned([0; L1_BASE_PADDED_OUTPUT_DIMS]),
            l1_pa_out: Aligned([0; L1_PA_PADDED_OUTPUT_DIMS]),
            l1_li_out: Aligned([0; L1_OUTPUT_DIMS]),
            l1_sqr_relu: Aligned([0; L1_OUTPUT_DIMS]),
            l1_relu: Aligned([0; L1_OUTPUT_DIMS]),
            l1_out: Aligned([0; L2_PADDED_INPUT_DIMS]),
            l2_li_out: Aligned([0; L2_PADDED_OUTPUT_DIMS]),
            l2_out: Aligned([0; L2_PADDED_OUTPUT_DIMS]),
            feature_indices: [0; NUM_FEATURES],
        }
    }
}

thread_local! {
    static NETWORK_BUFFERS: std::cell::RefCell<NetworkBuffers> =
        std::cell::RefCell::new(NetworkBuffers::new());
}

pub struct Network {
    base_input: BaseInput<INPUT_FEATURE_DIMS, BASE_OUTPUT_DIMS, { BASE_OUTPUT_DIMS * 2 }>,
    pa_inputs: Vec<PhaseAdaptiveInput<INPUT_FEATURE_DIMS, PA_OUTPUT_DIMS>>,
    layer_stacks: Vec<LayerStack>,
}

impl Network {
    pub fn new(file_path: &str) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut decoder = zstd::stream::read::Decoder::new(reader)?;

        let base_input = BaseInput::<INPUT_FEATURE_DIMS, BASE_OUTPUT_DIMS, { BASE_OUTPUT_DIMS * 2 }>::load(&mut decoder,)?;

        let mut pa_inputs = Vec::with_capacity(NUM_PHASE_ADAPTIVE_INPUT);
        for _ in 0..NUM_PHASE_ADAPTIVE_INPUT {
            let pa_input = PhaseAdaptiveInput::<INPUT_FEATURE_DIMS, PA_OUTPUT_DIMS>::load(&mut decoder,)?;
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

        Ok(Network {
            base_input,
            pa_inputs,
            layer_stacks,
        })
    }

    pub fn evaluate(&self, ctx: &SearchContext, board: &Board) -> Score {
        let ply = ctx.ply();
        let mobility = board.get_moves().count_ones();
        let feature_indices = if ctx.player == 0 {
            &ctx.feature_set.p_features[ply]
        } else {
            &ctx.feature_set.o_features[ply]
        };

        NETWORK_BUFFERS.with(|buffers| {
            let mut buffers = buffers.borrow_mut();

            for (i, &offset) in (0..NUM_PATTERN_FEATURES).zip(PATTERN_FEATURE_OFFSETS.iter()) {
                buffers.feature_indices[i] = unsafe { feature_indices.v1 }[i] as usize + offset;
            }

            let score = self.forward(&mut buffers, mobility as u8, ply);
            score.clamp(MID_SCORE_MIN + 1, MID_SCORE_MAX - 1)
        })
    }

    #[inline(always)]
    fn forward(&self, buffers: &mut NetworkBuffers, mobility: u8, ply: usize) -> Score {
        self.forward_input_base(buffers, mobility);
        self.forward_input_pa(buffers, mobility, ply);

        let ls_index = ply / (60 / NUM_LAYER_STACKS);
        let ls = &self.layer_stacks[ls_index];
        self.forward_l1(ls, buffers);
        self.forward_l2(ls, buffers);
        self.forward_output(ls, buffers)
    }

    #[inline(always)]
    fn forward_input_base(&self, buffers: &mut NetworkBuffers, mobility: u8) {
        let feature_indices = &buffers.feature_indices;
        let output = &mut buffers.base_out;

        self.base_input.forward(feature_indices, &mut output[0..BASE_OUTPUT_DIMS]);
        output[L1_BASE_INPUT_DIMS - 1] = mobility * MOBILITY_SCALE;
    }

    #[inline(always)]
    fn forward_input_pa(&self, buffers: &mut NetworkBuffers, mobility: u8, ply: usize) {
        let feature_indices = &buffers.feature_indices;
        let output = &mut buffers.pa_out;

        let pa_index = ply / (60 / NUM_PHASE_ADAPTIVE_INPUT);
        let pa_input = &self.pa_inputs[pa_index];
        pa_input.forward_leaky_relu(feature_indices, &mut output[0..PA_OUTPUT_DIMS]);
        output[L1_PA_INPUT_DIMS - 1] = mobility * MOBILITY_SCALE;
    }

    #[inline(always)]
    fn forward_l1(&self, ls: &LayerStack, buffers: &mut NetworkBuffers) {
        ls.l1_base.forward(
            &buffers.base_out,
            &mut buffers.l1_base_out,
        );
        ls.l1_pa.forward(
            &buffers.pa_out,
            &mut buffers.l1_pa_out);

        buffers.l1_li_out[..L1_BASE_OUTPUT_DIMS]
            .copy_from_slice(&buffers.l1_base_out[..L1_BASE_OUTPUT_DIMS]);
        buffers.l1_li_out[L1_BASE_OUTPUT_DIMS..L1_OUTPUT_DIMS]
            .copy_from_slice(&buffers.l1_pa_out[..L1_PA_OUTPUT_DIMS]);

        sqr_clipped_relu::<L1_OUTPUT_DIMS>(&buffers.l1_li_out, &mut buffers.l1_sqr_relu);
        clipped_relu::<L1_OUTPUT_DIMS>(&buffers.l1_li_out, &mut buffers.l1_relu);

        const L2_INPUT_DIMS_HALF: usize = L2_INPUT_DIMS / 2;
        buffers.l1_out[..L2_INPUT_DIMS_HALF]
            .copy_from_slice(buffers.l1_sqr_relu.as_slice());
        buffers.l1_out[L2_INPUT_DIMS_HALF..L2_INPUT_DIMS]
            .copy_from_slice(buffers.l1_relu.as_slice());
    }

    #[inline(always)]
    fn forward_l2(&self, ls: &LayerStack, buffers: &mut NetworkBuffers) {
        let input = &buffers.l1_out;
        let li_output = &mut buffers.l2_li_out;
        let output = &mut buffers.l2_out;

        ls.l2.forward(input, li_output);
        clipped_relu::<L2_PADDED_OUTPUT_DIMS>(li_output, output);
    }

    #[inline(always)]
    fn forward_output(&self, ls: &LayerStack, buffers: &mut NetworkBuffers) -> Score {
        let input = &buffers.l2_out;
        let mut output: Aligned<A64, [i32; 32]> = Aligned([0; 32]);

        ls.lo.forward(input, &mut output);
        output[0] >> OUTPUT_WEIGHT_SCALE_BITS
    }
}

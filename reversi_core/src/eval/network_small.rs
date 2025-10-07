//! Small neural network for endgame evaluation.

use std::fs::File;
use std::io::{self, BufReader};

use crate::board::Board;
use crate::constants::{MID_SCORE_MAX, MID_SCORE_MIN};
use crate::eval::activations::clipped_relu;
use crate::eval::input_layer::PhaseAdaptiveInput;
use crate::eval::linear_layer::LinearLayer;
use crate::eval::pattern_feature::{NUM_PATTERN_FEATURES, PatternFeature};
use crate::types::Score;
use crate::util::align::Align64;
use crate::util::ceil_to_multiple;

use super::constants::{
    INPUT_FEATURE_DIMS, MOBILITY_SCALE, NUM_FEATURES, OUTPUT_WEIGHT_SCALE_BITS,
    PATTERN_FEATURE_OFFSETS,
};

const PA_OUTPUT_DIMS: usize = 64;

const L1_PA_INPUT_DIMS: usize = PA_OUTPUT_DIMS + 1;
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

/// Simplified layer stack with fewer layers than the main network
struct LayerStack {
    pub l1_pa: LinearLayer<
        L1_PA_INPUT_DIMS,
        L1_PA_OUTPUT_DIMS,
        L1_PA_PADDED_INPUT_DIMS,
        L1_PA_PADDED_OUTPUT_DIMS,
    >,
    pub l2: LinearLayer<L2_INPUT_DIMS, L2_OUTPUT_DIMS, L2_PADDED_INPUT_DIMS, L2_PADDED_OUTPUT_DIMS>,
    pub lo: LinearLayer<
        LO_INPUT_DIMS,
        1,
        { ceil_to_multiple(LO_INPUT_DIMS, 32) },
        { ceil_to_multiple(1, 32) },
    >,
}

/// Thread-local buffers for small network computation
struct NetworkBuffers {
    pa_out: Align64<[u8; L1_PA_PADDED_INPUT_DIMS]>,
    l1_pa_out: Align64<[i32; L1_PA_PADDED_OUTPUT_DIMS]>,
    l1_out: Align64<[u8; L2_PADDED_INPUT_DIMS]>,
    l2_li_out: Align64<[i32; L2_PADDED_OUTPUT_DIMS]>,
    l2_out: Align64<[u8; L2_PADDED_OUTPUT_DIMS]>,
    feature_indices: [usize; NUM_FEATURES],
}

impl NetworkBuffers {
    #[inline]
    fn new() -> Self {
        Self {
            pa_out: Align64([0; L1_PA_PADDED_INPUT_DIMS]),
            l1_pa_out: Align64([0; L1_PA_PADDED_OUTPUT_DIMS]),
            l1_out: Align64([0; L2_PADDED_INPUT_DIMS]),
            l2_li_out: Align64([0; L2_PADDED_OUTPUT_DIMS]),
            l2_out: Align64([0; L2_PADDED_OUTPUT_DIMS]),
            feature_indices: [0; NUM_FEATURES],
        }
    }
}

thread_local! {
    static NETWORK_BUFFERS: std::cell::RefCell<NetworkBuffers> =
        std::cell::RefCell::new(NetworkBuffers::new());
}

/// Small neural network optimized for endgame positions
pub struct NetworkSmall {
    pa_inputs: Vec<PhaseAdaptiveInput<INPUT_FEATURE_DIMS, PA_OUTPUT_DIMS>>,
    layer_stacks: Vec<LayerStack>,
}

impl NetworkSmall {
    /// Creates a new small network from compressed weights file
    pub fn new(file_path: &str) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut decoder = zstd::stream::read::Decoder::new(reader)?;

        let mut pa_inputs = Vec::with_capacity(NUM_PHASE_ADAPTIVE_INPUT);
        for _ in 0..NUM_PHASE_ADAPTIVE_INPUT {
            let pa_input =
                PhaseAdaptiveInput::<INPUT_FEATURE_DIMS, PA_OUTPUT_DIMS>::load(&mut decoder)?;
            pa_inputs.push(pa_input);
        }

        let mut layer_stacks = Vec::with_capacity(NUM_LAYER_STACKS);
        for _ in 0..NUM_LAYER_STACKS {
            let l1_pa = LinearLayer::load(&mut decoder)?;
            let l2 = LinearLayer::load(&mut decoder)?;
            let lo = LinearLayer::load(&mut decoder)?;
            layer_stacks.push(LayerStack { l1_pa, l2, lo });
        }

        Ok(NetworkSmall {
            pa_inputs,
            layer_stacks,
        })
    }

    /// Evaluates a position using the small network
    ///
    /// Faster but less accurate than the main network
    ///
    /// # Arguments
    /// * `board` - The current board state
    /// * `pattern_feature` - Extracted pattern features from the board
    /// * `ply` - Current game ply (move number)
    pub fn evaluate(&self, board: &Board, pattern_feature: &PatternFeature, ply: usize) -> Score {
        let mobility = board.get_moves().count_ones();

        NETWORK_BUFFERS.with(|buffers_cell| {
            let mut buffers = buffers_cell.borrow_mut();
            for (i, &offset) in (0..NUM_PATTERN_FEATURES).zip(PATTERN_FEATURE_OFFSETS.iter()) {
                buffers.feature_indices[i] = pattern_feature[i] as usize + offset;
            }

            let score = self.forward(&mut buffers, mobility as u8, ply);
            score.clamp(MID_SCORE_MIN + 1, MID_SCORE_MAX - 1)
        })
    }

    #[inline(always)]
    fn forward(&self, buffers: &mut NetworkBuffers, mobility: u8, ply: usize) -> Score {
        self.forward_input_pa(buffers, mobility, ply);

        let ls_index = ply / (60 / NUM_LAYER_STACKS);
        let ls = &self.layer_stacks[ls_index];
        self.forward_l1(ls, buffers);
        self.forward_l2(ls, buffers);
        self.forward_output(ls, buffers)
    }

    #[inline(always)]
    fn forward_input_pa(&self, buffers: &mut NetworkBuffers, mobility: u8, ply: usize) {
        let pa_index = if ply < 30 { 0 } else { (ply - 30) / 6 + 1 };
        let pa_input = &self.pa_inputs[pa_index];
        pa_input.forward(&buffers.feature_indices, buffers.pa_out.as_mut_slice());
        buffers.pa_out[L1_PA_INPUT_DIMS - 1] = mobility * MOBILITY_SCALE;
    }

    #[inline(always)]
    fn forward_l1(&self, ls: &LayerStack, buffers: &mut NetworkBuffers) {
        ls.l1_pa.forward(&buffers.pa_out, &mut buffers.l1_pa_out);
        clipped_relu::<L2_PADDED_INPUT_DIMS>(&buffers.l1_pa_out, &mut buffers.l1_out);
    }

    #[inline(always)]
    fn forward_l2(&self, ls: &LayerStack, buffers: &mut NetworkBuffers) {
        ls.l2.forward(&buffers.l1_out, &mut buffers.l2_li_out);
        clipped_relu::<L2_PADDED_OUTPUT_DIMS>(&buffers.l2_li_out, &mut buffers.l2_out);
    }

    #[inline(always)]
    fn forward_output(&self, ls: &LayerStack, buffers: &mut NetworkBuffers) -> Score {
        let input = &buffers.l2_out;
        let mut output = Align64([0; 32]);

        ls.lo.forward(input, &mut output);
        output[0] >> OUTPUT_WEIGHT_SCALE_BITS
    }
}

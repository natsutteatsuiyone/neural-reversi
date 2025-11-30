//! Neural network for midgame evaluation.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use crate::board::Board;
use crate::constants::{MID_SCORE_MAX, MID_SCORE_MIN};
use crate::eval::activations::{clipped_relu, screlu, sqr_clipped_relu};
use crate::eval::input_layer::{BaseInput, PhaseAdaptiveInput};
use crate::eval::linear_layer::LinearLayer;
use crate::eval::output_layer::OutputLayer;
use crate::eval::pattern_feature::{INPUT_FEATURE_DIMS, PatternFeature};
use crate::types::Score;
use crate::util::align::Align64;
use crate::util::ceil_to_multiple;

const BASE_OUTPUT_DIMS: usize = 128;
const PA_OUTPUT_DIMS: usize = 128;

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

const LO_INPUT_DIMS: usize = L2_OUTPUT_DIMS + BASE_OUTPUT_DIMS + PA_OUTPUT_DIMS;
const LO_PADDED_INPUT_DIMS: usize = ceil_to_multiple(LO_INPUT_DIMS, 32);

const MOBILITY_SCALE: u8 = 7;
const OUTPUT_WEIGHT_SCALE_BITS: u32 = 6;
const NUM_LAYER_STACKS: usize = 60;
const NUM_PA_INPUTS: usize = 6;
const PA_INPUT_BUCKET_SIZE: usize = 60 / NUM_PA_INPUTS;

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
    pub l2: LinearLayer<L2_INPUT_DIMS, L2_OUTPUT_DIMS, L2_PADDED_INPUT_DIMS, L2_PADDED_OUTPUT_DIMS>,
    pub lo: OutputLayer<LO_INPUT_DIMS, LO_PADDED_INPUT_DIMS>,
}

/// Thread-local working buffers for network computation
struct NetworkBuffers {
    base_out: Align64<[u8; L1_BASE_PADDED_INPUT_DIMS]>,
    pa_out: Align64<[u8; L1_PA_PADDED_INPUT_DIMS]>,
    l1_base_out: Align64<[i32; L1_BASE_PADDED_OUTPUT_DIMS]>,
    l1_pa_out: Align64<[i32; L1_PA_PADDED_OUTPUT_DIMS]>,
    l1_li_out: Align64<[i32; L1_OUTPUT_DIMS]>,
    l1_out: Align64<[u8; L2_PADDED_INPUT_DIMS]>,
    l2_li_out: Align64<[i32; L2_PADDED_OUTPUT_DIMS]>,
    l2_out: Align64<[u8; L2_PADDED_OUTPUT_DIMS]>,
}

impl NetworkBuffers {
    #[inline]
    fn new() -> Self {
        Self {
            base_out: Align64([0; L1_BASE_PADDED_INPUT_DIMS]),
            pa_out: Align64([0; L1_PA_PADDED_INPUT_DIMS]),
            l1_base_out: Align64([0; L1_BASE_PADDED_OUTPUT_DIMS]),
            l1_pa_out: Align64([0; L1_PA_PADDED_OUTPUT_DIMS]),
            l1_li_out: Align64([0; L1_OUTPUT_DIMS]),
            l1_out: Align64([0; L2_PADDED_INPUT_DIMS]),
            l2_li_out: Align64([0; L2_PADDED_OUTPUT_DIMS]),
            l2_out: Align64([0; L2_PADDED_OUTPUT_DIMS]),
        }
    }
}

thread_local! {
    static NETWORK_BUFFERS: std::cell::RefCell<NetworkBuffers> =
        std::cell::RefCell::new(NetworkBuffers::new());
}

/// Main neural network structure for position evaluation
pub struct Network {
    base_input: BaseInput<INPUT_FEATURE_DIMS, BASE_OUTPUT_DIMS, { BASE_OUTPUT_DIMS * 2 }>,
    pa_inputs: Vec<PhaseAdaptiveInput<INPUT_FEATURE_DIMS, PA_OUTPUT_DIMS>>,
    layer_stacks: Vec<LayerStack>,
}

impl Network {
    /// Creates a new network by loading weights from a compressed file
    pub fn new(file_path: &Path) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        Self::from_reader(reader)
    }

    /// Creates a new network by loading weights from an in-memory blob
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let cursor = io::Cursor::new(bytes);
        Self::from_reader(cursor)
    }

    fn from_reader<R: Read>(reader: R) -> io::Result<Self> {
        let mut decoder = zstd::stream::read::Decoder::new(reader)?;

        let base_input =
            BaseInput::<INPUT_FEATURE_DIMS, BASE_OUTPUT_DIMS, { BASE_OUTPUT_DIMS * 2 }>::load(
                &mut decoder,
            )?;

        let mut pa_inputs = Vec::with_capacity(NUM_PA_INPUTS);
        for _ in 0..NUM_PA_INPUTS {
            let pa_input =
                PhaseAdaptiveInput::<INPUT_FEATURE_DIMS, PA_OUTPUT_DIMS>::load(&mut decoder)?;
            pa_inputs.push(pa_input);
        }

        let mut layer_stacks = Vec::with_capacity(NUM_LAYER_STACKS);
        for _ in 0..NUM_LAYER_STACKS {
            let l1_base = LinearLayer::load(&mut decoder)?;
            let l1_pa = LinearLayer::load(&mut decoder)?;
            let l2 = LinearLayer::load(&mut decoder)?;
            let lo = OutputLayer::load(&mut decoder)?;
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

    /// Evaluates a board position using the neural network
    ///
    /// # Arguments
    /// * `board` - The current board state
    /// * `pattern_feature` - Extracted pattern features from the board
    /// * `ply` - Current game ply (move number)
    pub fn evaluate(&self, board: &Board, pattern_feature: &PatternFeature, ply: usize) -> Score {
        let mobility = board.get_moves().count_ones();

        NETWORK_BUFFERS.with(|buffers| {
            let mut buffers = buffers.borrow_mut();
            let score = self.forward(&mut buffers, pattern_feature, mobility as u8, ply);
            score.clamp(MID_SCORE_MIN + 1, MID_SCORE_MAX - 1)
        })
    }

    #[inline(always)]
    fn forward(
        &self,
        buffers: &mut NetworkBuffers,
        pattern_feature: &PatternFeature,
        mobility: u8,
        ply: usize,
    ) -> Score {
        self.forward_input_base(buffers, pattern_feature);
        self.forward_input_pa(buffers, pattern_feature, ply);
        let mobility_scaled = mobility.saturating_mul(MOBILITY_SCALE);
        buffers.base_out[L1_BASE_INPUT_DIMS - 1] = mobility_scaled;
        buffers.pa_out[L1_PA_INPUT_DIMS - 1] = mobility_scaled;

        let ls = &self.layer_stacks[ply];
        self.forward_l1(ls, buffers);
        self.forward_l2(ls, buffers);
        self.forward_output(ls, buffers)
    }

    #[inline(always)]
    fn forward_input_base(&self, buffers: &mut NetworkBuffers, pattern_feature: &PatternFeature) {
        let output = &mut buffers.base_out[0..BASE_OUTPUT_DIMS];
        self.base_input.forward(pattern_feature, output);
    }

    #[inline(always)]
    fn forward_input_pa(
        &self,
        buffers: &mut NetworkBuffers,
        pattern_feature: &PatternFeature,
        ply: usize,
    ) {
        let output = &mut buffers.pa_out[0..PA_OUTPUT_DIMS];

        let pa_index = ply / PA_INPUT_BUCKET_SIZE;
        let pa_input = &self.pa_inputs[pa_index];
        pa_input.forward(pattern_feature, output);
    }

    #[inline(always)]
    fn forward_l1(&self, ls: &LayerStack, buffers: &mut NetworkBuffers) {
        ls.l1_base
            .forward(&buffers.base_out, &mut buffers.l1_base_out);
        ls.l1_pa.forward(&buffers.pa_out, &mut buffers.l1_pa_out);

        buffers.l1_li_out[..L1_BASE_OUTPUT_DIMS]
            .copy_from_slice(&buffers.l1_base_out[..L1_BASE_OUTPUT_DIMS]);
        buffers.l1_li_out[L1_BASE_OUTPUT_DIMS..L1_OUTPUT_DIMS]
            .copy_from_slice(&buffers.l1_pa_out[..L1_PA_OUTPUT_DIMS]);

        const L2_INPUT_DIMS_HALF: usize = L2_INPUT_DIMS / 2;
        sqr_clipped_relu::<L1_OUTPUT_DIMS>(
            buffers.l1_li_out.as_slice(),
            &mut buffers.l1_out[..L2_INPUT_DIMS_HALF],
        );
        clipped_relu::<L1_OUTPUT_DIMS>(
            buffers.l1_li_out.as_slice(),
            &mut buffers.l1_out[L2_INPUT_DIMS_HALF..L2_INPUT_DIMS],
        );
    }

    #[inline(always)]
    fn forward_l2(&self, ls: &LayerStack, buffers: &mut NetworkBuffers) {
        let input = &buffers.l1_out;
        let li_output = &mut buffers.l2_li_out;
        let output = &mut buffers.l2_out;

        ls.l2.forward(input, li_output);
        screlu::<L2_PADDED_OUTPUT_DIMS>(li_output.as_slice(), output.as_mut_slice());
    }

    #[inline(always)]
    fn forward_output(&self, ls: &LayerStack, buffers: &mut NetworkBuffers) -> Score {
        let segments = [
            &buffers.l2_out[..L2_OUTPUT_DIMS],
            &buffers.base_out[..BASE_OUTPUT_DIMS],
            &buffers.pa_out[..PA_OUTPUT_DIMS],
        ];

        ls.lo.forward(segments) >> OUTPUT_WEIGHT_SCALE_BITS
    }
}

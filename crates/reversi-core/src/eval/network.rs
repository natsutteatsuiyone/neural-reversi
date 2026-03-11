//! Neural network for midgame evaluation.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use crate::board::Board;
use crate::eval::activations::{clipped_relu, screlu, sqr_clipped_relu};
use crate::eval::input_layer::{BaseInput, PhaseAdaptiveInput};
use crate::eval::linear_layer::LinearLayer;
use crate::eval::output_layer::OutputLayer;
use crate::eval::pattern_feature::{INPUT_FEATURE_DIMS, PatternFeature};
use crate::eval::util::ceil_to_multiple;
use crate::types::ScaledScore;
use crate::util::align::Align64;

const BASE_OUTPUT_DIMS: usize = 128;
const PA_OUTPUT_DIMS: usize = 128;

const L1_INPUT_DIMS: usize = BASE_OUTPUT_DIMS + PA_OUTPUT_DIMS + 1;
const L1_PADDED_INPUT_DIMS: usize = ceil_to_multiple(L1_INPUT_DIMS, 32);
const L1_OUTPUT_DIMS: usize = 16;
const L1_PADDED_OUTPUT_DIMS: usize = ceil_to_multiple(L1_OUTPUT_DIMS, 32);

const L2_INPUT_DIMS: usize = L1_OUTPUT_DIMS * 2;
const L2_PADDED_INPUT_DIMS: usize = ceil_to_multiple(L2_INPUT_DIMS, 32);
const L2_OUTPUT_DIMS: usize = 64;
const L2_PADDED_OUTPUT_DIMS: usize = ceil_to_multiple(L2_OUTPUT_DIMS, 32);

const LO_INPUT_DIMS: usize = L2_OUTPUT_DIMS + BASE_OUTPUT_DIMS + PA_OUTPUT_DIMS;
const LO_PADDED_INPUT_DIMS: usize = ceil_to_multiple(LO_INPUT_DIMS, 32);

const MOBILITY_SCALE: u8 = 7;
const OUTPUT_WEIGHT_SCALE_BITS: u32 = 6;
const NUM_LAYER_STACKS: usize = 60;

/// Layer stack for a specific game ply.
struct LayerStack {
    l1: LinearLayer<L1_INPUT_DIMS, L1_OUTPUT_DIMS, L1_PADDED_INPUT_DIMS, L1_PADDED_OUTPUT_DIMS>,
    l2: LinearLayer<L2_INPUT_DIMS, L2_OUTPUT_DIMS, L2_PADDED_INPUT_DIMS, L2_PADDED_OUTPUT_DIMS>,
    lo: OutputLayer<LO_INPUT_DIMS, LO_PADDED_INPUT_DIMS>,
}

/// Thread-local working buffers for network computation.
struct NetworkBuffers {
    l1_input: Align64<[u8; L1_PADDED_INPUT_DIMS]>,
    l1_li_out: Align64<[i32; L1_PADDED_OUTPUT_DIMS]>,
    l1_out: Align64<[u8; L2_PADDED_INPUT_DIMS]>,
    l2_li_out: Align64<[i32; L2_PADDED_OUTPUT_DIMS]>,
    l2_out: Align64<[u8; L2_PADDED_OUTPUT_DIMS]>,
}

impl NetworkBuffers {
    #[inline]
    fn new() -> Self {
        Self {
            l1_input: Align64([0; L1_PADDED_INPUT_DIMS]),
            l1_li_out: Align64([0; L1_PADDED_OUTPUT_DIMS]),
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

/// Main neural network structure for position evaluation.
pub struct Network {
    base_input: BaseInput<INPUT_FEATURE_DIMS, BASE_OUTPUT_DIMS, { BASE_OUTPUT_DIMS * 2 }>,
    pa_input: PhaseAdaptiveInput<INPUT_FEATURE_DIMS, PA_OUTPUT_DIMS>,
    layer_stacks: Vec<LayerStack>,
}

impl Network {
    /// Creates a new network by loading weights from a compressed file.
    pub fn new(file_path: &Path) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        Self::from_reader(reader)
    }

    /// Creates a new network by loading weights from an in-memory blob.
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

        let pa_input =
            PhaseAdaptiveInput::<INPUT_FEATURE_DIMS, PA_OUTPUT_DIMS>::load(&mut decoder)?;

        let mut layer_stacks = Vec::with_capacity(NUM_LAYER_STACKS);
        for _ in 0..NUM_LAYER_STACKS {
            let l1 = LinearLayer::load(&mut decoder)?;
            let l2 = LinearLayer::load(&mut decoder)?;
            let lo = OutputLayer::load(&mut decoder)?;
            layer_stacks.push(LayerStack { l1, l2, lo });
        }

        Ok(Network {
            base_input,
            pa_input,
            layer_stacks,
        })
    }

    /// Evaluates a board position using the neural network.
    pub fn evaluate(
        &self,
        board: &Board,
        pattern_feature: &PatternFeature,
        ply: usize,
    ) -> ScaledScore {
        let mobility = board.get_moves().count();

        NETWORK_BUFFERS.with(|buffers| {
            let mut buffers = buffers.borrow_mut();
            let score = self.forward(&mut buffers, pattern_feature, mobility as u8, ply);
            score.clamp(ScaledScore::MIN + 1, ScaledScore::MAX - 1)
        })
    }

    #[inline(always)]
    fn forward(
        &self,
        buffers: &mut NetworkBuffers,
        pattern_feature: &PatternFeature,
        mobility: u8,
        ply: usize,
    ) -> ScaledScore {
        self.base_input
            .forward(pattern_feature, &mut buffers.l1_input[..BASE_OUTPUT_DIMS]);
        self.pa_input.forward(
            pattern_feature,
            ply,
            &mut buffers.l1_input[BASE_OUTPUT_DIMS..BASE_OUTPUT_DIMS + PA_OUTPUT_DIMS],
        );

        buffers.l1_input[L1_INPUT_DIMS - 1] = mobility.saturating_mul(MOBILITY_SCALE);

        let ls = &self.layer_stacks[ply];
        self.forward_l1(ls, buffers);
        self.forward_l2(ls, buffers);
        self.forward_output(ls, buffers)
    }

    #[inline(always)]
    fn forward_l1(&self, ls: &LayerStack, buffers: &mut NetworkBuffers) {
        ls.l1.forward(&buffers.l1_input, &mut buffers.l1_li_out);

        const L2_INPUT_DIMS_HALF: usize = L2_INPUT_DIMS / 2;
        let l1_out = &buffers.l1_li_out[..L1_OUTPUT_DIMS];
        sqr_clipped_relu::<L1_OUTPUT_DIMS>(l1_out, &mut buffers.l1_out[..L2_INPUT_DIMS_HALF]);
        clipped_relu::<L1_OUTPUT_DIMS>(
            l1_out,
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
    fn forward_output(&self, ls: &LayerStack, buffers: &mut NetworkBuffers) -> ScaledScore {
        let segments = [
            &buffers.l2_out[..L2_OUTPUT_DIMS],
            &buffers.l1_input[..BASE_OUTPUT_DIMS],
            &buffers.l1_input[BASE_OUTPUT_DIMS..BASE_OUTPUT_DIMS + PA_OUTPUT_DIMS],
        ];

        ScaledScore::from_raw(ls.lo.forward(segments) >> OUTPUT_WEIGHT_SCALE_BITS)
    }
}

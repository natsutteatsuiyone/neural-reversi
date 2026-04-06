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
const L2_INPUT_DIMS_HALF: usize = L2_INPUT_DIMS / 2;

const LO_INPUT_DIMS: usize = L2_OUTPUT_DIMS + BASE_OUTPUT_DIMS + PA_OUTPUT_DIMS;
const LO_PADDED_INPUT_DIMS: usize = ceil_to_multiple(LO_INPUT_DIMS, 32);

const PA_INPUT_START: usize = BASE_OUTPUT_DIMS;
const PA_INPUT_END: usize = PA_INPUT_START + PA_OUTPUT_DIMS;
const MOBILITY_INPUT_INDEX: usize = L1_INPUT_DIMS - 1;
const MOBILITY_SCALE: u8 = 7;
const OUTPUT_WEIGHT_SCALE_BITS: u32 = 6;
const NUM_LAYER_STACKS: usize = 60;

type L1Layer =
    LinearLayer<L1_INPUT_DIMS, L1_OUTPUT_DIMS, L1_PADDED_INPUT_DIMS, L1_PADDED_OUTPUT_DIMS>;
type L2Layer =
    LinearLayer<L2_INPUT_DIMS, L2_OUTPUT_DIMS, L2_PADDED_INPUT_DIMS, L2_PADDED_OUTPUT_DIMS>;
type FinalOutputLayer = OutputLayer<LO_INPUT_DIMS, LO_PADDED_INPUT_DIMS>;

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

    #[inline(always)]
    fn base_input_mut(&mut self) -> &mut [u8] {
        &mut self.l1_input[..BASE_OUTPUT_DIMS]
    }

    #[inline(always)]
    fn pa_input_mut(&mut self) -> &mut [u8] {
        &mut self.l1_input[PA_INPUT_START..PA_INPUT_END]
    }

    #[inline(always)]
    fn output_segments(&self) -> [&[u8]; 3] {
        [
            &self.l2_out[..L2_OUTPUT_DIMS],
            &self.l1_input[..BASE_OUTPUT_DIMS],
            &self.l1_input[PA_INPUT_START..PA_INPUT_END],
        ]
    }
}

thread_local! {
    static NETWORK_BUFFERS: std::cell::RefCell<NetworkBuffers> =
        std::cell::RefCell::new(NetworkBuffers::new());
}

/// Layer stack for a specific game ply.
struct LayerStack {
    l1: L1Layer,
    l2: L2Layer,
    lo: FinalOutputLayer,
}

impl LayerStack {
    fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let l1 = LinearLayer::load(reader)?;
        let l2 = LinearLayer::load(reader)?;
        let lo = OutputLayer::load(reader)?;
        Ok(Self { l1, l2, lo })
    }

    #[inline(always)]
    fn forward(&self, buffers: &mut NetworkBuffers) -> ScaledScore {
        self.forward_l1(buffers);
        self.forward_l2(buffers);
        self.forward_output(buffers)
    }

    #[inline(always)]
    fn forward_l1(&self, buffers: &mut NetworkBuffers) {
        self.l1.forward(&buffers.l1_input, &mut buffers.l1_li_out);
        let l1_out = &buffers.l1_li_out[..L1_OUTPUT_DIMS];
        let (sqr_out, relu_out) = buffers.l1_out[..L2_INPUT_DIMS].split_at_mut(L2_INPUT_DIMS_HALF);
        sqr_clipped_relu::<L1_OUTPUT_DIMS>(l1_out, sqr_out);
        clipped_relu::<L1_OUTPUT_DIMS>(l1_out, relu_out);
    }

    #[inline(always)]
    fn forward_l2(&self, buffers: &mut NetworkBuffers) {
        self.l2.forward(&buffers.l1_out, &mut buffers.l2_li_out);
        screlu::<L2_PADDED_OUTPUT_DIMS>(
            buffers.l2_li_out.as_slice(),
            buffers.l2_out.as_mut_slice(),
        );
    }

    #[inline(always)]
    fn forward_output(&self, buffers: &NetworkBuffers) -> ScaledScore {
        let score = (self.lo.forward(buffers.output_segments()) >> OUTPUT_WEIGHT_SCALE_BITS)
            .clamp(-ScaledScore::INF.value(), ScaledScore::INF.value());
        ScaledScore::from_raw(score)
    }
}

/// Main neural network structure for position evaluation.
pub struct Network {
    base_input: BaseInput<INPUT_FEATURE_DIMS, BASE_OUTPUT_DIMS, { BASE_OUTPUT_DIMS * 2 }>,
    pa_input: PhaseAdaptiveInput<INPUT_FEATURE_DIMS, PA_OUTPUT_DIMS>,
    layer_stacks: Box<[LayerStack]>,
}

impl Network {
    /// Creates a new network by loading weights from a compressed file.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the file cannot be opened or the weights are malformed.
    pub fn new(file_path: &Path) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        Self::from_reader(reader)
    }

    /// Creates a new network by loading weights from an in-memory blob.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the weights are malformed.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let cursor = io::Cursor::new(bytes);
        Self::from_reader(cursor)
    }

    fn load_layer_stacks<R: Read>(reader: &mut R) -> io::Result<Box<[LayerStack]>> {
        let mut layer_stacks = Vec::with_capacity(NUM_LAYER_STACKS);
        for _ in 0..NUM_LAYER_STACKS {
            layer_stacks.push(LayerStack::load(reader)?);
        }
        Ok(layer_stacks.into_boxed_slice())
    }

    fn from_reader<R: Read>(reader: R) -> io::Result<Self> {
        let mut decoder = zstd::stream::read::Decoder::new(reader)?;

        let base_input =
            BaseInput::<INPUT_FEATURE_DIMS, BASE_OUTPUT_DIMS, { BASE_OUTPUT_DIMS * 2 }>::load(
                &mut decoder,
            )?;

        let pa_input =
            PhaseAdaptiveInput::<INPUT_FEATURE_DIMS, PA_OUTPUT_DIMS>::load(&mut decoder)?;

        let layer_stacks = Self::load_layer_stacks(&mut decoder)?;

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
        debug_assert!(ply < self.layer_stacks.len());
        let mobility = board.get_moves().count() as u8;

        NETWORK_BUFFERS.with(|buffers| {
            let mut buffers = buffers.borrow_mut();
            let score = self.forward(&mut buffers, pattern_feature, mobility, ply);
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
            .forward(pattern_feature, buffers.base_input_mut());
        self.pa_input
            .forward(pattern_feature, ply, buffers.pa_input_mut());
        buffers.l1_input[MOBILITY_INPUT_INDEX] = mobility.saturating_mul(MOBILITY_SCALE);
        self.layer_stacks[ply].forward(buffers)
    }
}

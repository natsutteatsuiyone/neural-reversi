use std::io::{self, Read};

use crate::types::ScaledScore;

use super::activations::{screlu, sqr_clipped_and_clipped_relu_16};
use super::linear_layer::LinearLayer;
use super::output_layer::OutputLayer;
use super::{
    L1_INPUT_DIMS, L1_OUTPUT_DIMS, L1_PADDED_INPUT_DIMS, L1_PADDED_OUTPUT_DIMS, L2_INPUT_DIMS,
    L2_OUTPUT_DIMS, L2_PADDED_INPUT_DIMS, L2_PADDED_OUTPUT_DIMS, LO_INPUT_DIMS,
    LO_PADDED_INPUT_DIMS, NetworkBuffers,
};

const OUTPUT_WEIGHT_SCALE_BITS: u32 = 6;
const NUM_LAYER_STACKS: usize = 60;

type L1Layer =
    LinearLayer<L1_INPUT_DIMS, L1_OUTPUT_DIMS, L1_PADDED_INPUT_DIMS, L1_PADDED_OUTPUT_DIMS>;
type L2Layer =
    LinearLayer<L2_INPUT_DIMS, L2_OUTPUT_DIMS, L2_PADDED_INPUT_DIMS, L2_PADDED_OUTPUT_DIMS>;
type FinalOutputLayer = OutputLayer<LO_INPUT_DIMS, LO_PADDED_INPUT_DIMS>;

/// Layer stack for a specific game ply.
pub(super) struct LayerStack {
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
    pub(super) fn forward(&self, buffers: &mut NetworkBuffers) -> ScaledScore {
        self.forward_l1(buffers);
        self.forward_l2(buffers);
        self.forward_output(buffers)
    }

    #[inline(always)]
    fn forward_l1(&self, buffers: &mut NetworkBuffers) {
        self.l1.forward(&buffers.l1_input, &mut buffers.l1_li_out);
        let l1_out = &buffers.l1_li_out[..L1_OUTPUT_DIMS];
        sqr_clipped_and_clipped_relu_16(l1_out, &mut buffers.l1_out[..L2_INPUT_DIMS]);
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

pub(super) fn load_layer_stacks<R: Read>(reader: &mut R) -> io::Result<Box<[LayerStack]>> {
    let mut layer_stacks = Vec::with_capacity(NUM_LAYER_STACKS);
    for _ in 0..NUM_LAYER_STACKS {
        layer_stacks.push(LayerStack::load(reader)?);
    }
    Ok(layer_stacks.into_boxed_slice())
}

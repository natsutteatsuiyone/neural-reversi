use std::io::{self, Read};

use crate::score::Score;

use super::activations::{screlu, screlu_store_order, sqr_clipped_and_clipped_relu_16};
use super::linear_layer::LinearLayer;
use super::output_layer::OutputLayer;
use super::{
    L1_INPUT_DIMS, L1_OUTPUT_DIMS, L1_PADDED_INPUT_DIMS, L1_PADDED_OUTPUT_DIMS, L2_INPUT_DIMS,
    L2_OUTPUT_DIMS, L2_PADDED_INPUT_DIMS, L2_PADDED_OUTPUT_DIMS, LO_PADDED_INPUT_DIMS,
    NetworkBuffers, PA_INPUT_END,
};

const OUTPUT_WEIGHT_SCALE_BITS: u32 = 6;
pub(in crate::eval::network) const NUM_LAYER_STACKS: usize = 60;

type L1Layer =
    LinearLayer<L1_INPUT_DIMS, L1_OUTPUT_DIMS, L1_PADDED_INPUT_DIMS, L1_PADDED_OUTPUT_DIMS>;
type L2Layer =
    LinearLayer<L2_INPUT_DIMS, L2_OUTPUT_DIMS, L2_PADDED_INPUT_DIMS, L2_PADDED_OUTPUT_DIMS>;
type FinalOutputLayer = OutputLayer<L2_OUTPUT_DIMS, PA_INPUT_END, LO_PADDED_INPUT_DIMS>;

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
        let mut lo = OutputLayer::load(reader)?;
        // `screlu` emits the L2 activations in its own store order; the first
        // output-layer segment reads them positionally.
        lo.permute_leading_weights(
            &screlu_store_order::<L2_PADDED_OUTPUT_DIMS>()[..L2_OUTPUT_DIMS],
        );
        Ok(Self { l1, l2, lo })
    }

    #[inline(always)]
    pub(super) fn forward(&self, buffers: &mut NetworkBuffers) -> Score {
        self.forward_l1(buffers);
        self.forward_l2(buffers);
        self.forward_output(buffers)
    }

    #[inline(always)]
    fn forward_l1(&self, buffers: &mut NetworkBuffers) {
        self.l1.forward(&buffers.l1_input, &mut buffers.l1_li_out);
        sqr_clipped_and_clipped_relu_16(&buffers.l1_li_out, &mut buffers.l1_out);
    }

    #[inline(always)]
    fn forward_l2(&self, buffers: &mut NetworkBuffers) {
        self.l2.forward(&buffers.l1_out, &mut buffers.l2_li_out);
        screlu(&buffers.l2_li_out, &mut buffers.l2_out);
    }

    #[inline(always)]
    fn forward_output(&self, buffers: &NetworkBuffers) -> Score {
        let (l2_out, bypass) = buffers.output_segments();
        let score = (self.lo.forward(l2_out, bypass) >> OUTPUT_WEIGHT_SCALE_BITS)
            .clamp(-Score::INF.value(), Score::INF.value());
        Score::from_raw(score)
    }
}

pub(super) fn load_layer_stacks<R: Read>(
    reader: &mut R,
) -> io::Result<Box<[LayerStack; NUM_LAYER_STACKS]>> {
    let mut layer_stacks = Vec::with_capacity(NUM_LAYER_STACKS);
    for _ in 0..NUM_LAYER_STACKS {
        layer_stacks.push(LayerStack::load(reader)?);
    }
    let Ok(layer_stacks) = Box::try_from(layer_stacks) else {
        unreachable!()
    };
    Ok(layer_stacks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::network::{LO_INPUT_DIMS, with_network_buffers};
    use std::io::Cursor;

    #[test]
    fn loaded_stack_preserves_output_order_and_signed_scaling() {
        let hidden: [i32; L2_OUTPUT_DIMS] = std::array::from_fn(|i| i as i32 * 283 - 401);
        let bypass: [u8; LO_INPUT_DIMS - L2_OUTPUT_DIMS] =
            std::array::from_fn(|i| (i * 29 + 71) as u8);
        let weights: [i16; LO_PADDED_INPUT_DIMS] =
            std::array::from_fn(|i| (i * 3299 + 12_307) as i16);
        let activations = hidden.map(|value| {
            let clipped = i64::from(value.clamp(0, 255 * 64));
            (clipped * clipped / (1 << 20)) as u8
        });
        let dot: i64 = activations
            .into_iter()
            .chain(bypass)
            .zip(weights)
            .map(|(a, w)| i64::from(a) * i64::from(w))
            .sum();

        let mut hidden_layers = vec![0; L1_OUTPUT_DIMS * (4 + L1_PADDED_INPUT_DIMS)];
        for value in hidden {
            hidden_layers.extend(value.to_le_bytes());
        }
        hidden_layers.resize(
            hidden_layers.len() + L2_OUTPUT_DIMS * L2_PADDED_INPUT_DIMS,
            0,
        );

        for raw in [
            i32::MIN,
            -2_097_089,
            -65,
            -1,
            0,
            63,
            64,
            2_097_152,
            i32::MAX,
        ] {
            let mut bytes = hidden_layers.clone();
            bytes.extend(raw.wrapping_sub(dot as i32).to_le_bytes());
            for weight in weights {
                bytes.extend(weight.to_le_bytes());
            }
            let len = bytes.len();
            let mut reader = Cursor::new(bytes);
            let stack = LayerStack::load(&mut reader).unwrap();
            assert_eq!(reader.position(), len as u64);

            let actual = with_network_buffers(|buffers| {
                buffers.l1_input[..bypass.len()].copy_from_slice(&bypass);
                stack.forward(buffers)
            });
            assert_eq!(actual.value(), raw.div_euclid(64).clamp(-32767, 32767));
        }
    }
}

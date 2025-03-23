use super::constants::*;
use super::linear_layer::LinearLayer;
use crate::misc::ceil_to_multiple;

pub struct LayerStack {
    pub l1_univ: LinearLayer<
        L1_UNIV_INPUT_DIMS,
        L1_UNIV_OUTPUT_DIMS,
        L1_UNIV_PADDED_INPUT_DIMS,
        L1_UNIV_PADDED_OUTPUT_DIMS,
        L1_UNIV_NUM_REGS,
    >,
    pub l1_pa: LinearLayer<
        L1_PS_INPUT_DIMS,
        L1_PS_OUTPUT_DIMS,
        L1_PS_PADDED_INPUT_DIMS,
        L1_PS_PADDED_OUTPUT_DIMS,
        L1_PS_NUM_REGS,
    >,
    pub l2: LinearLayer<
        L2_INPUT_DIMS,
        L2_OUTPUT_DIMS,
        L2_PADDED_INPUT_DIMS,
        L2_PADDED_OUTPUT_DIMS,
        L2_NUM_REGS,
    >,
    pub lo: LinearLayer<
        LO_INPUT_DIMS,
        1,
        { ceil_to_multiple(LO_INPUT_DIMS, 32) },
        { ceil_to_multiple(1, 32) },
        0,
    >,
}

use super::constants::*;
use super::linear_layer::LinearLayer;
use crate::misc::ceil_to_multiple;

pub struct LayerStack {
    pub l1_base: LinearLayer<
        L1_BASE_INPUT_DIMS,
        L1_BASE_OUTPUT_DIMS,
        L1_BASE_PADDED_INPUT_DIMS,
        L1_BASE_PADDED_OUTPUT_DIMS,
        L1_BASE_NUM_REGS,
    >,
    pub l1_pa: LinearLayer<
        L1_PA_INPUT_DIMS,
        L1_PA_OUTPUT_DIMS,
        L1_PA_PADDED_INPUT_DIMS,
        L1_PA_PADDED_OUTPUT_DIMS,
        L1_PA_NUM_REGS,
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

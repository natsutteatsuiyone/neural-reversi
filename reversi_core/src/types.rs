pub type Depth = u32;
pub type Score = i32;
pub type Scoref = f32;
pub type Selectivity = u8;

pub struct NonPV;
pub struct PV;
pub struct Root;
pub trait NodeType {
    const PV_NODE: bool;
    const ROOT_NODE: bool;
    const TYPE_ID: u32;
}

impl NodeType for NonPV {
    const PV_NODE: bool = false;
    const ROOT_NODE: bool = false;
    const TYPE_ID: u32 = 1;
}

impl NodeType for PV {
    const PV_NODE: bool = true;
    const ROOT_NODE: bool = false;
    const TYPE_ID: u32 = 2;
}

impl NodeType for Root {
    const PV_NODE: bool = true;
    const ROOT_NODE: bool = true;
    const TYPE_ID: u32 = 3;
}

//! Node type definitions for the alpha-beta search algorithm.

/// Runtime identifier for dispatching node-type-specialized search code.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum NodeTypeId {
    NonPv,
    Pv,
    Root,
}

/// Non-PV (non-principal variation) node type.
///
/// These are nodes that are not part of the principal variation and can be searched
/// with zero-width windows for more efficient pruning.
pub struct NonPV;

/// PV (principal variation) node type.
///
/// These nodes are part of the principal variation and require full-width alpha-beta
/// windows to find the best move.
pub struct PV;

/// Root node type.
pub struct Root;

/// Trait for compile-time node type specialization.
pub trait NodeType {
    /// Whether this is a PV node (true for PV and Root nodes).
    const PV_NODE: bool;
    /// Whether this is the root node.
    const ROOT_NODE: bool;
    /// Runtime identifier for dispatching node-type-specialized code paths.
    const ID: NodeTypeId;
}

impl NodeType for NonPV {
    const PV_NODE: bool = false;
    const ROOT_NODE: bool = false;
    const ID: NodeTypeId = NodeTypeId::NonPv;
}

impl NodeType for PV {
    const PV_NODE: bool = true;
    const ROOT_NODE: bool = false;
    const ID: NodeTypeId = NodeTypeId::Pv;
}

impl NodeType for Root {
    const PV_NODE: bool = true;
    const ROOT_NODE: bool = true;
    const ID: NodeTypeId = NodeTypeId::Root;
}

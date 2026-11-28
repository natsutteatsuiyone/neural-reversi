//! Embedded zstd-compressed neural network weights for `reversi-core`.

macro_rules! eval_main_weights_literal {
    () => {
        "eval-e6bbc4f6.zst"
    };
}

macro_rules! eval_small_weights_literal {
    () => {
        "eval_sm-e6bbc4f6.zst"
    };
}

/// Filename for the main neural network weights (zstd compressed).
pub const MAIN_FILE_NAME: &str = eval_main_weights_literal!();

/// Filename for the small neural network weights (zstd compressed).
pub const SMALL_FILE_NAME: &str = eval_small_weights_literal!();

/// Returns the embedded zstd-compressed main neural network weights.
#[must_use]
pub fn main_bytes() -> &'static [u8] {
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../",
        eval_main_weights_literal!()
    ))
}

/// Returns the embedded zstd-compressed small neural network weights.
#[must_use]
pub fn small_bytes() -> &'static [u8] {
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../",
        eval_small_weights_literal!()
    ))
}

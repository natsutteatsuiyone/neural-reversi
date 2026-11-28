//! Embedded zstd-compressed neural network weights for `reversi-core`.
//!
//! Release builds embed the weight files via `include_bytes!`. Debug and test
//! builds (`debug_assertions`) load them lazily from the workspace root so
//! every test and dev binary does not link ~250 MiB of weight data.

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

#[cfg(debug_assertions)]
fn read_workspace_weights(path: &str) -> Box<[u8]> {
    std::fs::read(path)
        .unwrap_or_else(|err| panic!("failed to read weight file \"{path}\": {err}"))
        .into_boxed_slice()
}

/// Returns the zstd-compressed main neural network weights.
#[must_use]
pub fn main_bytes() -> &'static [u8] {
    #[cfg(not(debug_assertions))]
    {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../",
            eval_main_weights_literal!()
        ))
    }

    #[cfg(debug_assertions)]
    {
        static BYTES: std::sync::OnceLock<Box<[u8]>> = std::sync::OnceLock::new();
        BYTES.get_or_init(|| {
            read_workspace_weights(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../",
                eval_main_weights_literal!()
            ))
        })
    }
}

/// Returns the zstd-compressed small neural network weights.
#[must_use]
pub fn small_bytes() -> &'static [u8] {
    #[cfg(not(debug_assertions))]
    {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../",
            eval_small_weights_literal!()
        ))
    }

    #[cfg(debug_assertions)]
    {
        static BYTES: std::sync::OnceLock<Box<[u8]>> = std::sync::OnceLock::new();
        BYTES.get_or_init(|| {
            read_workspace_weights(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../",
                eval_small_weights_literal!()
            ))
        })
    }
}

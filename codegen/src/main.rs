//! Code generation utility for reversi pattern feature lookup tables.
//!
//! This tool generates two static lookup tables used by the neural network evaluator:
//! - EVAL_FEATURE: Maps board positions to feature indices
//! - EVAL_X2F: Maps squares to their associated features

use reversi_core::eval::pattern_feature::{FeatureToCoordinate, EVAL_F2X};
use reversi_core::square::Square as Sq;

fn main() {
    generate_EVAL_FEATURE();
    println!();
    generate_EVAL_X2F();
}

/// Generates EVAL_FEATURE lookup table that maps each board square to its feature indices.
///
/// For each of the 64 squares, this computes indices for all pattern features
/// that include that square. The generated table is used for fast feature extraction
/// during board evaluation.
#[allow(non_snake_case)]
fn generate_EVAL_FEATURE() {
    // Collect all feature definitions and pad to 32 features
    let mut feature_defs = EVAL_F2X.iter().collect::<Vec<_>>();
    let len = feature_defs.len();
    // Pad with empty features to reach 32 total
    for _i in len..32 {
        feature_defs.push(&FeatureToCoordinate {
            n_square: 0,
            squares: [Sq::None; 10],
        });
    }

    println!("#[rustfmt::skip]");
    println!("const EVAL_FEATURE: [Feature; 64] = [");
    // Process each square on the board
    for i in 0..64 {
        // Create bitboard with only this square set
        let board: u64 = 1u64 << i;
        // Compute feature index for this square in each pattern
        let feature_indices: Vec<u32> = feature_defs
            .iter()
            .map(|feature| compute_feature_index(board, feature))
            .collect();

        let indices_str = feature_indices
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");

        println!("    PatternFeature {{ v1: [{}] }},", indices_str);
    }
    println!("];");
}

/// Generates EVAL_X2F lookup table that maps each square to its associated features.
///
/// This is the reverse mapping of EVAL_FEATURE. For each square, it lists which
/// features include that square and what index within that feature the square occupies.
/// Used for incremental feature updates during move generation.
#[allow(non_snake_case)]
fn generate_EVAL_X2F() {
    const MAX_FEATURES_PER_SQUARE: usize = 4;

    println!("#[rustfmt::skip]");
    println!("static EVAL_X2F: [CoordinateToFeature; 64] = [");
    // Process each square
    for i in 0..64 {
        let board = 1 << i;
        print!("    ctf!(");
        let mut n_feature = 0;
        let mut features = String::new();
        // Find all features that include this square
        for (feature_i, ftc) in EVAL_F2X.iter().enumerate() {
            let feature_value = compute_feature_index(board, ftc);
            if feature_value > 0 {
                n_feature += 1;
                // Store [feature_index, position_within_feature]
                features.push_str(&format!("[{}, {}],", feature_i, feature_value));
            }
        }
        // Pad with empty entries
        for _ in 0..(MAX_FEATURES_PER_SQUARE - n_feature) {
            features.push_str("[0, 0],");
        }
        print!("{}, [{}]", n_feature, features);
        println!("),");
    }
    println!("];");
}

/// Computes the feature index for a given board position within a pattern feature.
///
/// Each pattern feature represents a specific configuration of squares on the board.
/// This function calculates an index based on which square in the pattern is occupied.
/// The index uses base-3 encoding where each square can be empty (0), have player 1's
/// piece (1), or have player 2's piece (2).
///
/// # Arguments
/// * `board` - Bitboard with a single bit set representing the square to check
/// * `feature` - Pattern feature definition containing the squares in this pattern
///
/// # Returns
/// The computed feature index, or 0 if the square is not part of this pattern
fn compute_feature_index(board: u64, feature: &FeatureToCoordinate) -> u32 {
    let mut multiplier = 0;
    let mut feature_index = 0;

    // Process squares in reverse order to match feature encoding
    for square in feature.squares.iter().rev() {
        if *square == Sq::None {
            continue;
        }

        // Update multiplier for base-3 encoding
        multiplier = if multiplier == 0 { 1 } else { multiplier * 3 };

        // If this is the square we're checking, record its position
        if board & square.bitboard() != 0 {
            feature_index = multiplier;
        }
    }
    feature_index
}

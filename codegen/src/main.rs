use reversi_core::eval::pattern_feature::{FeatureToCoordinate, EVAL_F2X};
use reversi_core::square::Square as Sq;

fn main() {
    generate_EVAL_FEATURE();
    println!();
    generate_EVAL_X2F();
}

#[allow(non_snake_case)]
fn generate_EVAL_FEATURE() {
    let mut feature_defs = EVAL_F2X.iter().collect::<Vec<_>>();
    let len = feature_defs.len();
    for _i in len..16 {
        feature_defs.push(&FeatureToCoordinate {
            n_square: 0,
            squares: [Sq::None; 10],
        });
    }

    println!("#[rustfmt::skip]");
    println!("const EVAL_FEATURE: [Feature; 64] = [");
    for i in 0..64 {
        let board: u64 = 1u64 << i;
        let feature_indices: Vec<u32> = feature_defs
            .iter()
            .map(|feature| compute_feature_index(board, feature))
            .collect();

        let indices_str = feature_indices
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");

        println!("    Feature {{ v1: [{}] }},", indices_str);
    }
    println!("];");
}

#[allow(non_snake_case)]
fn generate_EVAL_X2F() {
    const MAX_FEATURES_PER_SQUARE: usize = 4;

    println!("#[rustfmt::skip]");
    println!("static EVAL_X2F: [CoordinateToFeature; 64] = [");
    for i in 0..64 {
        let board = 1 << i;
        print!("    ctf!(");
        let mut n_feature = 0;
        let mut features = String::new();
        for (feature_i, ftc) in EVAL_F2X.iter().enumerate() {
            let feature_value = compute_feature_index(board, ftc);
            if feature_value > 0 {
                n_feature += 1;
                features.push_str(&format!("[{}, {}],", feature_i, feature_value));
            }
        }
        for _ in 0..(MAX_FEATURES_PER_SQUARE - n_feature) {
            features.push_str("[0, 0],");
        }
        print!("{}, [{}]", n_feature, features);
        println!("),");
    }
    println!("];");
}

fn compute_feature_index(board: u64, feature: &FeatureToCoordinate) -> u32 {
    let mut multiplier = 0;
    let mut feature_index = 0;

    for square in feature.squares.iter().rev() {
        if *square == Sq::None {
            continue;
        }

        multiplier = if multiplier == 0 { 1 } else { multiplier * 3 };

        if board & square.bitboard() != 0 {
            feature_index = multiplier;
        }
    }
    feature_index
}

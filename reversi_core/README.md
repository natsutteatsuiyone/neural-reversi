# Reversi Core

The core engine for the neural-reversi project, implementing a reversi (Othello) AI with neural network evaluation and search algorithms.

## Overview

This crate provides the fundamental components for reversi game logic, AI evaluation, and search algorithms. It features:

- **Bitboard-based board representation** for efficient move generation
- **Neural network evaluation** with pattern-based features
- **Search algorithms** including alpha-beta pruning, endgame solving, and parallel search

## Pattern Feature Generation

The crate uses compile-time const functions to generate lookup tables for efficient pattern feature computation. The const functions in `src/eval/pattern_feature.rs` create two critical lookup tables:

### EVAL_FEATURE Table

**Purpose**: Maps each board square to its contribution across all pattern features.

**Structure**: `[PatternFeature; 64]` where each entry contains a 32-element array  of pattern indices.

**Generated by**:

```rust
const EVAL_FEATURE: [PatternFeature; BOARD_SQUARES] = generate_eval_feature();
```

**Function**: For each of the 64 board squares, this table contains the power-of-3 weight that the square contributes to each of the pattern features. This enables incremental updates when pieces are placed or flipped during move generation.

**Example**: Square A1 might contribute:

- Pattern 0 (corner): weight 2187 (3^7)
- Pattern 8 (diagonal): weight 2187 (3^7)  
- Pattern 10 (edge): weight 2187 (3^7)
- Pattern 12 (file): weight 2187 (3^7)
- All other patterns: weight 0

### EVAL_X2F Table

**Purpose**: Provides reverse mapping from board squares to the pattern features they participate in.

**Structure**: `[CoordinateToFeature; 64]` where each entry lists which patterns include that square.

**Generated by**:

```rust
static EVAL_X2F: [CoordinateToFeature; BOARD_SQUARES] = generate_eval_x2f();
```

**Function**: For each board square, this table contains:

- `n_features`: Number of patterns this square participates in
- `features`: Array of [feature_index, power_of_3] pairs

**Example**: Square A1 participates in:

- Feature 0 with weight 2187
- Feature 8 with weight 2187
- Feature 10 with weight 2187
- Feature 12 with weight 2187

### Pattern Encoding

The system uses **ternary (base-3) encoding** for board positions:

- `0` = Current player's piece
- `1` = Opponent's piece  
- `2` = Empty square

Each pattern covers specific board squares and can represent 3^8 = 6561 possible configurations for 8-square patterns.

### Compile-Time Generation

The lookup tables are generated at compile time using const functions:

```rust
const fn compute_pattern_feature_index(board: u64, feature: &FeatureToCoordinate) -> u32 {
    // Computes the positional weight (power of 3) for a square within a pattern
    // Uses ternary encoding with base-3 arithmetic
}
```

### Generated Code Examples

The const functions produce the following lookup tables at compile time.

#### EVAL_FEATURE Example Output

```rust
const EVAL_FEATURE: [PatternFeature; 64] = [
    // A1
    PatternFeature { v1: [2187, 0, 0, 0, 0, 0, 0, 0, 2187, 0, 2187, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    // B1
    PatternFeature { v1: [729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 729, 0, 0, 0, 2187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    // C1
    PatternFeature { v1: [243, 0, 0, 0, 0, 0, 0, 0, 0, 0, 243, 0, 0, 0, 729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    // D1
    PatternFeature { v1: [81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 243, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
    ... // Additional entries for each square
];
```

#### EVAL_X2F Example Output

```rust
static EVAL_X2F: [CoordinateToFeature; 64] = [
    // A1
    CoordinateToFeature {
        n_features: 4,
        features: [[0, 2187], [8, 2187], [10, 2187], [12, 2187]]
    },
    // B1
    CoordinateToFeature {
        n_features: 3,
        features: [[0, 729], [10, 729], [14, 2187], [0, 0]]
    },
    // C1
    CoordinateToFeature {
        n_features: 3,
        features: [[0, 243], [10, 243], [14, 729], [0, 0]]
    },
    // D1
    CoordinateToFeature {
        n_features: 4,
        features: [[0, 81], [10, 81], [14, 243], [15, 81]]
    },
    ... // Additional entries for each square
];
```

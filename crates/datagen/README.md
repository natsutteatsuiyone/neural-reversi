# datagen

A tool for generating and processing Reversi AI neural network training data.

## Commands

### selfplay

Generates game data through AI self-play for neural network training. The self-play process works by having the AI play against itself. It can either generate a specified number of games or play through a list of predefined opening sequences. Key characteristics include:

- Early game moves (first 10-49 plies) are selected randomly to ensure diversity in the training data when not using predefined openings.
- Subsequent moves use the AI search algorithm to find optimal plays
- Each position is recorded with evaluation scores and game outcome information

```bash
datagen selfplay --games 100000 --hash-size 128 --mid-depth 12 --end-depth 21 --prefix game --output-dir ./data
```

To use predefined openings:

```bash
datagen selfplay --openings openings.txt --resume --hash-size 128 --mid-depth 12 --end-depth 21 --prefix game --output-dir ./data
```

#### Options

- `--games`: Number of games to generate (ignored if `--openings` is used). Default: 100,000,000.
- `--games-per-file`: Number of games to store in each output file (1-65,535; default: 10,000).
- `--hash-size`: Transposition table size in MB for the search algorithm (default: 512).
- `--mid-depth`: Midgame search depth (1-60; default: 12).
- `--end-depth`: Endgame search depth. Single value for all selectivities, or 5 comma-separated values for per-selectivity configuration (Level1,Level2,Level3,Level4,None) (default: 21).
- `--prefix`: Output file prefix for generated data files (default: machine hostname).
- `--output-dir`: Required output directory where game data will be stored.
- `--openings`: Optional path to a file containing opening sequences. If provided, selfplay will iterate through these openings instead of generating a set number of games.
- `--resume`: Resume selfplay from the last processed opening in the `--openings` file. Requires `--openings` to be set (default: false).

#### Data format

Each board position is a 27-byte record. Multi-byte values use little-endian byte order, and
the field order matters — see `src/record.rs::write_records`.

| Offset | Field | Type |
|--------|-------|------|
| 0-7 | Player bitboard | u64 LE |
| 8-15 | Opponent bitboard | u64 LE |
| 16-19 | Evaluation score | f32 LE |
| 20 | Game score | i8 |
| 21 | Ply | u8 |
| 22 | Random-move flag | u8 |
| 23 | Best move (square index) | u8 |
| 24 | Side to move (0 = Black, 1 = White) | u8 |
| 25-26 | Game id | u16 LE |

### opening

Generates all possible Reversi opening sequences of exactly a specified depth, starting with F5 as the first move.

```bash
datagen opening --depth 9 > openings.txt
```

#### Options

- `--depth`: Maximum number of moves to include in the sequences (required, 1-20)

### probcut

Generates training data for calculating ProbCut parameters. This command analyzes game positions with multiple search depths to create correlation data between shallow and deep search results.

```bash
datagen probcut --input ./games.txt --output ./probcut_training_data.csv
```

#### Options

- `--input`: Input file containing game sequences (one move sequence per line, moves in algebraic notation like "f5d6c3")
- `--output`: Output CSV file containing the training samples

#### Data format

Midgame CSV format has the following columns:

- `game`: One-based line number of the source game
- `ply`: Move number in the game (0-59)
- `shallow_depth`: Search depth for shallow search
- `shallow_score`: Score of the shallow search, in discs
- `deep_depth`: Search depth for deep search
- `deep_score`: Score of the deep search, in discs
- `diff`: Score difference between deep and shallow search results

### fit-probcut

Fits the midgame ProbCut parameters from a CSV produced by `probcut` and writes
the Rust `PROBCUT_PARAMS` block to stdout. Diagnostics go to stderr, so the
source block can be redirected independently.

```bash
datagen fit-probcut --input ./probcut_training_data.csv > probcut_params.rs
```

CSVs without the `game` column are rejected; regenerate them with the current
`probcut` command.

### Endgame ProbCut

Endgame ProbCut is fitted on positions traced from the engine's own search
tree. Each position is labeled by an exact solve, so any game records can
supply the roots.

- Games are split into five folds by `game_fold`. Fold 4 is never used for
  fitting, only for evaluation.
- The ProbCut screen and verification search use the main network (the rest
  of the endgame search uses the small one). All sampled scores therefore
  come from the main network.

```bash
# Tree samples from the fitting folds (needs the `trace` feature).
cargo run -p datagen --release --features trace -- endgame-trace     -i games.txt -o trace-train.csv --fit-folds     --roots 1500 --root-min-empties 20 --root-max-empties 30     --one-in 256 --max-per-empties 20000
# The same from the holdout fold, for validation.
cargo run -p datagen --release --features trace -- endgame-trace     -i games.txt -o trace-holdout.csv     --roots 500 --root-min-empties 20 --root-max-empties 30     --one-in 256 --max-per-empties 20000
# Fit, report calibration, and print the generated block.
cargo run -p datagen --release -- fit-endgame-probcut     -i trace-train.csv --validate trace-holdout.csv > endgame_models.rs
```

Then replace the generated block in
`crates/reversi-core/src/probcut/endgame.rs` (from `// Generated by` up to
the tests) with `endgame_models.rs`. The block records the main-network
weights it was fitted for. A unit test fails when the engine's weights
change, until the models are refitted.

#### Subcommands

- `endgame-trace`: runs Level1 to Level4 solves from game positions and
  records the ProbCut attempts whose position hash is divisible by
  `--one-in`.
  - Labeling: positions with 24 or more empties are solved one at a time
    with every thread; the rest are solved in parallel, one thread each.
  - Output: a `# main_weights=` line, then one row per attempt: static
    evaluation (`s0`), verification score (`s2`), exact score, beta, level,
    and node type.
  - Check: it replays each cut decision from the sampled scores and reports
    how many differ from the engine's. Any difference means the samples do
    not match what the search saw.
- `fit-endgame-probcut`: fits the verification model and the screen margins.
  - Verification model: `exact − s2`. Its mean and log scale are linear in
    the empty count and parity, node type, `s0 − beta`, and beta: values
    known before the verification, so the cut threshold has a closed form.
  - Screen margins: the upper quantiles of `s2 − s0` at `SCREEN_RATES`, one
    per node type.

  Diagnostics go to stderr: cut decisions of every level replayed on the
  held-out games and, with `--validate`, on another trace. The Rust block
  goes to stdout.
- `endgame-holdout`: writes one position per game as OBF, scored with the
  game's move and result, for `evaltest`. Options: `--min-empties`,
  `--max-empties`, and `--fit-folds` (draw from the fitting folds instead of
  fold 4). The score is exact only for endgame-corrected games; otherwise run
  `evaltest` with `--solve-reference`.

#### Calibrating the tail schedule

`TAIL_SCHEDULE` in `endgame.rs` is calibrated so that each level's root
score is exact with its stated probability (73%, 86%, 95%, 99%). A refit
changes the scale of `sigma`, so recheck the schedule after each refit. To
check a schedule:

1. Export positions with `endgame-holdout`.
2. Label them once with a full `evaltest --solve-reference` run. Its `Score`
   column is the exact score.
3. Check each candidate schedule with `evaltest --selective-only`. It stops
   before the exact solve and judges the levels against the labels, so it is
   fast.

### shuffle

Shuffles and redistributes game records from input files into a new set of output files. This is useful for preparing training data by randomizing the order of game records and potentially splitting them into a different number of files.

```bash
datagen shuffle --input-dir ./data --output-dir ./shuffled_data --pattern "*.bin" --files-per-chunk 10 --num-output-files 50
```

To filter out lower-quality records while shuffling:

```bash
datagen shuffle --input-dir ./data --output-dir ./filtered_data --min-ply 8 --max-score-diff 12 --drop-random --keep-above-ply 50
```

By default, the command refuses to run when the output directory already contains any
`shuffled_*.bin` files. To add newly shuffled inputs to an existing dataset, opt in with
`--append`:

```bash
datagen shuffle --input-dir ./more_data --output-dir ./shuffled_data --append
```

Append mode validates every existing shard before writing. Names must use the canonical
`shuffled_00000.bin` form, shard sizes must be multiples of the 27-byte record size, and shard
indices must fit the resolved output layout. Existing data is never truncated, rebalanced, or
rewritten.

When `--append` is used without `--num-output-files`, the layout is inferred as one more than the
highest existing shard index. This preserves the previous layout on repeated invocations. An
explicit `--num-output-files` overrides inference, so provide the existing layout's matching file
count when overriding it; append is rejected if any existing shard index falls outside that count.

#### Options

- `--input-dir`: Input directory containing the game data files to be shuffled.
- `--output-dir`: Output directory where the shuffled game data files will be stored. This directory will be created if it doesn't exist.
- `--pattern`: Glob pattern to match input files within the `input-dir` (default: "*.bin").
- `--files-per-chunk`: Number of input files to read and shuffle in memory at a time (default: 10). Adjust based on available memory and the size of your input files.
- `--num-output-files`: Optional number of output files to create. Fresh runs default to the number of input files. Append runs with existing shards infer the count from the highest existing index unless this option is provided.
- `--append`: Validate existing `shuffled_*.bin` files and append new records to them. Without this flag, existing shuffle output causes the command to fail.
- `--min-ply`: Drop records from earlier plies than this threshold. Useful for excluding highly unstable opening positions (default: 0).
- `--max-score-diff`: Drop records where the absolute difference between the stored evaluation score and the final game score exceeds this threshold. Records with unavailable game scores are kept.
- `--drop-random`: Drop records whose move was selected randomly during self-play instead of by search.
- `--keep-above-ply`: Keep all records with ply >= this value, bypassing `--drop-random` and `--max-score-diff` filters. Useful for preserving high-quality endgame solver results unconditionally. Note that `--min-ply` still applies independently.

Filtering is applied while reading the serialized records, so large datasets can be filtered without fully deserializing every record into an intermediate structure. The shuffle summary reports how many records were dropped by each filter, how many records and bytes were added by the current invocation, and the post-append totals for the dataset and each shard.

### score-openings

Enumerates every unique board position reachable within a given number of plies from the initial position and scores each one with the search algorithm. Symmetric positions are canonicalized via board uniqueness, so each equivalence class is evaluated only once. If the output file already exists, previously scored positions are loaded and skipped so the command can be resumed.

```bash
datagen score-openings --depth 9 --mid-depth 16 --end-depth 24 --hash-size 512 --output ./openings_scored.bin
```

#### Options

- `--depth`: Number of plies to enumerate from the initial position (1-20).
- `--hash-size`: Transposition table size in MB (default: 512).
- `--mid-depth`: Midgame search depth (1-60, default: 16).
- `--end-depth`: Endgame search depth. Single value for all selectivities, or 5 comma-separated values for per-selectivity configuration (Level1,Level2,Level3,Level4,None) (default: 24).
- `--output`: Output file path where scored positions are written.

#### Data format

Same binary record format as `selfplay`. The `game_score` field stores the rounded evaluation score (since no full game is played), and the random move flag is always 0.

### overwrite-scores

Overwrites the `score`, `game_score`, and `is_random` fields of records inside binary data files based on a source binary file. Matching is performed via `Board::unique()`, so symmetric variants of the same position are treated as equal. All other fields (`game_id`, `ply`, `side_to_move`, `sq`, the player/opponent bitboards) are preserved.

```bash
datagen overwrite-scores --source ./rescored.bin --target-dir ./data --pattern "*.bin"
```

#### Options

- `--source`: Source binary file (same format as `selfplay`). Each record contributes a `(Board::unique(), patch)` entry to the lookup table.
- `--target-dir`: Directory containing files to be updated in place.
- `--pattern`: Glob pattern relative to `--target-dir` (default: `*.bin`).

Records in target files whose canonical board is absent from the source are left unchanged. Files are rewritten via a `.tmp` sibling and renamed only after a successful write.

## Workflow

1. Generate self-play data
2. Train neural network using self-play data

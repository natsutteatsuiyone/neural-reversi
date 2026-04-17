# datagen

A tool for generating and processing Reversi AI neural network training data.

## Commands

### selfplay

Generates game data through AI self-play for neural network training. The self-play process works by having the AI play against itself. It can either generate a specified number of games or play through a list of predefined opening sequences. Key characteristics include:

- Early game moves (first 10-30 plies) are selected randomly to ensure diversity in the training data when not using predefined openings.
- Subsequent moves use the AI search algorithm to find optimal plays
- Each position is recorded with evaluation scores and game outcome information

```bash
datagen selfplay --games 100000 --hash-size 128 --mid-depth 12 --end-depth 21 --selectivity 0 --prefix game --output-dir ./data
```

To use predefined openings:

```bash
datagen selfplay --openings openings.txt --resume --hash-size 128 --mid-depth 12 --end-depth 21 --selectivity 0 --prefix game --output-dir ./data
```

#### Options

- `--games`: Number of games to generate (ignored if `--openings` is used). Default: 100,000,000.
- `--records_per_file`: Number of records to store in each output file (default: 1,000,000)
- `--hash-size`: Transposition table size in MB for the search algorithm (default: 128)
- `--mid-depth`: Midgame search depth (1-60, default: 12)
- `--end-depth`: Endgame search depth. Single value for all selectivities, or 4 comma-separated values for per-selectivity configuration (Level1,Level3,Level5,None) (default: 21)
- `--selectivity`: Search selectivity parameter controlling move pruning (0: 73%, 1: 87%, 2: 95%, 3: 98%, 4: 99%, 5: 100%) (default: 0)
- `--prefix`: Output file prefix for generated data files (default: "game")
- `--output-dir`: Output directory where game data will be stored
- `--openings`: Optional path to a file containing opening sequences. If provided, selfplay will iterate through these openings instead of generating a set number of games.
- `--resume`: Resume selfplay from the last processed opening in the `--openings` file. Requires `--openings` to be set. (default: false)

#### Data format

Binary format with the following information for each board position:

- Player and opponent bitboards (u64 x 2) - representing the current board state
- Evaluation score (f32) - the position evaluation from the search algorithm
- Game score (i8) - the final game outcome (e.g., disc difference) from the current player's perspective, stored as an 8-bit integer.
- Ply (u8) - the move number in the game (0-60)
- Random move flag (u8) - indicates whether this position resulted from a random move (1) or AI search (0)
- Best move (u8) - the square index (0-63) of the move made from this position.

### opening

Generates all possible Reversi opening sequences up to a specified depth, starting with F5 as the first move.

```bash
datagen opening --depth 9 > openings.txt
```

#### Options

- `--depth`: Maximum number of moves to include in the sequences (default: 8)

### probcut

Generates training data for calculating ProbCut parameters. This command analyzes game positions with multiple search depths to create correlation data between shallow and deep search results.

```bash
datagen probcut --input ./games.txt --output ./probcut_training_data.csv
```

#### Options

- `--input`: Input file containing game sequences (one move sequence per line, moves in algebraic notation like "f5d6c3")
- `--output`: Output CSV file containing training data with columns: ply, shallow_depth, deep_depth, diff

#### Data format

CSV format with the following columns:

- `ply`: Move number in the game (0-59)
- `shallow_depth`: Search depth for shallow search
- `deep_depth`: Search depth for deep search  
- `diff`: Score difference between deep and shallow search results

### shuffle

Shuffles and redistributes game records from input files into a new set of output files. This is useful for preparing training data by randomizing the order of game records and potentially splitting them into a different number of files.

```bash
datagen shuffle --input-dir ./data --output-dir ./shuffled_data --pattern "*.bin" --files-per-chunk 10 --num-output-files 50
```

To filter out lower-quality records while shuffling:

```bash
datagen shuffle --input-dir ./data --output-dir ./filtered_data --min-ply 8 --max-score-diff 12 --drop-random --keep-above-ply 50
```

#### Options

- `--input-dir`: Input directory containing the game data files to be shuffled.
- `--output-dir`: Output directory where the shuffled game data files will be stored. This directory will be created if it doesn't exist.
- `--pattern`: Glob pattern to match input files within the `input-dir` (default: "*.bin").
- `--files-per-chunk`: Number of input files to read and shuffle in memory at a time (default: 10). Adjust based on available memory and the size of your input files.
- `--num-output-files`: Optional number of output files to create. If not specified, it defaults to the number of input files. The shuffled records will be distributed among these output files.
- `--min-ply`: Drop records from earlier plies than this threshold. Useful for excluding highly unstable opening positions (default: 0).
- `--max-score-diff`: Drop records where the absolute difference between the stored evaluation score and the final game score exceeds this threshold. Records with unavailable game scores are kept.
- `--drop-random`: Drop records whose move was selected randomly during self-play instead of by search.
- `--keep-above-ply`: Keep all records with ply >= this value, bypassing `--drop-random` and `--max-score-diff` filters. Useful for preserving high-quality endgame solver results unconditionally. Note that `--min-ply` still applies independently.

Filtering is applied while reading the serialized records, so large datasets can be filtered without fully deserializing every record into an intermediate structure. The shuffle summary reports how many records were dropped by each filter.

### score-openings

Enumerates every unique board position reachable within a given number of plies from the initial position and scores each one with the search algorithm. Symmetric positions are canonicalized via board uniqueness, so each equivalence class is evaluated only once. If the output file already exists, previously scored positions are loaded and skipped so the command can be resumed.

```bash
datagen score-openings --depth 9 --mid-depth 16 --end-depth 24 --selectivity 0 --hash-size 512 --output ./openings_scored.bin
```

#### Options

- `--depth`: Number of plies to enumerate from the initial position (1-20).
- `--hash-size`: Transposition table size in MB (default: 512).
- `--mid-depth`: Midgame search depth (1-60, default: 16).
- `--end-depth`: Endgame search depth. Single value for all selectivities, or 4 comma-separated values for per-selectivity configuration (Level1,Level3,Level5,None) (default: 24).
- `--selectivity`: Search selectivity parameter (0: 73%, 1: 87%, 2: 95%, 3: 98%, 4: 99%, 5: 100%) (default: 0).
- `--output`: Output file path where scored positions are written.

#### Data format

Same binary record format as `selfplay`. The `game_score` field stores the rounded evaluation score (since no full game is played), and the random move flag is always 0.

## Workflow

1. Generate self-play data
2. Train neural network using self-play data

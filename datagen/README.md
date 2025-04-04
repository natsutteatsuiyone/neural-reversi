# datagen

A tool for generating and processing Reversi AI neural network training data.

## Commands

### selfplay

Generates game data through AI self-play for neural network training. The self-play process works by having the AI play against itself, with the following characteristics:

- Early game moves (first 10-30 plies) are selected randomly to ensure diversity in the training data
- Subsequent moves use the AI search algorithm to find optimal plays
- Each position is recorded with evaluation scores and game outcome information

```bash
datagen selfplay --games 100000 --hash-size 128 --level 12 --selectivity 1 --prefix game --output-dir ./data
```

#### Options

- `--games`: Number of games to generate
- `--records_per_file`: Number of records to store in each output file (default: 1,000,000)
- `--hash-size`: Transposition table size in MB for the search algorithm
- `--level`: Search depth level - higher values result in stronger play but slower generation
- `--selectivity`: Search selectivity parameter controlling move pruning (1: 73%, 2: 87%, 3: 95%, 4: 98%, 5: 99%, 6: 100%)
- `--prefix`: Output file prefix for generated data files
- `--output-dir`: Output directory where game data will be stored

#### Data format

Binary format with the following information for each board position:

- Player and opponent bitboards (u64 x 2) - representing the current board state
- Evaluation score (f32) - the position evaluation from the search algorithm
- Game score (f32) - the final game outcome from the current player's perspective
- Ply (u8) - the move number in the game (0-60)
- Random move flag (u8) - indicates whether this position resulted from a random move (1) or AI search (0)

### feature

Extracts neural network training features from self-play data.

```bash
datagen feature --input-dir ./data --output-dir ./features --threads 8
```

#### Options

- `--input-dir`: Input directory with self-play data
- `--output-dir`: Output directory for feature data
- `--threads`: Number of threads to use for feature extraction

#### Data format

zstd compressed format with:

- Score (f32) - neural network teacher signal
- Pattern features (u16 array) - neural network input features
- Mobility (u8) - number of legal moves for the current player
- Ply (u8) - can be used for training weights

### probcut

Calculates ProbCut parameters.

```bash
datagen probcut --input ./games.txt --output ./probcut_params.csv
```

#### Options

- `--input`: Input file (game records)
- `--output`: Output file (CSV format)

## Workflow

1. Generate self-play data
2. Extract features from self-play data
3. Train neural network using extracted features

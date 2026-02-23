# Evaluation Test Suite

Evaluation test suite runner for measuring the endgame search performance of Reversi (Othello) AI engines. Test positions are loaded from OBF (Othello Board Format) files in the `problem/` directory at the project root.

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--depth` or `-d` | Maximum search depth in plies | 60 |
| `--selectivity` | Search selectivity level: 0: 73% (fastest, less accurate) 1: 87% 2: 95% 3: 98% 4: 99% 5: 100% (complete search) | 0 |
| `--hash-size` | Transposition table size in MB | 512 |
| `--threads` | Number of parallel search threads | System default |
| `--problem` | Problem set to run: preset name or `.obf` file path. Repeatable. | All `.obf` files in problem directory |
| `--problem-dir` | Path to the directory containing `.obf` problem files | Auto-discovered |

### Presets

The `--problem` option accepts the following preset names:

| Preset | Description |
|--------|-------------|
| `fforum` | Loads all FFO Forum files (`fforum-1-19.obf` through `fforum-60-79.obf`) |
| `hard-20` | Loads `hard-20.obf` |
| `hard-25` | Loads `hard-25.obf` |
| `hard-30` | Loads `hard-30.obf` |

You can also pass a direct path to any `.obf` file.

### Problem Directory Discovery

The problem directory is located automatically by checking, in order:

1. A `problem/` directory next to the executable
2. A `problem/` directory in the current working directory
3. The path specified by the `EVALTEST_PROBLEM_DIR` environment variable

Use `--problem-dir` to override this search.

## Examples

### Run all problem files with default settings

```bash
cargo run -p evaltest --release
```

### Run with search depth and larger hash table

```bash
cargo run -p evaltest --release -- --depth 20 --hash-size 2048
```

### Run with complete search (no selectivity)

```bash
cargo run -p evaltest --release -- --selectivity 5
```

### Run a specific preset

```bash
cargo run -p evaltest --release -- --problem fforum
```

### Run multiple problem sets

```bash
cargo run -p evaltest --release -- --problem fforum --problem hard-20
```

### Run a custom OBF file

```bash
cargo run -p evaltest --release -- --problem /path/to/custom.obf
```

### Quick test with shallow search

```bash
cargo run -p evaltest --release -- --depth 8 --selectivity 0
```

## Output Format

Results are displayed per file, each with a tabular section:

- **#**: Line number within the OBF file
- **Depth**: Search depth achieved (may show percentage for selective searches)
- **Time(s)**: Time taken to solve the position
- **Nodes**: Number of positions searched
- **NPS**: Nodes per second (search speed)
- **Line**: Principal variation (best move sequence)
- **Score**: The engine's evaluated score
- **Expected**: The known optimal score and best moves

When multiple files are loaded, an overall statistics summary is printed at the end.

## Performance Metrics

After each file section, aggregate statistics are displayed:

- **Total time**: Combined solving time for all positions
- **Total nodes**: Total positions searched
- **NPS**: Average search speed
- **Best move**: Percentage of positions where the best move was found
- **Top 2/3 move**: Percentage where one of the top moves was found
- **Score accuracy**: Percentage of positions solved within various error margins
- **MAE**: Mean Absolute Error of scores
- **Std Dev**: Standard deviation of score errors

## References

- [FFO test suite](http://radagast.se/othello/ffotest.html)
- [Edax test results](https://github.com/abulmo/edax-reversi/tree/master/problem)

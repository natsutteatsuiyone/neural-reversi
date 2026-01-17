# FFO Test

FFO Test is a standard test suite for evaluating the endgame search performance of Reversi (Othello) AI engines.

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--depth` or `-d` | Maximum search depth in plies | 60 |
| `--selectivity` | Search selectivity level: 0: 73% (fastest, less accurate) 1: 87% 2: 95% 3: 98% 4: 99% 5: 100% (complete search) | 0 |
| `--hash-size` | Transposition table size in MB | 256 |
| `--threads` | Number of parallel search threads | System default |
| `--case` | Run only a specific test case (1-79) | All cases |
| `--from` | Start from test case number | 1 |
| `--to` | Run up to test case number | 79 |

## Examples

### Run all test cases with default settings

```bash
cargo run --release
```

### Run with search depth and larger hash table

```bash
cargo run --release -- --depth 20 --hash-size 2048
```

### Run with complete search (no selectivity)

```bash
cargo run --release -- --selectivity 5
```

### Run a specific test case

```bash
cargo run --release -- --case 42
```

### Run a range of cases

```bash
cargo run --release -- --from 20 --to 30
```

### Quick test with shallow search

```bash
cargo run --release -- --depth 8 --selectivity 0
```

## Output Format

The test runner displays results in a tabular format:

- **#**: Test case number
- **Depth**: Search depth achieved (may show percentage for selective searches)
- **Time(s)**: Time taken to solve the position
- **Nodes**: Number of positions searched
- **NPS**: Nodes per second (search speed)
- **Line**: Principal variation (best move sequence)
- **Score**: The engine's evaluated score
- **Expected**: The known optimal score and best moves

## Performance Metrics

After running the tests, aggregate statistics are displayed:

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

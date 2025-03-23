# FFO Test

FFO Test is a standard test suite for evaluating the endgame search performance of Reversi AI engines.

```bash
cargo run --release -- [options]
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--depth` or `-d` | Search depth | 60 |
| `--selectivity` | Search selectivity (1: 73%, 2: 87%, 3: 95%, 4: 98%, 5: 99%, 6: 100%) | 1 |
| `--hash-size` | Transposition table size (MB) | 1024 |
| `--threads` | Number of threads to use | System default |
| `--case` | Run only a specific case number | All cases |
| `--from` | Start from this number | 1 |
| `--to` | Run up to this number | 79 |

## Examples

Run all test cases:

```bash
cargo run --release
```

Run with specific depth and hash size:

```bash
cargo run --release -- --depth 10 --hash-size 2048
```

Run a specific case:

```bash
cargo run --release -- --case 42
```

Run a range of cases:

```bash
cargo run --release -- --from 20 --to 30
```

## References

- [FFO test suite](http://radagast.se/othello/ffotest.html)
- [Edax](https://github.com/abulmo/edax-reversi/tree/master/problem)

# Weight Tournament

Compare `reversi-core` main-network weights with in-process one-ply games. The
tool discovers every `*.zst` file in a directory, plays every opening twice
with colors swapped, and ranks the weights in a full round-robin tournament.

Every discovered file is loaded as a `reversi-core` main-network weight. The
one-ply player uses that network at every non-terminal position, matching the
weight comparison from the `v7-wasm` branch.

While games are running, a progress bar shows completed games, elapsed time,
and the estimated time remaining.

## Usage

```bash
cargo run -p weight-tournament --release -- <weights-dir> [options]
```

Options:

| Option | Description |
| --- | --- |
| `-o, --opening-file <FILE>` | Opening file in `match-runner` format |
| `-j, --jobs <N>` | Parallel comparisons (default: `1`) |

If no opening file is supplied, two games are played from the initial
position for each comparison. Opening files contain one compact coordinate
sequence per line, such as `f5d6c4d3`; blank lines and lines beginning with
`#` are ignored.

Examples:

```bash
cargo run -p weight-tournament --release -- ./weights
cargo run -p weight-tournament --release -- ./weights -o ./openings.txt -j 4
```

# web

WebAssembly version of Neural Reversi, running the Rust AI engine in the browser.

## Requirements

- [Rust](https://www.rust-lang.org/) (1.88.0+)
- [Bun](https://bun.sh/)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/)
- clang (for zstd compilation)

Install wasm-pack:

```bash
cargo install wasm-pack
```

Add wasm32 target:

```bash
rustup target add wasm32-unknown-unknown
```

## Build

### Development

```bash
bun install
bun run dev
```

Open http://127.0.0.1:8080 in your browser.

### Production

```bash
bun run build
```

Output will be in the `dist/` directory.

## Browser E2E Tests

Install the Playwright browser once:

```bash
bun run test:e2e:install
```

Run the browser smoke tests:

```bash
bun run build:wasm:dev # only needed when pkg/ is missing or stale
bun run test:e2e
```

Use the interactive runner while developing:

```bash
bun run test:e2e:ui
```

## Endgame Solver Benchmark

Run FFO endgame test positions against the WebAssembly engine from the terminal.

### Build

```bash
bun run build:wasm:node
```

### Usage

```bash
bun scripts/endgame-bench.js [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-p, --problem` | OBF file stem or path (default: `fforum-40-59`) |
| `-t, --tt-size` | Transposition table size in MB (default: `32`) |
| `-e, --max-empties` | Max empty squares to include (default: `24`) |
| `-h, --help` | Show help message |

### Examples

```bash
bun scripts/endgame-bench.js
bun scripts/endgame-bench.js -p fforum-60-79 -t 64
bun scripts/endgame-bench.js -p fforum-40-59 -e 20
```

## Network Forward Benchmark

Run the raw `crates/web/src/eval/network.rs` forward path from the terminal.

### Build and Run

```bash
bun run bench:network
```

### Usage

```bash
bun run build:wasm:node
bun scripts/network-bench.js [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-n, --iterations` | Timed outer iterations (default: `10000`) |
| `-w, --warmup` | Warmup outer iterations (default: `1000`) |
| `-h, --help` | Show help message |

### Examples

```bash
bun scripts/network-bench.js
bun scripts/network-bench.js --iterations 50000 --warmup 1000
```

## Weight Tournament

Estimate the strongest `.zst` file in a folder with in-process one-ply matches.
This does not use GTP or any other protocol; evaluators are loaded into the same
WebAssembly module and play directly against each other. If an opening file is
provided, it uses the same compact opening format as `crates/match-runner` and
plays each opening twice with colors swapped.

### Build

```bash
bun run build:wasm:node
```

### Usage

```bash
bun scripts/weight-tournament.js <weights-dir> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-o, --opening-file` | Opening file in match-runner format |
| `-j, --jobs` | Parallel comparisons per round. Default: 1 |
| `-r, --rounds` | Pairing rounds. Default: full round-robin for <= 8 weights, otherwise 4 |
| `--full-round-robin` | Play every pair once. Cannot be combined with `--rounds` |
| `--seed` | Stable seed for the initial pairing order |
| `-h, --help` | Show help message |

### Examples

```bash
bun scripts/weight-tournament.js <weights-dir> --opening-file <openings.txt>
bun scripts/weight-tournament.js <weights-dir> --opening-file <openings.txt> --rounds 6
bun scripts/weight-tournament.js <weights-dir> --opening-file <openings.txt> --full-round-robin --jobs 4
bun scripts/weight-tournament.js ../../weights --opening-file ../../openings.txt
```

## ProbCut Training Data Generation

Generate ProbCut training data from game sequences using the WebAssembly engine.

### Build and Run

```bash
bun run probcut -- -i <input> -o <output>
```

Options:

| Option | Description |
|--------|-------------|
| `-i, --input` | Input file containing game sequences (one per line) |
| `-o, --output` | Output CSV file for ProbCut training data |
| `-e, --endgame` | Generate endgame ProbCut data (depth-2 shallow search vs exact final result, ply >= 30 only) |
| `-h, --help` | Show help message |

### Examples

```bash
bun run probcut -- -i games.txt -o probcut.csv
bun run probcut -- -i games.txt -o probcut_endgame.csv --endgame
```

### Input Format

One game per line, moves concatenated:

```
D3C3C4C5D6E3F4E6F5E2
F5D6C3D3C4F4F6F3E6E7
```

### Output Format

CSV with columns: `ply,shallow_depth,shallow_score,deep_depth,deep_score,diff`

```csv
ply,shallow_depth,shallow_score,deep_depth,deep_score,diff
0,0,0,3,0,0
0,0,0,4,0,0
1,0,-3,3,-1,2
```

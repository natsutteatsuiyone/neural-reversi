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

## Endgame Solver Benchmark

Run FFO endgame test positions against the WebAssembly engine from the terminal.

### Build

```bash
bun run build:wasm:node
```

### Usage

```bash
bun endgame-bench.js [options]
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
bun endgame-bench.js
bun endgame-bench.js -p fforum-60-79 -t 64
bun endgame-bench.js -p fforum-40-59 -e 20
```

## ProbCut Training Data Generation

Generate ProbCut training data from game sequences using the WebAssembly engine.

### Build

```bash
bun run build:wasm:node
```

### Usage

```bash
node probcut-cli.js -i <input> -o <output>
```

Options:

| Option | Description |
|--------|-------------|
| `-i, --input` | Input file containing game sequences (one per line) |
| `-o, --output` | Output CSV file for ProbCut training data |
| `-h, --help` | Show help message |

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

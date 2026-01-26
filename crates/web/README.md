# web

WebAssembly version of Neural Reversi, running the Rust AI engine in the browser.

## Requirements

- [Rust](https://www.rust-lang.org/) (1.88.0+)
- [Node.js](https://nodejs.org/) and npm
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
npm install
npm run dev
```

Open http://127.0.0.1:8080 in your browser.

### Production

```bash
npm run build
```

Output will be in the `dist/` directory.

## ProbCut Training Data Generation

Generate ProbCut training data from game sequences using the WebAssembly engine.

### Build

```bash
npm run build:wasm:node
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

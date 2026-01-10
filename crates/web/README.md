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

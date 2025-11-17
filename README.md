# Neural Reversi

This is an experimental project to develop a high-accuracy neural network evaluation function for Reversi (Othello).

Play it online (Lite version): [https://neural-reversi.net/](https://neural-reversi.net/)

## Crates

- **reversi_core**: Core library implementing the AI search algorithms.
- **reversi_cli**: Command-line interface for playing Reversi.
- **reversi_gui**: Tauri-based graphical user interface for playing Reversi.
- **reversi_web**: WebAssembly build of the Rust engine, packaged with wasm-pack and Vite, and used as the frontend bundle for [https://neural-reversi.net](https://neural-reversi.net).
- **match_runner**: Tool for automatically running matches between Reversi engines supporting the Go Text Protocol.
- **datagen**: Tool for generating neural network training data, including self-play games and feature extraction.
- **ffotest**: FFO endgame test suite, including Edax problem sets.

## Neural Network

### Architecture

![Neural network architecture](docs/img/nn_architecture.svg)

### Features

- Mobility: Number of legal moves for the current player.
- Patterns:
  ![Pattern features](docs/img/pattern_features.svg)

### Training

[neural-reversi-training](https://github.com/natsutteatsuiyone/neural-reversi-training)

## Weight Files

Neural network weight files (`eval*.zst`, `eval_sm*.zst`, and `eval_wasm*.zst`) can be downloaded from [here](https://github.com/natsutteatsuiyone/neural-reversi-weights/releases/latest).

When developing or running the application locally (for example, using `cargo run`), place the weight files in the project root directory.

## Build

For best performance, build with native CPU optimizations. This enables CPU-specific instructions (BMI2, LZCNT, AVX2, AVX-512), which significantly improve evaluation and search speed.

### Using cargo-make

Build native-optimized binaries for both CLI and GUI using [cargo-make](https://github.com/sagiegurari/cargo-make):

```bash
# All platforms (Windows + Linux)
cargo make build-native

# Windows only
cargo make build-cli-windows-native
cargo make build-gui-windows-native

# Linux only
cargo make build-cli-linux-native
cargo make build-gui-linux-native
```

For portable release builds targeting multiple CPU tiers (x86-64-v2/v3/v4):

```bash
# CLI binaries
cargo make build-cli-windows
cargo make build-cli-linux

# GUI binaries
cargo make build-gui-windows
cargo make build-gui-linux
```

All cargo-make builds are placed under the `dist/` directory.

## License

This project is licensed under the [GNU General Public License v3 (GPL v3)](LICENSE). By using or contributing to this project, you agree to comply with the terms of the license.

Neural Reversi includes code originally licensed under GPL v3 from the following projects:

- **[Edax](https://github.com/abulmo/edax-reversi)**
- **[Stockfish](https://github.com/official-stockfish/Stockfish)**

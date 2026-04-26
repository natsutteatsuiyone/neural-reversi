# Neural Reversi

[![Build](https://github.com/natsutteatsuiyone/neural-reversi/actions/workflows/test.yml/badge.svg)](https://github.com/natsutteatsuiyone/neural-reversi/actions/workflows/test.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

This is an experimental project to develop a highly accurate neural network evaluation function for Reversi (Othello).

**[Play online (Lite version)](https://neural-reversi.net/)**

## Features

- Neural network-based position evaluation
- High-performance multi-threaded search
- Supports CLI, desktop GUI (Tauri), and WebAssembly

## Benchmarks (v6.0.0)

### Environment

- **CPU:** AMD Ryzen 9 9950X3D
- **Threads:** 32
- **Hash size:** 2048 MB

### Evaluation Accuracy

| Test | Problems | Depth | Time | Nodes | NPS | Move Acc. | Score ±3 | MAE |
|:--|:-:|:-:|--:|--:|--:|--:|--:|--:|
| [Hard-30](docs/6.0.0/benchmarks/hard-30-depth15.md) | 289 | 15 | 3.758s | 151,563,290 | 40,335,131 | 83.0% | 87.9% | 1.64 |

### Endgame Solving

| Test | Problems | Depth | Time | Nodes | NPS |
|:--|:-:|:-:|--:|--:|--:|
| [FFO #40–59](docs/6.0.0/benchmarks/fforum-40-59.md) | 20 | 20–34 | 7.659s | 12,785,599,475 | 1,669,384,514 |
| [FFO #60–79](docs/6.0.0/benchmarks/fforum-60-79.md) | 20 | 24–36 | 235.436s | 339,714,844,461 | 1,442,919,528 |
| [Hard-20](docs/6.0.0/benchmarks/hard-20.md) | 276 | 20 | 2.580s | 1,681,608,922 | 651,747,905 |
| [Hard-25](docs/6.0.0/benchmarks/hard-25.md) | 311 | 25 | 32.421s | 46,126,798,879 | 1,422,746,901 |
| [Hard-30](docs/6.0.0/benchmarks/hard-30.md) | 289 | 30 | 904.934s | 1,471,667,735,637 | 1,626,269,909 |
| [Small-35](docs/6.0.0/benchmarks/small-35.md) | 20 | 35 | 10,494.203s | 14,526,843,918,418 | 1,384,273,243 |

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install)
- [cargo-make](https://github.com/sagiegurari/cargo-make) (recommended)
- [Bun](https://bun.sh/) (for GUI and Web development)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/natsutteatsuiyone/neural-reversi.git
   cd neural-reversi
   ```

2. Download the neural network weight files from the [latest release](https://github.com/natsutteatsuiyone/neural-reversi-weights/releases/latest)
   and place them in the project root directory:
   - `eval-*.zst`
   - `eval_sm-*.zst`
   - `eval_wasm-*.zst`

3. Run the interface you want to use:
   ```bash
   cargo run -p cli --release    # Play in the terminal (TUI)
   ```

   ```bash
   cd crates/gui
   bun install
   bun run tauri dev             # Launch the desktop GUI in development mode
   ```

   ```bash
   cd crates/web
   bun install
   bun run dev                   # Start the web version in development mode
   ```

## Crates

- **[reversi-core](crates/reversi-core/)**: Core library implementing the AI search algorithms.
- **[cli](crates/cli/)**: Command-line interface for playing Reversi.
- **[gui](crates/gui/)**: Tauri-based graphical user interface for playing Reversi.
- **[web](crates/web/)**: WebAssembly build of the Rust engine, packaged with wasm-pack and Vite, and used as the frontend bundle for [neural-reversi.net](https://neural-reversi.net).
- **[match-runner](crates/match-runner/)**: Tool for automatically running matches between Reversi engines supporting the Go Text Protocol.
- **[datagen](crates/datagen/)**: Tool for generating neural network training data, including self-play games and feature extraction.
- **[evaltest](crates/evaltest/)**: Evaluation test suite runner for benchmarking engine performance using OBF problem files (FFO Forum, Edax hard sets).

## Neural Network

### Architecture

#### Midgame

![Neural network architecture](docs/5.0.0/nn_architecture_5.0.0.svg)

#### Endgame

![Small neural network architecture](docs/5.0.0/nn_architecture_small_5.0.0.svg)

### Features

- Mobility: The number of legal moves for the current player.
- Patterns:  
  ![Pattern features](docs/5.0.0/pattern_features_5.0.0.svg)

### Training

[neural-reversi-training](https://github.com/natsutteatsuiyone/neural-reversi-training)

## Build

For best performance, build with native CPU optimizations. This enables CPU-specific instructions (BMI2, LZCNT, AVX2, AVX-512 on x86-64; NEON on Apple Silicon), which significantly improve evaluation and search speed.

### Using cargo-make

Build native-optimized binaries for both CLI and GUI using [cargo-make](https://github.com/sagiegurari/cargo-make):

```bash
# All platforms (Windows + Linux + macOS)
cargo make build-native

# Windows only
cargo make build-cli-windows-native
cargo make build-gui-windows-native

# Linux only
cargo make build-cli-linux-native
cargo make build-gui-linux-native

# macOS only (Apple Silicon)
cargo make build-cli-macos-native
cargo make build-gui-macos-native
```

For portable release builds (CPU tiers `x86-64-v2/v3/v4` on Windows/Linux, `apple-m1` on macOS):

```bash
# CLI binaries
cargo make build-cli-windows
cargo make build-cli-linux
cargo make build-cli-macos

# GUI binaries
cargo make build-gui-windows
cargo make build-gui-linux
cargo make build-gui-macos
```

All cargo-make builds are placed under the `dist/` directory. The macOS GUI build produces a `.dmg` installer (requires building on macOS).

## License

This project is licensed under the [GNU General Public License v3 (GPL v3)](LICENSE). By using or contributing to this project, you agree to comply with the terms of the license.

Neural Reversi includes code originally licensed under GPL v3 from the following projects:

- **[Edax](https://github.com/abulmo/edax-reversi)**
- **[Stockfish](https://github.com/official-stockfish/Stockfish)**

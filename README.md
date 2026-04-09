# Neural Reversi

This is an experimental project to develop a highly accurate neural network evaluation function for Reversi (Othello).

**[Play online (Lite version)](https://neural-reversi.net/)**

## Features

- Neural network-based position evaluation
- High-performance multi-threaded search
- Supports CLI, desktop GUI (Tauri), and WebAssembly

## Benchmark (v6.0.0-dev)

### Environment

- **CPU:** AMD Ryzen 9 9950X3D (no overclock)
- **Hash size:** 1024MB

### Search Accuracy

| Test | Problems | Depth | Time | Nodes | NPS | Move Acc. | Score ±3 | MAE |
|:--|:-:|:-:|--:|--:|--:|--:|--:|--:|
| [Hard-30](docs/6.0.0-dev/hard-30-depth15.md) | 289 | 15 | 4.22s | 157,847,499 | 37,443,661 | 81.3% | 88.2% | 1.65 |

### Endgame Solving

| Test | Problems | Depth | Time | Nodes | NPS |
|:--|:-:|:-:|--:|--:|--:|
| [FFO #40–59](docs/6.0.0-dev/fforum-40-59.md) | 20 | 20–34 | 7.85s | 12,816,866,056 | 1,632,243,554 |
| [FFO #60–79](docs/6.0.0-dev/fforum-60-79.md) | 20 | 24–36 | 246.10s | 344,382,051,901 | 1,399,342,843 |
| [Hard-20](docs/6.0.0-dev/hard-20.md) | 276 | 20 | 2.65s | 1,685,702,316 | 635,131,425 |
| [Hard-25](docs/6.0.0-dev/hard-25.md) | 311 | 25 | 33.02s | 46,181,908,759 | 1,398,748,168 |
| [Hard-30](docs/6.0.0-dev/hard-30.md) | 289 | 30 | 936.21s | 1,479,391,622,235 | 1,580,200,514 |

## Crates

- **[reversi-core](crates/reversi-core/)**: Core library implementing the AI search algorithms.
- **[cli](crates/cli/)**: Command-line interface for playing Reversi.
- **[gui](crates/gui/)**: Tauri-based graphical user interface for playing Reversi.
- **[web](crates/web/)**: WebAssembly build of the Rust engine, packaged with wasm-pack and Vite, and used as the frontend bundle for [neural-reversi.net](https://neural-reversi.net).
- **[match-runner](crates/match-runner/)**: Tool for automatically running matches between Reversi engines supporting the Go Text Protocol.
- **[datagen](crates/datagen/)**: Tool for generating neural network training data, including self-play games and feature extraction.
- **[evaltest](crates/evaltest/)**: Evaluation test suite runner for benchmarking engine performance using OBF problem files (FFO Forum, Edax hard sets).

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) 1.92.0+
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

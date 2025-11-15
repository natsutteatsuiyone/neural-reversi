# Neural Reversi

This is an experimental project to develop a high-accuracy neural network evaluation function for Reversi (Othello).

Play it online (Lite version): https://neural-reversi.net/ 

## Crates

- **reversi_core**: Core library implementing the AI search algorithms.
- **reversi_cli**: Command-line interface for playing Reversi.
- **reversi_gui**: Tauri-based graphical user interface for playing Reversi.
- **reversi_web**: WebAssembly build of the Rust engine, packaged with wasm-pack and Vite, and used as the frontend bundle for [https://neural-reversi.net](https://neural-reversi.net).
- **match_runner**: Tool for automatically running matches between Reversi engines supporting the Go Text Protocol.
- **datagen**: Tool for generating neural-network training data, including self-play games and feature extraction.
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

### Weight Files

The neural network weight files (`eval.zst` and `eval_sm.zst`) can be downloaded from the [Releases page](https://github.com/natsutteatsuiyone/neural-reversi/releases).

When developing and running the application locally from the source code (e.g., using `cargo run`), make sure to place `eval.zst` and `eval_sm.zst` in the root directory.

## License

This project is licensed under the [GNU General Public License v3 (GPL v3)](LICENSE). By using or contributing to this project, you agree to comply with the terms of the license.

Neural Reversi includes code originally licensed under GPL v3 from the following projects:

- **[Edax](https://github.com/abulmo/edax-reversi)**
- **[Stockfish](https://github.com/official-stockfish/Stockfish)**

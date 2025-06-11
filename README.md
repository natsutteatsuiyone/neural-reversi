# Neural Reversi

This is an experimental project to develop a high-accuracy neural network evaluation function for Reversi (Othello).

## Crates

- **reversi_core**: Core library implementing AI search algorithms.
- **reversi_cli**: Command-line interface for playing Reversi.
- **reversi_gui**: Graphical user interface built with Tauri for playing Reversi.
- **match_runner**: Tool for automatically running matches between Reversi engines supporting the Go Text Protocol.
- **build_common**: Common build script utilities shared across different crates.
- **codegen**: Utility for generating evaluation feature-related code.
- **datagen**: Tool for generating neural network training data, including self-play games and feature extraction.
- **ffotest**: [FFO endgame test suite](http://radagast.se/othello/ffotest.html). [Edax problems](https://github.com/abulmo/edax-reversi/tree/master/problem).

## Neural Network

### Architecture

![Neural network architecture](docs/img/nn_architecture.svg)

### Features

- Mobility: Number of legal moves for the current player.
- Patterns: 6561 x 22  
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

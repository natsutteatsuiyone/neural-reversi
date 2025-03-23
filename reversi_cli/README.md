# Neural Reversi CLI

A command-line interface for playing Reversi (Othello) with an AI opponent powered by the Neural Reversi engine.

## Usage

### Interactive Mode

Run the CLI without arguments to start in interactive mode:

```bash
neural-reversi-cli
```

In interactive mode, you can use the following commands:

- `play <position>` - Make a move (e.g., `play e3`)
- `undo` - Undo the last move
- `new` - Start a new game
- `mode <number>` - Change game mode:
  - `0`: Human (Black) vs AI (White)
  - `1`: AI (Black) vs Human (White)
  - `2`: AI vs AI
  - `3`: Human vs Human
- `level <number>` - Set AI difficulty level (1-21)
- `quit` - Exit the program

### GTP Mode

Run the CLI in GTP mode for integration with other applications:

```bash
neural-reversi-cli gtp --level <level> --selectivity <selectivity>
```

Options:

- `--level <level>`: Set the AI search level (default: 10)

In GTP mode, the program accepts standard GTP commands plus some Reversi-specific extensions:

- `boardsize 8` - Set board size (only 8x8 is supported)
- `clear_board` - Reset the board to starting position
- `play <color> <move>` - Make a move (e.g., `play b e3`)
- `genmove <color>` - Let the AI generate a move
- `showboard` - Display the current board state
- `set_level <level>` - Change the AI difficulty level

## Global Options

These options apply to both interactive and GTP modes:

- `--selectivity <value>` - Set the search selectivity (1: 73%, 2: 87%, 3: 95%, 4: 98%, 5: 99%, 6: 100%) (default: 1)
- `--hash-size <size>` - Set the transposition table size in MB (default: 1)

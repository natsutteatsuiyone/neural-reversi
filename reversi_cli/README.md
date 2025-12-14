# Neural Reversi CLI

A command-line interface for playing Reversi (Othello) with an AI opponent powered by the Neural Reversi engine.

## Usage

### Interactive Mode

Run the CLI without arguments to start in interactive mode:

```bash
neural-reversi-cli [options]
```

Options:

- `--hash-size <size>` - Set the transposition table size in MB (default: 64)
- `-l, --level <level>` - Set the AI difficulty level (default: 21)
- `--selectivity <value>` - Set the search selectivity (1: 73%, 2: 87%, 3: 95%, 4: 98%, 5: 99%, 6: 100%) (default: 1)
- `--threads <number>` - Set the number of threads to use for search (default: number of CPU cores)

In interactive mode, you can use the following commands:

- `<square>` - Make a move (e.g., `d3`, `e4`)
- `init`, `i` - Initialize a new game
- `new`, `n` - Start a new game
- `undo`, `u` - Undo last move
- `level`, `l <n>` - Set AI level
- `mode`, `m [n]` - Show/set game mode:
  - `0`: Black-Human, White-AI
  - `1`: Black-AI, White-Human
  - `2`: Black-AI, White-AI
  - `3`: Black-Human, White-Human
- `go` - Let AI make a move with analysis
- `play <moves>` - Play a sequence of moves
- `setboard <pos>` - Set board position (64 board chars + optional spaces + 1 side to move char)
- `help`, `h` - Show this help
- `quit`, `q` - Exit the program

### GTP Mode

Run the CLI in GTP mode for integration with other applications:

```bash
neural-reversi-cli gtp [options]
```

Options:

- `--hash-size <size>` - Set the transposition table size in MB (default: 64)
- `--level <level>` - Set the AI search level (default: 21)
- `--selectivity <value>` - Set the search selectivity (1: 73%, 2: 87%, 3: 95%, 4: 98%, 5: 99%, 6: 100%) (default: 1)
- `--threads <number>` - Set the number of threads to use for search (default: number of CPU cores)

In GTP mode, the program accepts standard GTP commands plus some Reversi-specific extensions:

- `boardsize 8` - Set board size (only 8x8 is supported)
- `clear_board` - Reset the board to starting position
- `play <color> <move>` - Make a move (e.g., `play b e3`)
- `genmove <color>` - Let the AI generate a move
- `showboard` - Display the current board state
- `set_level <level>` - Change the AI difficulty level
- `time_settings <main_time> <byoyomi_time> <byoyomi_stones>` - Configure time control
- `time_left <color> <time> <stones>` - Update remaining time for a player

#### Time Control

The GTP mode supports time control for timed games. Use `time_settings` to configure the time control mode:

**Byoyomi (fixed time per move):**

```
time_settings 0 5 1
```

This sets 5 seconds per move with no main time.

**Fischer (main time + increment):**

```
time_settings 300 5 1
```

This sets 300 seconds main time with 5 seconds increment per move.

Before each `genmove`, send `time_left` to inform the engine of the remaining time:

```
time_left black 295 0
time_left white 300 0
genmove black
```

### Solve Mode

Run the CLI in solve mode to analyze positions from a file:

```bash
neural-reversi-cli solve <file> [options]
```

Options:

- `<file>` - Path to the position file (required)
- `--exact` - Solve to the end with exact depth (ignore level for perfect endgame solving)
- `--hash-size <size>` - Set the transposition table size in MB (default: 64)
- `-l, --level <level>` - Set the AI search level (default: 21)
- `--selectivity <value>` - Set the search selectivity (1: 73%, 2: 87%, 3: 95%, 4: 98%, 5: 99%, 6: 100%) (default: 1)
- `--threads <number>` - Set the number of threads to use for search (default: number of CPU cores)

The position file should contain one position per line in the following format:

```text
<64-character board><side-to-move>; [optional analysis data]
```

Where:

- Board positions use `-` for empty, `X` for black, `O` for white
- Side-to-move is either `X` (black) or `O` (white)
- Lines starting with `%` are treated as comments
- Analysis data after `;` is ignored

Example position line:

```text
O--OOOOX-OOOOOOXOOXXOOOXOOXOOOXXOOOOOOXX---OOOOX----O--X-------- X; A2:+38
-OOOOO----OOOOX--OOOOOO-XXXXXOO--XXOOX--OOXOXX----OXXO---OOO--O- X; H4:+0
```

# Time Debug

A self-play debugging tool for testing and verifying time control implementation in the Reversi engine.

## Usage

```bash
time-debug [OPTIONS]
```

### Options

- `-g, --games <GAMES>`: Number of games to play (default: 1)
- `-t, --time-mode <TIME_MODE>`: Time control mode (`none`, `byoyomi`, `fischer`) (default: `byoyomi`)
- `--main-time <MAIN_TIME>`: Main time in milliseconds for Fischer mode (default: 60000)
- `--byoyomi <BYOYOMI>`: Time per move (byoyomi) or increment (Fischer) in milliseconds (default: 5000)
- `-l, --level <LEVEL>`: Search level (default: 24, use high value to exercise time control)
- `--hash-size <HASH_SIZE>`: Hash table size in MB (default: 256)
- `--selectivity <SELECTIVITY>`: Search selectivity 0-5 (default: 0)
- `-o, --opening <OPENING>`: Opening moves (e.g., "f5d6c3")
- `-v, --verbose`: Show search progress for each move

## Debugging

You can enable detailed time control debug logs by setting the `REVERSI_DEBUG_TIME` environment variable.

### PowerShell
```powershell
$env:REVERSI_DEBUG_TIME=1; cargo run -p time-debug --release -- --main-time 10000
```

### Bash
```bash
REVERSI_DEBUG_TIME=1 cargo run -p time-debug --release -- --main-time 10000
```

This will output logs for:
- Initial time limits
- Iteration completion stats (depth, nodes, NPS)
- Time budget updates
- Time extension triggers
- Endgame transitions

## Time Control Modes

### None
No time limit. The engine searches to the specified level without time constraints.

### Byoyomi
Fixed time per move. Each move must be completed within the byoyomi time. If time is exceeded, the game is forfeited.

### Fischer
Main time plus increment per move. The engine starts with main time, and each move adds the increment to the remaining time. If time runs out, the game is forfeited.

## Examples

### Basic Self-Play with Byoyomi

Play a single game with 5 seconds per move:

```bash
time-debug --time-mode byoyomi --byoyomi 5000
```

### Fischer Time Control

Play with 60 seconds main time and 2 seconds increment:

```bash
time-debug --time-mode fischer --main-time 60000 --byoyomi 2000
```

### Multiple Games with Statistics

Run 10 games to gather timing statistics:

```bash
time-debug --games 10 --time-mode byoyomi --byoyomi 3000
```

### With Opening Sequence

Start from a specific opening position:

```bash
time-debug --opening "f5d6c3d3" --time-mode byoyomi --byoyomi 2000
```

### Verbose Output

Show search progress for each move:

```bash
time-debug --time-mode byoyomi --byoyomi 5000 --verbose
```

## Output

The tool displays a detailed move-by-move table during each game:

```
  Move   Side   Square   Time(ms)  Remaining    Depth  Score
  ----------------------------------------------------------------------
     1  Black       F5          1       5000        5   0.47
     2  White       D6          2       5000        5  -0.33
     3  Black       C3          1       5000        5   0.52
```

- **Move**: Move number in the game
- **Side**: Player to move (Black/White)
- **Square**: Move played (e.g., F5)
- **Time(ms)**: Time taken for this move in milliseconds
- **Remaining**: Remaining time (for Fischer mode)
- **Depth**: Search depth reached
- **Score**: Evaluation score (positive favors Black)

Time values are color-coded:
- Green: Under 50% of time budget
- Yellow: 50-90% of time budget
- Red: Over 90% of time budget

### Final Statistics

After all games complete, statistics are displayed:

```
Results:
  Black wins:    5 (50.0%)
  White wins:    4 (40.0%)
  Draws:         1 (10.0%)

Time Statistics:
  Black:
    Total time:   12345 ms
    Total moves:  290
    Avg time:     42.6 ms/move
    Max time:     498 ms
    Budget usage: 85.2%
```

## Notes on Time Control

In endgame positions (roughly 30 or fewer empty squares), the engine switches to an endgame solver which may have different time control behavior.

## Building

```bash
cargo build --release -p time-debug
```

The binary will be available at `target/release/time-debug`.

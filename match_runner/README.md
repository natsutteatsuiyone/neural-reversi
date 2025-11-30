# Match Runner

A tool for playing matches between two computer Reversi programs using the Go Text Protocol (GTP).

## Usage

```bash
match_runner [OPTIONS] --engine1 <ENGINE1> --engine2 <ENGINE2> --opening-file <OPENING_FILE>
```

### Options

- `-1, --engine1 <ENGINE1>`: Command for the first engine (executable path and arguments) (required)
- `--engine1-working-dir <ENGINE1_WORKING_DIR>`: Working directory for the first engine
- `-2, --engine2 <ENGINE2>`: Command for the second engine (executable path and arguments) (required)
- `--engine2-working-dir <ENGINE2_WORKING_DIR>`: Working directory for the second engine
- `-o, --opening-file <OPENING_FILE>`: File containing opening sequences (required)
- `--time-control <MODE>`: Time control mode (`none`, `byoyomi`, `fischer`) (default: `none`)
- `--main-time <SECONDS>`: Main time in seconds for Fischer mode (default: 300)
- `--byoyomi-time <SECONDS>`: Time per move (byoyomi) or increment (Fischer) in seconds (default: 5)

### Opening File Format

The opening file contains one opening sequence per line. Each sequence is written as a series of coordinates, such as `f5d6c4d3`.
Lines starting with `#` are treated as comments, and empty lines are ignored.

Example:

```text
# Standard opening
f5d6c4d3

# Tiger strategy
f5f6e6f4
```

For each opening sequence in the file, two games will be played (with colors swapped in the second game).

## Examples

### Basic Match

```bash
match-runner --engine1 "./reversi_cli gtp --level 10" --engine2 "./reversi_cli gtp --level 5" --opening-file openings.txt
```

### Match with Custom Working Directories

```bash
match-runner --engine1 "./reversi_cli gtp --level 10" --engine1-working-dir "./engine1_dir" --engine2 "./reversi_cli gtp --level 5" --engine2-working-dir "./engine2_dir" --opening-file openings.txt
```

### Match with Time Control (Byoyomi)

Play with 5 seconds per move:

```bash
match-runner --engine1 "./reversi_cli gtp --level 10" --engine2 "./reversi_cli gtp --level 10" --opening-file openings.txt --time-control byoyomi --byoyomi-time 5
```

### Match with Time Control (Fischer)

Play with 60 seconds main time and 2 seconds increment per move:

```bash
match-runner --engine1 "./reversi_cli gtp --level 10" --engine2 "./reversi_cli gtp --level 10" --opening-file openings.txt --time-control fischer --main-time 60 --byoyomi-time 2
```

## GTP Protocol

This tool communicates with Reversi programs using the [Go Text Protocol (GTP)](https://www.gnu.org/software/gnugo/gnugo_19.html).
The programs must support the following GTP commands:

- `name` - Return the program's name
- `version` - Return the program's version (optional)
- `clear_board` - Reset the board to the initial state
- `play <color> <move>` - Play a move of the specified color at the given coordinates
- `genmove <color>` - Generate a move for the specified color
- `quit` - Exit the program

When time control is enabled, the following commands are also used (optional support):

- `time_settings <main_time> <byoyomi_time> <byoyomi_stones>` - Configure time control
- `time_left <color> <time> <stones>` - Update remaining time for a player

## Match Output

During the match, the tool displays:
- Real-time progress visualization with win/loss/draw statistics
- Live updating score bars for each engine
- Progress bar showing game completion

After the match completes, detailed statistics are displayed including:
- Total games played with win/loss/draw breakdown
- Score percentage and average disc difference for each engine
- ELO rating estimation with confidence intervals
- Pentanomial statistics for paired game analysis

## Building

```bash
cargo build --release
```

### References

- [Go Text Protocol (GTP)](https://www.gnu.org/software/gnugo/gnugo_19.html)
- [XOT openings](https://berg.earthlingz.de/xot/aboutxot.php?lang=en)

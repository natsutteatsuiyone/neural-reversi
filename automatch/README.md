# Auto Match

A tool for playing matches between two computer Reversi programs using the Go Text Protocol (GTP).

## Usage

```bash
automatch [OPTIONS] --engine1 <ENGINE1> --engine2 <ENGINE2> --opening-file <OPENING_FILE>
```

### Options

- `-1, --engine1 <ENGINE1>`: Command for the first engine (executable path and arguments) (required)
- `--engine1-working-dir <ENGINE1_WORKING_DIR>`: Working directory for the first engine
- `-2, --engine2 <ENGINE2>`: Command for the second engine (executable path and arguments) (required)
- `--engine2-working-dir <ENGINE2_WORKING_DIR>`: Working directory for the second engine
- `-o, --opening-file <OPENING_FILE>`: File containing opening sequences (required)

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
automatch --engine1 "./reversi_cli gtp --level 10" --engine2 "./reversi_cli gtp --level 5" --opening-file openings.txt
```

### Match with Custom Working Directories

```bash
automatch --engine1 "./reversi_cli gtp --level 10" --engine1-working-dir "./engine1_dir" --engine2 "./reversi_cli gtp --level 5" --engine2-working-dir "./engine2_dir" --opening-file openings.txt
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

## Building

```bash
cargo build --release
```

### References

- [Go Text Protocol (GTP)](https://www.gnu.org/software/gnugo/gnugo_19.html)
- [XOT openings](https://berg.earthlingz.de/xot/aboutxot.php?lang=en)

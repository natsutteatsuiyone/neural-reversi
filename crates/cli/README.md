# Neural Reversi CLI

A command-line interface for playing Reversi (Othello) with an AI opponent powered by the Neural Reversi engine.

## Usage

### Interactive Mode

Run the CLI without arguments to start in interactive mode:

```bash
cli [options]
```

Options:

- `--hash-size <size>` - Set the transposition table size in MB (default: 512)
- `-l, --level <level>` - Set the AI difficulty level (default: 21)
- `--threads <number>` - Set the number of threads to use for search (default: number of CPU cores)
- `--eval-file <FILE>` - Path to the main network weight file (optional; defaults to the engine's standard weights)
- `--eval-sm-file <FILE>` - Path to the small network weight file (optional; defaults to the engine's standard weights)

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
- `hint [n]` - Show top n move hints with evaluations (default: 3, max: 10)
- `play <moves>` - Play a sequence of moves
- `setboard <pos>` - Set board position (64 board chars + optional spaces + 1 side to move char)
- `help`, `h` - Show this help
- `quit`, `q` - Exit the program

### GTP Mode

Run the CLI in GTP mode for integration with other applications:

```bash
cli gtp [options]
```

Options:

- `--hash-size <size>` - Set the transposition table size in MB (default: 512)
- `--level <level>` - Set the AI search level (default: 21)
- `--threads <number>` - Set the number of threads to use for search (default: number of CPU cores)
- `--eval-file <FILE>` - Path to the main network weight file (optional; defaults to the engine's standard weights)
- `--eval-sm-file <FILE>` - Path to the small network weight file (optional; defaults to the engine's standard weights)

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

### GGS Mode

Connect to the Othello [Generic Game Server](https://skatgame.net/mburo/ggs/) and play `/os` matches. The engine plays moves on your turn. In an interactive terminal, the TUI shows every game's score at a glance, the selected game's board and both clocks, the complete transcript, and a command editor.

```bash
cli ggs --user <name> --password <password> [--max-games <n>] [options]
```

Options:

- `--user <name>` - GGS login name.
- `--password <password>` - GGS password, sent verbatim at the password prompt.
- `--host <host>` - GGS server hostname. Default: `localhost`.
- `--port <port>` - GGS server port. Default: `5000`.
- `--max-games <n>` - Maximum simultaneous matches, matching GGS `/os open`. Default: `1`; a synchro match counts as one match with two game boards. Configured TT memory is divided across up to two boards per match (at least two MB per match), and all boards share one pool of `--threads` search threads and search one at a time.
- `--ponder` - While every game waits for the opponent, search the opponent-to-move board to warm each game's transposition table. Off by default.
- `--plain` - Use the original line input and output. This mode is also selected automatically when stdin or stdout is redirected.
- `--command <TEXT>` - Send one startup command after login and client normalization. Repeat this option to send multiple commands in order.
- `--hash-size <size>` - Transposition table size in MB. Default: `512`.
- `-l, --level <level>` - Fallback search level when server time is unparseable. Default: `21`.
- `--threads <n>` - Number of search threads. Default: CPU count.
- `--eval-file <FILE>` - Path to the main network weight file. Optional; defaults to the engine's standard weights.
- `--eval-sm-file <FILE>` - Path to the small network weight file. Optional; defaults to the engine's standard weights.

After login, type commands such as `info <text>`, `tell /os rated +`, `tell /os dform <formula>`, or `tell /os ask ...` in either frontend.

Use repeated `--command` options in a launcher so stdin stays attached to the console:

```bat
cli ggs --user <name> --password <password> --command "tell /os rated +" --command "tell /os dform size != 8"
```

Each `--command` value must be one line. The client preserves leading and trailing spaces, preserves blank commands, and sends the values in option order after its built-in login and normalization commands. The same ask-clock validation applies to startup commands and commands entered after startup.

The TUI command editor forwards GGS commands, including `help` and `quit`, without local command aliases. Enter sends the command, preserving spaces and blank lines. In terminals that provide bracketed paste events, pasted text stays in the editor until Enter. Multiline paste sends one command per line, preserving empty lines. Windows console input delivers paste as keystrokes, so pasted line breaks send commands immediately.

| Key | Action |
| --- | --- |
| F1 | Show or hide keyboard help. |
| Left, Right, Home, End | Move the command cursor. |
| Backspace, Delete | Remove a character. |
| Up, Down | Recall command history and restore the current draft. |
| Ctrl+U | Clear the command editor. |
| Tab or F6, Shift+Tab | Select the next or previous game, including synchro child games. |
| Page Up, Page Down | Scroll the transcript vertically. New output preserves the viewed line. |
| Alt+Left, Alt+Right | Scroll the transcript horizontally. |
| Ctrl+Home, Ctrl+End | Show the oldest transcript lines or follow new output. |
| Mouse wheel | Scroll the transcript; over the game list, select the next or previous game. Shift+wheel scrolls horizontally. |
| Click | Select a game in the list, or click or drag the transcript scrollbar. |
| Shift+drag | Select text to copy while the TUI captures the mouse. |
| Ctrl+C | Exit the client and restore the terminal. |

The game list shows one row per game, including both synchro games, with disc counts, the last search evaluation from your side, and the status or result. Finished games show Win, Loss, or Draw with the disc difference from your side. The board uses absolute black and white colors and highlights the last move. Every game with a board is shown at once, one board below another, when the height allows. Terminals at least 130 columns wide place the transcript in a column to the right of the game. Before a board arrives, a compact welcome panel leaves more room for server output and shows command examples.

Search status distinguishes waiting for the shared engine from actually searching or pondering. The last completed search report remains with its game. Completed games retain their last board and result until the next match starts. Watched games display a board when the server provides a supported snapshot. Both players' clocks show the latest server values, and the side to move counts down locally until the next server update. Small terminals show compact game status while keeping the transcript and command editor usable. The transcript retains all server lines and diagnostics, with control characters shown as escaped text in the TUI.

Use `--plain` when enabling `REVERSI_DEBUG_TIME`, which writes engine timing diagnostics directly to stderr.

Notes:

- Plain mode echoes server output to stdout and writes engine diagnostics to stderr. Both frontends forward commands verbatim, except `tell /os ask` lines whose clock the engine cannot play (or that repeat a stored ask by `.id`), which are dropped with a diagnostic. The engine additionally auto-sends moves and `accept`/`decline` for incoming ask offers, and sends the no-op `:` command after 50 seconds without outbound traffic to keep the connection alive.
- Simultaneous matches are tracked independently. `--max-games` also sets `/os open`; excess direct asks are declined and over-cap matches are resigned. Only asks addressed to you are accepted automatically; global asks are ignored.
- After authentication, the client restores the standard `READY`/`ALERT` prompts and forces `/os client -`, `/os trust +`, and `/os vt100 -`. This keeps messages parseable even when account settings persisted from an earlier client. Moves include the elapsed turn time in seconds, including engine queue and result handling waits, so the server uses the reported time rather than charging network latency within its trust tolerance.
- Clocks use the live main budget plus additive every-move increments. GGS extension, Bronstein, and periodic-increment fields are not misreported as per-move time; unparseable clocks use `--level`.
- Reconnecting into an active match recovers from the server's join state, including a one-shot verbose-state request when the persisted client mode initially produced compact GGF.

### Solve Mode

Run the CLI in solve mode to analyze positions from a file:

```bash
cli solve <file> [options]
```

Options:

- `<file>` - Path to the position file (required)
- `--exact` - Solve for exact score with perfect play (ignores level setting)
- `--all-moves` - Report the score and principal variation for every legal move in each position
- `--hash-size <size>` - Set the transposition table size in MB (default: 512)
- `-l, --level <level>` - Set the AI search level (default: 21)
- `--threads <number>` - Set the number of threads to use for search (default: number of CPU cores)
- `--eval-file <FILE>` - Path to the main network weight file (optional; defaults to the engine's standard weights)
- `--eval-sm-file <FILE>` - Path to the small network weight file (optional; defaults to the engine's standard weights)

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

### License Information

Print the GPL-3.0 license covering this program:

```bash
cli show-license
```

Print the license texts of all bundled third-party crates:

```bash
cli show-licenses
```

The CLI bundle is generated from the workspace-level `about.toml` +
`about.hbs` by [cargo-about](https://github.com/EmbarkStudios/cargo-about).
The GUI bundle additionally merges frontend npm-package licenses via
[generate-license-file](https://github.com/TomerAberbach/generate-license-file)
over `crates/gui/package.json`. Regenerate both bundles after changing
dependencies with:

```bash
cargo make licenses
```

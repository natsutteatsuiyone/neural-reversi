# Neural Reversi CLI

A command-line interface for playing Reversi (Othello) with an AI opponent powered by the Neural Reversi engine.

## Usage

### Interactive Mode

Run the CLI without a subcommand to open the terminal UI:

```bash
cli [options]
```

Options:

- `--hash-size <size>` - Set the transposition table size in MB (default: 512)
- `-l, --level <level>` - Set the AI difficulty level (default: 21)
- `--threads <number>` - Set the number of threads to use for search (default: number of CPU cores, capped at 64; explicit values must be 1 to 64 and are lowered to the CPU count)
- `--eval-file <FILE>` - Path to the main network weight file (optional; defaults to the engine's standard weights)
- `--eval-sm-file <FILE>` - Path to the small network weight file (optional; defaults to the engine's standard weights)

Use the following keys in the game view:

| Key | Action |
| --- | --- |
| Arrow keys, WASD, or HJKL | Move the board cursor. |
| Enter or Space | Play at the cursor. Clicking a square also plays there. |
| U | Undo a move, or back to your previous turn when playing against the AI. |
| N | Start a new game. |
| I | Search all legal moves and show hints with scores and principal variations. |
| G | Let the AI play the current side. |
| V | Set the AI difficulty level. |
| M | Select a game mode: 1 for Human vs AI, 2 for AI vs Human, 3 for AI vs AI, or 4 for Human vs Human. |
| E | Open the board editor: move sequence, board string, or bitboards. |
| Q or Esc | Open quit confirmation. |
| Ctrl+C | Exit immediately. |

AI moves and hints share one engine and run in the background. Press Esc or I
while hints are being computed to cancel them. The cancelled result is discarded;
the engine stays busy until the worker returns it. Moves, undo, new games, board
editing, and further searches remain unavailable during that interval.

### GTP Mode

Run the CLI in GTP mode for integration with other applications:

```bash
cli gtp [options]
```

Options:

- `--hash-size <size>` - Set the transposition table size in MB (default: 512)
- `-l, --level <level>` - Set the AI search level (default: 21)
- `--threads <number>` - Set the number of threads to use for search (default: number of CPU cores, capped at 64; explicit values must be 1 to 64 and are lowered to the CPU count)
- `--eval-file <FILE>` - Path to the main network weight file (optional; defaults to the engine's standard weights)
- `--eval-sm-file <FILE>` - Path to the small network weight file (optional; defaults to the engine's standard weights)

In GTP mode, the program accepts standard GTP commands plus some Reversi-specific extensions:

- `protocol_version` - Report the GTP protocol version (`2`)
- `name` - Report the engine name
- `version` - Report the engine version
- `known_command <command>` - Check whether a GTP command is supported
- `list_commands` - List all supported GTP commands
- `quit` - Exit GTP mode
- `boardsize 8` - Set board size (only 8x8 is supported)
- `clear_board` - Reset the board to starting position
- `setboard <board> <B|W|X|O|black|white>` - Set the board from a 64-character string (`-`/`X`/`O`) and side to move
- `play <color> <move>` - Make a move (e.g., `play b e3`)
- `genmove <color>` - Let the AI generate a move
- `showboard` - Display the current board state
- `undo` - Undo the last move
- `set_level <level>` - Change the AI difficulty level
- `time_settings <main_time> <byoyomi_time> <byoyomi_stones>` - Configure time control
- `time_left <color> <time> <stones>` - Update remaining time for a player

#### Time Control

The GTP mode supports time control for timed games. Use `time_settings` to configure the time control mode:

**Byoyomi (fixed time per move):**

```
time_settings 0 5 0
```

This sets 5 seconds per move with no main time. (When both `byoyomi_time` and `byoyomi_stones` are greater than 0, `time_settings <main_time> <byoyomi_time> <byoyomi_stones>` configures Japanese/Canadian byoyomi with `<byoyomi_time>` seconds per `<byoyomi_stones>` moves.)

**Fischer (main time + increment):**

```
time_settings 300 5 0
```

This sets 300 seconds main time with 5 seconds increment per move.

Before each `genmove`, send `time_left` to inform the engine of the remaining time:

```
time_left black 295 0
time_left white 300 0
genmove black
```

### GGS Mode

Connect to the Othello [Generic Game Server](https://skatgame.net/mburo/ggs/) and play `/os` matches. The engine plays moves on your turn. In an interactive terminal, the TUI shows every game's score at a glance, the selected game's board and both clocks, the transcript, and a command editor.

```bash
cli ggs --user <name> --password <password> [--max-games <n>] [options]
```

Options:

- `--user <name>` - GGS login name: 2 to 8 printable ASCII characters without spaces, `,` or `\`, not starting with `.` or `_`. Other names are refused before connecting, as the server would refuse them.
- `--password <password>` - GGS password, sent verbatim at the password prompt. It must be one line.
- `--host <host>` - GGS server hostname; must not be empty. Default: `localhost`.
- `--port <port>` - GGS server port, 1 to 65535. Default: `5000`.
- `--max-games <n>` - Maximum simultaneous matches, matching GGS `/os open`. Default: `1`; a synchro match counts as one match with two game boards. Each match can allocate up to two transposition tables of `--hash-size` (the two boards of a synchro match share one table until their move sequences diverge, then each has its own), and all boards share one pool of `--threads` search threads and search one at a time.
- `--ponder` - While every game waits for the opponent, search the opponent-to-move board to warm each game's transposition table. Off by default.
- `--plain` - Use the original line input and output. This mode is also selected automatically when stdin or stdout is redirected.
- `--join-tournaments` - Join a `/td` tournament when its director tells us `tell /td join <id>`, for unattended tournament entry. The client first asks `tell /td finger <id>` and joins only if the sender is the tournament's director, its service is `/os`, and the engine plays its game type and clock; requests from anyone else are logged and ignored.
- `--command <TEXT>` - Send one startup command after login and client normalization. Repeat this option to send multiple commands in order.
- `--log-file <PATH>` - Append every line sent to and received from the server after the password, with Unix timestamps.
- `--hash-size <size>` - Transposition table size in MB. Default: `512`.
- `-l, --level <level>` - Fallback search level when server time is unparseable. Default: `21`.
- `--threads <n>` - Number of search threads. Default: CPU count (capped at 64; explicit values must be 1 to 64 and are lowered to the CPU count).
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

Search status distinguishes waiting for the shared engine from actually searching or pondering. The last completed search report remains with its game. Completed games retain their last board and result until the next match starts. Watched games are ignored and do not display a board. Both players' clocks show the latest server values along with the elapsed time since that snapshot (clocks are not counted down locally). Small terminals show compact game status while keeping the transcript and command editor usable. The transcript retains up to the newest 10,000 server lines and diagnostics, with control characters shown as escaped text in the TUI.

Use `--plain` when enabling `REVERSI_DEBUG_TIME`, which writes engine timing diagnostics directly to stderr.

Notes:

- Plain mode echoes server output to stdout and writes engine diagnostics to stderr. Both frontends forward commands verbatim, except `tell /os ask` lines whose clock the engine cannot play (or that repeat a stored ask by `.id`), which are dropped with a diagnostic. The engine additionally auto-sends moves and `accept`/`decline` for incoming ask offers, and sends the no-op `:` command after 50 seconds without outbound traffic to keep the connection alive.
- Simultaneous matches are tracked independently. `--max-games` also sets `/os open`; excess direct asks are declined and over-cap matches are resigned. Only asks addressed to you are accepted automatically; global asks are ignored.
- After authentication, the client restores the standard `READY`/`ALERT` prompts and forces `/os client -`, `/os trust +`, and `/os vt100 -`. This keeps messages parseable even when account settings persisted from an earlier client. Moves include the elapsed turn time in seconds, including engine queue and result handling waits, so the server uses the reported time rather than charging network latency within its trust tolerance.
- Clocks use the live main budget plus additive every-move increments. GGS extension, Bronstein, and periodic-increment fields are not misreported as per-move time; unparseable clocks use `--level`.
- Reconnecting into an active match recovers from the server's join state, including a one-shot verbose-state request when the persisted client mode initially produced compact GGF.

### FlipSight Mode

Play as a [FlipSight](https://othelloclub.com/ja/flipsight/lab/api) bot through the site's bot connector (`bot_match_client.py`). The connector starts one `cli flipsight` process per game and exchanges one JSON line per move over stdin and stdout.

```powershell
py bot_match_client.py --engine-threads 4 --max-think-seconds 0 --engine 'C:\path\to\cli.exe flipsight --hash-size 512'
```

Options are the engine options of the other modes. `--threads` defaults to the request's `engine_threads`, so `--engine-threads` sets both the engine and the value the connector reports. Each game uses at most 4 threads, the FlipSight limit; larger values are lowered to 4 and the actual count is reported. Both games of a bot match search at the same time in separate processes, so keep the threads × (concurrent matches + 1) within the CPU's threads; `--hash-size` is allocated by every process.

Notes:

- Only standard 8x8 Othello is played. Any other game mode makes the process exit with an error, so leave variants unchecked in the bot's supported games.
- The engine keeps its transposition table for the whole game. It answers a pass or a single legal move at once, and otherwise searches with the game clock as a sudden-death bank (`remaining_ms` less a reserve of 1 s plus 100 ms for each move it has yet to play, since the server charges network time to the clock; the first move is also charged for process startup). With less than 1 s above the reserve, each move gets a fixed 150 ms search instead.
- The search is always stopped 300 ms before the connector's `budget_ms`. Use `--max-think-seconds 0` so this cap is the remaining clock rather than the connector's default 10 s per move.
- `--ponder` searches, while the opponent thinks, the position after our move (or our next position when the opponent must pass) to warm the transposition table, and stops when the next request arrives. Off by default. It keeps the game's threads busy on the opponent's turn too, so both games of a bot match then keep all their threads busy all the time instead of only when both are ours to move.
- Each reply carries `used_ms` and `hard_ms` (the stop cap), which the connector prints locally, and the thread count, hash size and `--ponder` setting, which it reports to FlipSight. `--level` is not used.
- A searched move also carries `eval`, the search score in discs from the mover's side, which connector 1.5.3 or later sends with the move in bot matches. A pass or a single legal move is answered without a search and has no `eval`.

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
- `--threads <number>` - Set the number of threads to use for search (default: number of CPU cores, capped at 64; explicit values must be 1 to 64 and are lowered to the CPU count)
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
- Analysis data after `;` must be `move:score` pairs; a line with a malformed pair is skipped
- A line that cannot be parsed is reported on stderr and skipped. The remaining lines are still solved, and `solve` then exits with status 1; it does the same when the file holds no position

Example position line:

```text
O--OOOOX-OOOOOOXOOXXOOOXOOXOOOXXOOOOOOXX---OOOOX----O--X-------- X; A2:+38
-OOOOO----OOOOX--OOOOOO-XXXXXOO--XXOOX--OOXOXX----OXXO---OOO--O- X; H4:+0
```

### Version and License Information

Display version information:

```bash
cli version
```

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
The native NSH GUI bundle is generated from the same configuration without
JavaScript dependencies. Regenerate both bundles after changing dependencies with:

```bash
cargo make licenses
```

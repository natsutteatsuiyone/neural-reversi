# Problem Files

Endgame and benchmark problem sets used by `search-test` and `cli solve`.

## OBF Problem Sets (`search-test`)

OBF (Othello Board Format) problem files from [Edax](https://github.com/abulmo/edax-reversi). `search-test` loads all `.obf` files in this directory by default, or specific sets via `--problem`:

- `fforum-1-19.obf`, `fforum-20-39.obf`, `fforum-40-59.obf`, `fforum-60-79.obf` — FFO Forum endgame positions with known exact per-move scores (`--problem fforum`).
- `hard-20.obf`, `hard-25.obf`, `hard-30.obf` — Edax hard positions at 20, 25, and 30 empty squares (`--problem hard-20`, etc.). Lines end with a `--:-127;` sentinel for uncomputed moves, which is ignored by `reversi-core`'s OBF parser.

## Plain Board Positions (`cli solve`)

- `small-35.txt` — 30 positions with 35 empty squares in `<64-char board> <side>` format (no move scores). Used with `cli solve` (for the `small-35` benchmark in `docs/`). Not readable by `search-test`, which only loads `.obf` files.

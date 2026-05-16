# reversi-core

The search and evaluation engine of Neural Reversi. It solves Reversi with
bitboards, parallel αβ search, and a pattern-feature neural-network
evaluator. This document describes the engine internals.

## Big picture

The engine is split into three layers.

```
┌────────────────────────────────────────────────────────┐
│  Search                                                │
│   ├─ midgame:  iterative deepening + aspiration        │
│   └─ endgame:  iterative selectivity (Level1 → None)   │
│        └─ Negamax PVS + (TT / ETC / ProbCut /          │
│                          stability / LMR / wipeout)    │
│              └─ YBWC parallel split (split point +     │
│                                      helper threads)   │
├────────────────────────────────────────────────────────┤
│  Evaluation                                            │
│   ├─ pattern feature: 32 patterns × base-3 encoding    │
│   │     (incrementally updated on every move/flip)     │
│   ├─ Network      : per-ply layer stacks (main NN)     │
│   ├─ NetworkSmall : endgame NN                         │
│   └─ EvalCache    : direct-mapped cache                │
├────────────────────────────────────────────────────────┤
│  Board / Bitboard                                      │
│   └─ two u64 bitboards (player, opponent)              │
│        flip / get_moves / hash …                       │
└────────────────────────────────────────────────────────┘
```

The entry point is `Search::run`. It compares the number of empty squares
against the `Level`'s `min_end_depth` and dispatches to either
`midgame::search_root` or `endgame::search_root`. Both share the same
`search()` negamax core, while the `SearchStrategy` trait
(`MidGameStrategy` / `EndGameStrategy`) supplies the constants and pruning
thresholds that distinguish the two phases.

## Board representation

`Board` carries nothing but two bitboards.

```rust
pub struct Board {
    pub player:   Bitboard,  // discs of the side to move
    pub opponent: Bitboard,  // discs of the other side
}
```

- `Bitboard(u64)` is `#[repr(transparent)]`. Bit 0 = A1 … bit 63 = H8.
- A pass is just `switch_players()` (swap player and opponent).
- The position hash is `Board::hash`; it doubles as the TT key.
- The inner search loop uses `make_move_with_flipped(flipped, sq)` so the
  result of `flip` is computed once and reused by both `make_move` and the
  evaluator.
- Final disc difference is `final_score(_scaled)` for full boards and
  `solve(_scaled)` when both sides have passed.

Game-level concerns (history, automatic passing, undo) are confined to
`GameState`. `Board` itself is a pure value type with no history.

## Move generation and flipping

`flip::flip(sq, p, o)` returns "the bitboard of opponent discs that flip when
the side to move plays at `sq`". The x86‑64 implementations use the
LR-mask table approach; the active variant is selected at compile time:

| target_feature                 | Implementation             |
| ------------------------------ | -------------------------- |
| `avx512cd` + `avx512vl`        | `flip/flip_avx512.rs`      |
| `avx2`                         | `flip/flip_avx2.rs`        |
| `neon` (aarch64)               | `flip/flip_neon.rs`        |
| none                           | `flip/flip_portable.rs`    |

The disc-count update for the very last empty square has a separate path
(`count_last_flip/`) used at endgame leaves, where the specialised
implementation skips work that the generic `flip` cannot avoid.

Legal-move generation goes straight from the player/opponent bitboards via
`Board::get_moves`, and `MoveList` enumerates them. `empty_list` keeps the
empty squares in a doubly-linked list; endgame search walks it to step to
the next empty square in O(1).

## Search

### The core: Negamax + PVS

`search<NT, SS>` is the shared body for midgame and endgame. `NT: NodeType`
selects between `Root / PV / NonPV`, and `SS: SearchStrategy` selects
midgame vs. endgame; both are generic so the branches collapse at compile
time (the `cut_node` flag is also threaded through). One node, in order:

1. **TT prefetch** — derive the key with `board.hash()` and software-prefetch
   the slot.
2. **Move generation** — `MoveList::new(board)`.
3. **Pass handling** — if no legal moves, `switch_players()` and recurse;
   if neither side moves, return `solve_scaled(n_empties)` for the terminal
   score.
4. **Wipeout shortcut** — if any move flips every opponent disc, return
   `MAX` immediately.
5. **TT probe** — in NonPV nodes, `TTEntryData::can_cut` may produce an
   immediate cutoff. In PV nodes only the TT best move is taken, for use in
   move ordering.
6. **ETC** (Enhanced Transposition Cutoff) — peek at children through the TT
   to claim a cutoff one ply earlier. Active when `depth >= SS::MIN_ETC_DEPTH`.
7. **ProbCut** — `SS::try_probcut`; details below.
8. **TT move first** — in NonPV nodes with a TT move, search the TT move at
   a null window before running move ordering. A fail-high here cuts off
   without sorting the rest.
9. **Move ordering** — `MoveList::evaluate_moves` scores the rest;
   `MoveList::sort` arranges them. The TT move is always pinned at index 0.
10. **Main loop**:
    - First move (PV): full window.
    - Later moves: `compute_lmr_reduction` picks a reduction; search at a
      null window. If it fails high, re-search at full depth (still null
      window). For PV nodes, if `score > alpha`, re-search at full depth and
      full window.
    - When the conditions are met, `Thread::split` hands the remaining moves
      to helper threads (see below).
11. **TT store** — pack `best_score / best_move / bound / depth /
    selectivity / is_endgame` and write back.

`Bound` is `None=0 / Lower=1 / Upper=2 / Exact=3`. The bit patterns are
chosen so the bound check inside `TTEntryData::can_cut` is a bitwise AND.

### Midgame root (`midgame::search_root`)

Iterative deepening with aspiration windows.

- The opening position (60 empties) returns a random move immediately
  unless `multi_pv` is requested.
- The deepening step shrinks from `+2` plies to `+1` once `depth` exceeds
  `DEPTH_STEP_THRESHOLD`.
- From `ASPIRATION_MIN_DEPTH` onwards, `aspiration_search` opens with the
  previous score ± `ASPIRATION_DELTA`; on a fail-high or fail-low it
  widens and re-searches.
- Multi-PV walks `set_pv_idx(i)` over root moves and pins one window at a
  time.
- After every iteration, `time_manager.report_iteration` is fed the current
  best move / score / depth so it can drive extensions and early-exit
  decisions.

LMR runs in non-PV midgame nodes once selectivity is enabled and the move
is past the first few siblings. From `LMR_MIN_DEPTH` the reduction is one
ply; from `LMR_DEEPER_DEPTH` with sufficiently many earlier siblings it
grows to two.

### Endgame root (`endgame::search_root`)

Iterative selectivity. The depth is fixed (all remaining empties are
solved); what changes between iterations is the confidence level. The loop
walks `Level::ENDGAME_SELECTIVITY = [Level1, Level3, Level5, None]`,
ending with `None` for an exact solve.

- Before the loop, a shallow PVS estimates a centre score for the
  aspiration window of half-width `INITIAL_ASPIRATION_WINDOW`.
- Between selectivity steps the window narrows by
  `INTER_SELECTIVITY_DELTA`. On a fail-high or fail-low it widens by
  `ASPIRATION_DELTA` and re-searches.
- The root loop keeps aspiration centers, windows, and search return values as
  `ScaledScore`; only the null-window and shallow endgame solvers convert to
  plain `Score` disc differences.
- `EvalMode::Small` is forced, so the endgame net runs.
- A pair of thread-local `EndGameCache`s accelerates shallow re-searches.

At depth `≤ DEPTH_TO_NWS` the search switches to a null-window-centric
specialisation; at `≤ DEPTH_TO_SHALLOW_SEARCH` empties it drops further
into a shallow variant.

### Transposition table

`TranspositionTable` is organised as small fixed-size clusters
(`CLUSTER_SIZE` slots each); the byte budget is `mb_size`. Each slot stores

- the full `Board` (raw 64-bit player / opponent),
- `TTDataFields` packed into 8 bytes (`score / best_move / bound / depth /
  selectivity / generation / is_endgame`),
- a 64-bit SeqLock sequence counter,

and probes / stores are lock-free.

- `score: i16` keeps a `ScaledScore` verbatim; `best_move: u8` is the raw
  `Square` discriminant.
- Hash collisions never produce false hits, because the comparison uses
  the full `Board`. They affect replacement only.
- `Search::run` calls `increment_generation`; the `generation` byte feeds
  the replacement policy so older entries are overwritten first.
- `prefetch(key)` is exposed for ETC and split-internal lookups.

### ProbCut

`probcut.rs`. The model assumes that a search at depth `deep` and a search
at depth `shallow` are correlated, so a shallow search alone can predict
a cutoff.

- The parameters per `(ply, shallow, deep)` are a linear / exponential fit:

  ```
  mean  = a + b·shallow + c·deep
  sigma = exp(a' + b'·shallow + c'·deep)
  ```

  `probcut::init` populates `MEAN_TABLE / SIGMA_TABLE` (midgame) and
  `MEAN_TABLE_END / SIGMA_TABLE_END` (endgame) into `OnceLock`s. Values are
  pre-multiplied by `ScaledScore::SCALE` so they live directly in
  `ScaledScore` units.
- The verification threshold is `compute_probcut_beta(beta, t, mean, sigma)
  = ceil(beta + t·sigma − mean)`. If a shallow PVS clears it, the deep
  search is skipped.
- `Selectivity` enumerates discrete confidence levels (`Level1` → `Level5`,
  ordered from most aggressive to most conservative) plus `None` for an
  exact solve. Per-level `t` multipliers and confidence percentages live
  in `Selectivity::CONFIG`. PV nodes and the final endgame confirmation
  push selectivity to `None`.

### Stability cutoff

`stability::stability_cutoff(board, n_empties, alpha)` cuts off when the
opponent has enough discs that can no longer be flipped to put the side
to move below alpha.

- An `EDGE_STABILITY` table covering all 256 × 256 edge configurations is
  generated at compile time. It is combined with detection of fully-filled
  same-colour ranks, files, and diagonals to lower-bound the stable count.

### YBWC parallel search

`src/search/threading.rs`. YBWC (Young Brothers Wait Concept).

- A split point is published from `Thread::split` when the search is past
  the first move, `depth >= SS::MIN_SPLIT_DEPTH`, at least two moves
  remain, and `Thread::can_split()` is true. Remaining moves are
  distributed through a shared `ConcurrentMoveIterator`.
- Helper threads that pick up the split point enter `search_split_point`,
  pull moves from the iterator, and write results back into the
  split-point state (`alpha / best_score / best_move / cutoff`) atomically.
- A single thread can own up to `MAX_SPLITPOINTS_PER_THREAD` splits at
  once, and a single split accepts up to `MAX_THREADS_PER_CUT_SPLITPOINT`
  (cut nodes) or `MAX_THREADS_PER_NON_CUT_SPLITPOINT` (non-cut) workers.
- A shared abort flag is polled at a short fixed interval
  (`CHECK_INTERVAL_MS`) so threads can exit on time-out or
  `Search::abort`.
- The physical thread count is
  `n_threads.min(num_cpus::get()).clamp(1, MAX_THREADS)`.

`SearchSharedResources` lets several `Search` instances share a single TT
and `Eval` while keeping independent `ThreadPool`s.

## Evaluation

`Eval` holds two neural networks and an evaluation cache. The entry point
is `Eval::evaluate(ctx, board)`.

```text
ply <  ENDGAME_START_PLY : main Network always
ply >= ENDGAME_START_PLY
  ├─ EvalMode::Main : main Network
  └─ EvalMode::Small: NetworkSmall (endgame-only)
```

### Pattern features

`eval/pattern_feature.rs`. The position is decomposed into 32 patterns.
Each pattern is collapsed into a single base-3 number
(player = 0, opponent = 1, empty = 2) that becomes a network input.

- Pattern set: inner 4-square × 2 blocks / diagonals / centre 2×4 /
  ranks and files / edge 2×4 / corner 3×3 / centre 3×3 / adjacent
  diagonals (`EVAL_F2X`).
- Eight-square patterns contribute `3⁸` input IDs each, nine-square
  patterns `3⁹`, and adjacent diagonals `3⁷`. Concatenated with per-pattern
  offsets, they form `INPUT_FEATURE_DIMS`.
- On every move and flip, the affected patterns are updated incrementally
  by walking `EVAL_X2F[sq]` (the patterns this square belongs to, with the
  base-3 weights of its position inside each). The cost is proportional to
  the number of changed squares.
- `EVAL_FEATURE` and `EVAL_X2F` are produced by `const fn`s and live in
  `.rodata`. Their construction (with worked-out A1 examples) is
  documented in `eval/pattern_feature.rs` on the table definitions.

### Main network

`eval/network.rs`. A fully-connected network with a layer stack swapped in
per ply.

```
                             ┌─ BaseInput
PatternFeature ─►            │
                  L1_input ──┼─ PhaseAdaptiveInput (ply-bucketed)
                             │
                             └─ mobility (get_moves().count() · MOBILITY_SCALE)
                       │
                       ▼
                LinearLayer L1 ─► sqr_clipped + clipped_relu
                       │
                       ▼
                LinearLayer L2 ─► screlu
                       │
                       ▼  + base_out + pa_out (skip connection)
                OutputLayer (i32 → ScaledScore)
```

- Weights are zstd-compressed `i16` blobs. `Network::from_reader` reads
  them in order: `BaseInput` → `PhaseAdaptiveInput` → one `LayerStack` per
  ply (`NUM_LAYER_STACKS` total).
- Input layers dispatch on `target_feature` to
  `forward_avx512 / forward_avx2 / forward_neon / forward_scalar`. The
  SIMD paths pre-permute the matrices (`simd_layout::permute_rows`) so
  loads stay contiguous.
- The working buffers (`NetworkBuffers`) are thread-local; one evaluation
  is computed in-place.
- The output is shifted right by `OUTPUT_WEIGHT_SCALE_BITS`, clamped to
  `±INF`, and saturated into `ScaledScore::MIN+1 ..= MAX-1`.

### Endgame network

`eval/network_small.rs`. Network responsible for the endgame plies (those
at or past `ENDGAME_START_PLY`).

- A small fixed number of input layers, each covering a contiguous slice
  of endgame plies, drives one output layer per ply.
- Hidden → output is a single clamped-ReLU stage; the activation clamp
  (`ACTIVATION_CLAMP_MAX`) gives the stage 10-bit precision.
- Bypasses the eval cache.

### Eval cache

`eval/eval_cache.rs`. Direct-mapped, `2^EVAL_CACHE_SIZE_LOG2` entries,
keyed by `Board::hash`. It records main-network evaluations only.
`Eval::prefetch(key)` is meant to be issued between `make_move` and the
next `evaluate` so the cache line load overlaps with the SIMD work in
between.

## ScaledScore

All in-search arithmetic is in `ScaledScore`:

```rust
#[repr(transparent)]
pub struct ScaledScore(i32);
// scale: 1 << SCALE_BITS
```

- `from_disc_diff(d) == d << SCALE_BITS`, `to_disc_diff() == >> SCALE_BITS`,
  and `to_disc_diff_f32` for a real-valued disc difference.
- Special values: `ZERO`, `MIN` / `MAX` (full-board disc differences as
  scaled values), and `INF = i16::MAX` (the αβ sentinel — bounded by
  `i16` because the TT packs the score in 16 bits).
- `Add / Sub / Neg` between `ScaledScore`s, plus `Add<i32> / Mul<i32> /
  Div<i32>`. This lets ProbCut expressions like `beta + t·sigma − mean`
  stay in `ScaledScore` without conversions.
- The fixed-point scale exists because the disc difference produced by a
  move is integer, but root-move ordering needs sub-disc precision to
  break ties. Conversions to `Score` / `Scoref` happen at the boundary
  (display, GTP output, multi-PV reporting).

## Levels and time control

### Level

`Level { mid_depth, end_depth: [Depth; 4] }`. The `end_depth` slots line
up with `Level::ENDGAME_SELECTIVITY` (`Level1 / Level3 / Level5 / None`).

- `get_level(i)` returns a preset configuration for UIs (see the array at
  the bottom of `level.rs`).
- `Level::unlimited()` is the baseline used in time-control mode: an
  effectively unlimited `mid_depth` paired with a moderate `end_depth`
  cap. Once a previous time-controlled `Search::run` has reached the
  endgame phase (recorded by `update_endgame_tracking`), subsequent calls
  lift the `end_depth` to `Level::perfect()`'s value so the endgame is
  solved exactly.

### Time control

When `SearchConstraint::Time(TimeControlMode)` is selected, `TimeManager`
distributes time across the move.

| Mode           | Fields                                  |
| -------------- | --------------------------------------- |
| `Infinite`     | —                                       |
| `Byoyomi`      | `time_per_move_ms`                      |
| `Fischer`      | `main_time_ms` + `increment_ms`         |
| `MovesToGo`    | `time_ms` + remaining `moves`           |
| `JapaneseByo`  | `main_time_ms` + `time_per_move_ms`     |

- An asymmetric Gaussian over the remaining empty count (peaked in the
  midgame, narrower toward the endgame) yields a per-move weight; the
  budget for the current move is its share of the sum over remaining
  moves.
- After each iteration, `report_iteration(best_move, score, depth)` is
  consumed and the manager grants an extension when either
  - the score has dropped by at least `SCORE_DROP_THRESHOLD`, or
  - the best move is unstable.

  Up to `MAX_EXTENSION_STEPS` extensions are granted in total. The
  cumulative cap pushes `max_time` no further than `base_max_time +
  EXTENSION_RESERVE_RATIO · (hard_time_limit − base_max_time)`.
- A `TIME_BUFFER_MS` safety buffer is always subtracted to avoid time
  forfeit.
- `STABILITY_THRESHOLD` consecutive iterations with the same best move
  qualify the search for early termination.

## Weight files

`Eval::with_weight_files(eval, eval_sm)` loads two zstd-compressed weight
blobs. Explicit paths are loaded when provided; `None` uses bytes embedded
into the binary at build time via `include_bytes!`.

`SearchSharedResources::new` passes `SearchOptions::with_eval_paths(...)`
directly to `Eval::with_weight_files`, so default `Search::new` uses embedded
weights unless paths are configured. `Eval::new` is the constructor that first
looks next to the executable for `EVAL_FILE_NAME` / `EVAL_SM_FILE_NAME`, then
falls back to embedded weights.

Filenames carry a hash: `eval-XXXXXXXX.zst` / `eval_sm-XXXXXXXX.zst`.
Released weights live at
[neural-reversi-weights](https://github.com/natsutteatsuiyone/neural-reversi-weights/releases),
and the training code at
[neural-reversi-training](https://github.com/natsutteatsuiyone/neural-reversi-training).
The integration tests in `tests/` require real weights to load.

## Module map

| Concern                              | Location                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------- |
| Bitboard                             | `src/bitboard.rs`                                                         |
| Board / squares / discs              | `src/board.rs`, `src/square.rs`, `src/disc.rs`                            |
| Flip dispatch                        | `src/flip.rs` + `src/flip/*`, `src/count_last_flip{.rs,/}`                |
| Move list / empty list               | `src/move_list.rs`, `src/empty_list.rs`                                   |
| Game state                           | `src/game_state.rs`                                                       |
| Search core                          | `src/search.rs`                                                           |
| Midgame search                       | `src/search/midgame.rs`                                                   |
| Endgame search                       | `src/search/endgame.rs`, `src/search/endgame_cache.rs`                    |
| Parallelisation                      | `src/search/threading.rs`                                                 |
| Search context / stack               | `src/search/search_context.rs`, `src/search/search_stack.rs`              |
| Strategy / node type                 | `src/search/search_strategy.rs`, `src/search/node_type.rs`                |
| Result / root moves / Multi-PV       | `src/search/search_result.rs`, `src/search/root_move.rs`                  |
| Time control                         | `src/search/time_control.rs`                                              |
| TT / ProbCut / Stability             | `src/transposition_table.rs`, `src/probcut.rs`, `src/stability.rs`        |
| Evaluator / cache                    | `src/eval.rs`, `src/eval/eval_cache.rs`                                   |
| Pattern features                     | `src/eval/pattern_feature.rs`                                             |
| Main NN                              | `src/eval/network.rs`, `src/eval/network/*`                               |
| Endgame NN                           | `src/eval/network_small.rs`                                               |
| Numeric types / constants            | `src/types.rs`, `src/constants.rs`                                        |
| Utilities                            | `src/util/{align,bitset,spinlock}.rs`                                     |
| Correctness checks                   | `src/perft.rs`, `tests/perft_tests.rs`, `tests/endgame_tests.rs`          |

## Build and test

```bash
cargo check -p reversi-core
cargo test  -p reversi-core
cargo bench -p reversi-core

cargo bench -p reversi-core --bench network
cargo bench -p reversi-core --bench perft
cargo bench -p reversi-core --bench bitboard
cargo bench -p reversi-core --bench flip
cargo bench -p reversi-core --bench count_last_flip
cargo bench -p reversi-core --bench stability
cargo bench -p reversi-core --bench move_list
```

`tests/perft_tests.rs` checks the move-generation node counts;
`tests/endgame_tests.rs` checks both the score and the best move on a set
of endgame positions. Both require the `.zst` weight files in the
workspace root.

Enabling the `search-stats` Cargo feature populates extra fields in
`SearchCounters`, exposing breakdown counters such as TT hit rate, ETC
attempt/cutoff count, ProbCut attempt/cutoff count, and stability cut count.

# Stop-token, abort-generation, and timer protocol model (TLA+)

Bounded PlusCal/TLA+ model check of generation-scoped search cancellation across consecutive searches in `crates/reversi-core`, using small wrapping bounds (`GenMod = 4`, `MaxRuns = 5 > GenMod`) to exercise `u32` generation wraparound (`0xFFFF_FFFF -> 0`).

See [`../README.md`](../README.md) for prerequisites, shared runner behavior, CI coverage, and `trace.py`. For the 64-bit bit-vector contract of `AbortState`, see [`SearchAbort.lean`](../../lean/NeuralReversi/Threading/SearchAbort.lean).

## Covered Rust code

- [`crates/reversi-core/src/search/stop.rs`](../../../crates/reversi-core/src/search/stop.rs): `StopToken`, `StopState`, `StopAttachment` (`stop`, `attach`, `Drop`)
- [`crates/reversi-core/src/search/threading/abort.rs`](../../../crates/reversi-core/src/search/threading/abort.rs): `AbortState` (`new`, `begin_search`, `request_abort`, `request_abort_for`, `is_aborted`)
- [`crates/reversi-core/src/search/threading/pool.rs`](../../../crates/reversi-core/src/search/threading/pool.rs): `start_thinking`, `start_timer`, `timer_loop`, `stop_timer`
- [`crates/reversi-core/src/search/threading/thread.rs`](../../../crates/reversi-core/src/search/threading/thread.rs): `main_thread_loop` dropping `StopAttachment` before sending `SearchResult`
- [`crates/reversi-core/src/search.rs`](../../../crates/reversi-core/src/search.rs): `execute_search` (`start_thinking` -> `recv` -> `stop_timer`)
- [`crates/reversi-core/src/search/time_control.rs`](../../../crates/reversi-core/src/search/time_control.rs): `TimeManager::check_time_for`

## Run

```bash
python3 verification/tla/stop-abort/check.py            # default bounds (GenMod=4, MaxRuns=5, NumTokens=2)
python3 verification/tla/stop-abort/check.py --deep     # two full generation wraps (MaxRuns=9, NumTokens=2)
python3 verification/tla/stop-abort/check.py --mutants  # verify every mutant fails its target property
python3 verification/tla/stop-abort/check.py --all      # clean + deep + all mutants
```

## Model overview

- **Wrapping `AbortState` word**: `(gen, abortGen) \in (0..GenMod-1)^2`, initialized to `(0, GenMod - 1)` matching `AbortState::new()`. `begin_search` atomically sets `gen' = (gen + 1) % GenMod` and `abortGen' = gen`; `request_abort_for(target)` writes `abortGen' = target` only when `gen = target`; `IsAborted` holds iff `gen = abortGen`.
- **`Runner`**: executes `1..MaxRuns` sequential searches (`begin_search` -> optional `StopToken::attach` -> optional `start_timer` -> `is_aborted` polling -> `StopAttachment::drop` -> `stop_timer` flag + join).
- **Concurrent actors**:
  - `StopCaller \in 1..NumTokens`: invokes `StopToken::stop()` at any point (before attach, during search, after detach, after wraparound, or never).
  - `Timer`: models `timer_loop`, either firing `check_time_for(armedGen)` or exiting upon observing `timer_stop = true`.
  - `StaleAbortCaller`: models a delayed `request_abort_for(prev_gen)` targeting the previous generation after `begin_search()` has advanced `gen`.

## Checked properties

### Safety invariants

- `TypeOK`: well-typed state domains and attachment bookkeeping.
- `NoStaleAbortLeak`: an earlier run's `StopToken::stop()`, unjoined/unscoped timer, or stale `request_abort_for(prev_gen)` never aborts a different or subsequent run.
- `FreshRunNotSpuriouslyAborted`: when no abort was requested for the active run, `IsAborted` is `FALSE` — including across the `GenMod - 1 -> 0` wraparound boundary.
- `StoppedTokenReachesAttached`: once `StopToken::stop()` completes and an active run is attached to that token (regardless of `stop()` vs `attach()` order), `IsAborted` is `TRUE`.

### Liveness and deadlock

- `EventuallyTerminates == <>AllRunsDone`: under weak fairness (`fair process`), every search run completes, detaches its `StopAttachment`, and joins its timer thread without deadlock or livelock.

## Mutants

| Mutant | Simulated regression | Expected catch |
| --- | --- | --- |
| `wrap_keeps_old_abort_marker` | `begin_search` increments `gen` without resetting `abort_gen` to `prev_gen`, colliding with the initial marker after wrap | `FreshRunNotSpuriouslyAborted` |
| `attach_skips_pre_stopped_check` | `StopToken::attach` pushes to `runs` without checking `if state.stopped` | `StoppedTokenReachesAttached` |
| `detach_does_not_remove` | `StopAttachment::drop` omits removing `(abort_state, generation)` from `runs`, leaking an abort after generation wrap (ABA) | `NoStaleAbortLeak` |
| `skip_timer_join` | `stop_timer` sets `timer_stop` without joining `timer_handle`, letting an old timer survive a full generation wrap (ABA) | `NoStaleAbortLeak` |
| `unscoped_timer_abort` | `timer_loop` calls unscoped `request_abort()` while `stop_timer` does not join | `NoStaleAbortLeak` |
| `request_abort_for_ignores_gen` | `request_abort_for` drops its generation check and sets `abort_gen := gen` unconditionally | `NoStaleAbortLeak` |
| `request_abort_for_clobbers_abort` | `request_abort_for` drops its generation check and writes `abort_gen := target_gen`, erasing an active abort | `StoppedTokenReachesAttached` |
| `stop_timer_skips_flag` | `stop_timer` joins `timer_handle` without storing `timer_stop = true`, deadlocking before deadline expiry | `Deadlock` |

`MaxRuns > GenMod` is essential so generation reuse catches the wraparound ABA bugs (`detach_does_not_remove` and `skip_timer_join`).

## Scope and assumptions

- **Memory ordering**: PlusCal steps are sequentially consistent and do not model `Acquire`/`Release`/`AcqRel` reorderings; keep the `AcqRel`/`Acquire` orderings on `AbortState` and `Release`/`Acquire` on `timer_stop`.
- **`StopState` mutex**: `stop`, `attach`, and `Drop` acquire only `StopToken`'s leaf mutex without nesting or blocking, so each is modeled as an atomic step.
- **Single pool**: models one `ThreadPool` (`AbortState`) shared across consecutive serialized searches with multiple reusable `StopToken`s.

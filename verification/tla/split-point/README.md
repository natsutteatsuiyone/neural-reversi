# Split-point protocol model (TLA+)

Bounded PlusCal/TLA+ model check of the endgame YBWC split-point concurrency and panic-unwinding protocol in `crates/reversi-core/src/search/threading/` (`SplitUnwindGuard`, `UnwindCleanup`, `abandon_active_split`, late joins, cutoffs, nested splits, and slot reuse).
Because a split point's `task` and `move_iter` live in `SyncUnsafeCell`s, the protocol ensures helpers never read a cleared or reused slot.

See [`../README.md`](../README.md) for prerequisites, shared runner behavior, CI coverage, and `trace.py`. For memory-ordering witnesses, see [`../lean-witnesses/README.md`](../lean-witnesses/README.md).

## Covered Rust code

- [`crates/reversi-core/src/search/threading/thread.rs`](../../../crates/reversi-core/src/search/threading/thread.rs): `SplitUnwindGuard`, `UnwindCleanup`, `abandon_active_split`, `detach_split`, `can_join`, `book_into`, `split`, `initialize_split_point`, `finalize_split_point`, `idle_loop`, `search_active_split_point`, `try_late_join`, and cutoff epoch caching.
- [`crates/reversi-core/src/search/threading/split_point.rs`](../../../crates/reversi-core/src/search/threading/split_point.rs): `SplitPoint`, `SplitPointState`, `helpers_mask`, `all_helpers_searching`, `cutoff`, `task`, and `move_iter` lifetime rules.
- [`crates/reversi-core/src/search/threading/pool.rs`](../../../crates/reversi-core/src/search/threading/pool.rs): `assign_helpers_to_split_point`.

## Run

```bash
python3 verification/tla/split-point/check.py            # default bounds (N=3, MaxSP=2, Moves=2, Budget=2, MaxThr=2, MaxUnwinds=1)
python3 verification/tla/split-point/check.py --deep     # deeper nesting and slot reuse (N=2, Budget=3)
python3 verification/tla/split-point/check.py --mutants  # verify every mutant fails its target property
python3 verification/tla/split-point/check.py --all      # clean + deep + all mutants
```

## Checked properties

In addition to TLC's deadlock check and four in-model assertions (`searching[self]` on init, `~searching[self]` on finalize, `active[self] # NONE`, and `task[sa]` on read), [`SplitPoint.cfg`](SplitPoint.cfg) checks ten safety invariants:

- `TaskAliveWhileUsed`: every slot being read by a searcher still has `task[sp]` live at its observed `slotGeneration`.
- `BookedOnlyIntoLiveSplitPoints`: no slot with a non-empty `helpersMask` has a cleared `task`.
- `BookingMatchesThread`: every non-failed thread booked into a split point (before entering `SearchActive`) has `active[t] = sp` and `searching[t]`.
- `ActiveChainAlive`: every split point on any thread's `active` ancestor chain has a live `task`.
- `CutoffCacheSound`: a thread whose cutoff cache matches the current `epoch` with `~chainCut` cannot have a published cutoff on its active chain.
- `CutoffCacheNoFalseStop`: `chainCut[t]` is set only when some split point on `active[t]`'s chain is actually cut off.
- `CleanTeardown`: once `~thinking`, every split point is cleared with empty `helpersMask` and released locks, and every thread (including unwound threads) has `~searching`, `splitPointsSize = 0`, `active = NONE`, and empty `observedReading`.
- `ParentChainAcyclic`: `parent` links form no cycles.
- `SplitBudgetRespected` / `UnwindBudgetRespected`: split and unwind budgets never go negative.

## Mutants

| Mutant | Simulated regression | Expected catch |
| --- | --- | --- |
| `late_join_no_recheck` | `try_late_join` skips rechecking split-point state under `sp.lock()` | `BookedOnlyIntoLiveSplitPoints` |
| `leave_keeps_all_helpers_searching` | Normal leave skips clearing `all_helpers_searching` | `BookedOnlyIntoLiveSplitPoints` |
| `reset_bit_before_merge_counters` | Helper resets its `helpers_mask` bit before finishing reads/counter merge | `TaskAliveWhileUsed` |
| `can_join_ignores_searching` | `can_join` ignores `searching` | `BookingMatchesThread` |
| `leave_keeps_active` | Normal leave does not clear `active_split_point` | `ActiveChainAlive` |
| `cutoff_check_only_own_split_point` | Cutoff walk checks only `active_split_point`, not ancestors | `CutoffCacheSound` |
| `leave_no_cache_reset` | Normal leave skips `reset_cutoff_cache` | `CutoffCacheNoFalseStop` |
| `finalize_skips_clear` | `finalize_split_point` skips clearing `task` and `parent` | `CleanTeardown` |
| `hold_lock_across_idle_loop` | `initialize_split_point` holds `sp.lock()` into `idle_loop` | `Deadlock` |
| `unwind_skips_helper_bit_reset` | `abandon_active_split` skips `helpers_mask.reset(self.idx)` | `Deadlock` |
| `unwind_clears_slot_before_helpers_leave` | `SplitUnwindGuard::drop` clears the slot without draining `helpers_mask` | `BookedOnlyIntoLiveSplitPoints` |
| `unwind_keeps_active` | `abandon_active_split` leaves `active_split_point` set on unwind | `ActiveChainAlive` |
| `unwind_can_join_ignores_failed` | `can_join` ignores `failed`, allowing an unwinding or exited thread to be rebooked | `Deadlock` |

## Scope and assumptions

- **Memory ordering**: PlusCal steps are sequentially consistent and do not model `Relaxed`/`Acquire`/`Release` reorderings; `AtomicBitSet` and `SpinLock` orderings are checked separately in Lean and [`../lean-witnesses/`](../lean-witnesses/README.md).
- **Panic unwinding**: bounded fault injection (`MaxUnwinds = 1`) models panics after initial split booking and inside `search_split_point` (including nested splits), unwinding through `UnwindCleanup` and `SplitUnwindGuard`.
- **Outside this model**: search values, `SearchCounters` contents, spinlock backoff, Lazy SMP jobs, multiple searches per pool, and pool shutdown (see [`../thread-pool/README.md`](../thread-pool/README.md)).

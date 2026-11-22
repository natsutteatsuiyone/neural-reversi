# Split-point protocol model (TLA+)

Bounded TLC model check of the endgame split-point concurrency protocol in `crates/reversi-core/src/search/threading/`.
A split point's `task` and `move_iter` live in `SyncUnsafeCell`s, so the protocol prevents helpers from reading a cleared or reused slot.

This checks a hand-written model under small bounds, not the compiled Rust binary.
For machine-checked proofs of other `reversi-core` models, see the [Lean proofs](../../lean/README.md).

## Run

Run from the repository root with Java 11 or later and Python 3.10 or later on `PATH`:

```bash
python3 verification/tla/split-point/check.py            # default bounds (must print PASS)
python3 verification/tla/split-point/check.py --deep     # N = 2, Budget = 3 (nesting and slot reuse)
python3 verification/tla/split-point/check.py --mutants  # every mutant must fail its target invariant
cargo make verify-tla                                    # runs all three (check.py --all)
```

[`check.py`](check.py) caches `tla2tools.jar` `v1.7.4` (SHA-256 pinned) under `~/.cache/tlaplus/v1.7.4/`, runs TLC in `target/tla/split-point/<run>/`, and writes the full log to `tlc.out`.
Pass `--mutant <name>` to run one mutant or `--workers <n>` to set TLC's worker count.
[`.github/workflows/verify-tla.yml`](../../../.github/workflows/verify-tla.yml) runs `check.py` and `check.py --mutants` when `verification/tla/**` or a covered `reversi-core` file changes.

Expected state counts (2026-11-22, 32 workers):

- Default (`N = 3, Budget = 2`): 632,251 distinct states (2,120,017 generated), depth 103, 5 s.
- `--deep` (`N = 2, Budget = 3`): 169,635 distinct states (420,802 generated), depth 141, 1 s.

## Where things live

- [`SplitPoint.tla`](SplitPoint.tla): PlusCal model of split initialization, helper assignment, late join, cutoff propagation, and teardown, plus four mirrored assertions and nine safety invariants.
- [`SplitPoint.cfg`](SplitPoint.cfg): Default bounds (`N = 3`, `MaxSP = 2`, `Moves = 2`, `Budget = 2`, `MaxThr = 2`) and checked invariants.
- [`check.py`](check.py): Runner for clean, `--deep`, and mutant runs (`MUTANTS` lists every simulated regression and its expected violation).
- [`trace.py`](trace.py): Prints a TLC counterexample one step per line (`python3 verification/tla/split-point/trace.py target/tla/split-point/<run>/tlc.out`).

## Scope and assumptions

- **Memory ordering.** TLC assumes sequential consistency and does not check `Relaxed`, `Acquire`, or `Release` orderings. Keep `AtomicBitSet::none` (`Acquire`), `AtomicBitSet::reset` (`Release`), and `SpinLock::unlock` (`Release`), which a former loom model (removed in `af8a332d`) proved necessary.
- **Bounds.** Default bounds exercise the per-split-point cap and `can_split`; `--deep` adds nesting with slot reuse. To run wider bounds, edit [`SplitPoint.cfg`](SplitPoint.cfg) temporarily and keep the committed default small.
- **Not modeled.** Search values, `SearchCounters` contents, spinlock backoff, Lazy SMP helper jobs, multiple searches per pool, pool shutdown, and panic unwinding (`SplitUnwindGuard`, `UnwindCleanup`).

## Maintenance

CI checks the unchanged model, so update [`SplitPoint.tla`](SplitPoint.tla) by hand whenever lock regions, atomics, or guards change in the covered `reversi-core` files (listed in [`.github/workflows/verify-tla.yml`](../../../.github/workflows/verify-tla.yml)).
When you add or remove a protocol guard, update `MUTANTS` in [`check.py`](check.py), rerun `cargo make verify-tla`, and update the expected state counts above.
If TLC reports a failure, inspect it with [`trace.py`](trace.py) and replay it against the Rust code before changing the model or code.

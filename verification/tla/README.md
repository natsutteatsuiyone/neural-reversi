# TLA+ and Quint concurrency verification

Bounded TLC model checks for the concurrency protocols in `crates/reversi-core`, plus a separate Quint -> TLC -> Lean witness replay suite.

These models check hand-written protocol specifications under small bounds rather than the compiled Rust binary. For machine-checked Lean 4 proofs and Rust differential checks, see [`verification/lean/README.md`](../lean/README.md).

## TLA+ suites

| Suite | Model | Covered `reversi-core` code | Focus |
| --- | --- | --- | --- |
| [`split-point`](split-point/README.md) | `SplitPoint.tla` | `search/threading/{split_point.rs,thread.rs}`, `util/{bitset.rs,spinlock.rs}` | Endgame YBWC split-point lifecycle, booking, cutoff propagation, slot reuse, and panic unwinding |
| [`thread-pool`](thread-pool/README.md) | `ThreadPool.tla` | `search/threading/{pool.rs,thread.rs}` | `ThreadPool` lifecycle, Lazy SMP <-> YBWC mode switching, and `Arc<ThreadPool>` shutdown |
| [`stop-abort`](stop-abort/README.md) | `StopAbort.tla` | `search/stop.rs`, `search/threading/{abort.rs,pool.rs}`, `search/time_control.rs` | Scoped `StopToken` attachment, timer aborts, and wrapping `AbortState` generations |
| [`transposition-table`](transposition-table/README.md) | `TranspositionTable.tla` | `transposition_table.rs` | Lock-free SeqLock concurrent probe, save, clear, and replacement protocol |
| [`parallel-move-list`](parallel-move-list/README.md) | `ConcurrentMoveList.tla` | `move_list/{iterator.rs,move_array.rs}`, `search/threading/in_flight.rs`, `search/pvs.rs` | Weak-CAS `ConcurrentMoveIterator`, `InFlightTable` slot ownership, and bounded `Busy` postponement |

## Run

Run from the repository root with Java 11 or later and Python 3.10 or later on `PATH`:

```bash
python3 verification/tla/check.py                          # clean check across all 5 suites
python3 verification/tla/check.py --mutants                # run every mutant across all 5 suites
python3 verification/tla/check.py --deep                   # run deep bounds where configured
python3 verification/tla/check.py --all                    # clean + deep + all mutants
python3 verification/tla/check.py --suite stop-abort --all # run one suite
cargo make verify-tla                                      # runs verification/tla/check.py --all
```

Each suite can also be run directly via `python3 verification/tla/<suite>/check.py [--deep|--mutants|--mutant <name>|--all] [--workers <n>]`.

[`tlc_runner.py`](tlc_runner.py) caches `tla2tools.jar` `v1.7.4` (SHA-256 pinned) under `~/.cache/tlaplus/v1.7.4/`, runs PlusCal translation (`pcal.trans`) and TLC under `target/tla/<suite>/<run>/`, and writes the full TLC log to `tlc.out`.
[`.github/workflows/verify-tla.yml`](../../.github/workflows/verify-tla.yml) runs `verification/tla/check.py` and `verification/tla/check.py --mutants` when `verification/tla/**` or a covered `reversi-core` file changes.

## Quint -> TLC -> Lean witness replay

[`lean-witnesses/`](lean-witnesses/README.md) holds the nine Quint models (`*.qnt`), TLC replay drivers (`check_*.py`), and Lean replay scripts (`Replay*.lean`) that extract positive and negative traces from TLC and verify them against the Lean 4 interleaving theorems.

```bash
npm exec --yes --package=@informalsystems/quint@0.34.0 -- python3 verification/tla/lean-witnesses/check.py
```

## Shared files

- [`tlc_runner.py`](tlc_runner.py): Shared `SuiteSpec`, `tla2tools.jar` fetcher/verifier, PlusCal translator, TLC runner, and mutant matcher.
- [`check.py`](check.py): Top-level runner that dynamically loads `SUITE` from all five TLA+ suites.
- [`trace.py`](trace.py): Pretty-prints a TLC counterexample one step per line (`python3 verification/tla/trace.py target/tla/<suite>/<run>/tlc.out`).
- [`quint_tlc.py`](quint_tlc.py): Shared Quint-to-TLA+ compiler, TLC witness extractor, and Lean replay helper used by [`lean-witnesses/`](lean-witnesses/README.md).

## Maintenance

Update the corresponding `.tla` model and mutant table whenever locks, atomics, generation markers, or guards change in a covered `reversi-core` file (listed in [`.github/workflows/verify-tla.yml`](../../.github/workflows/verify-tla.yml)), then rerun `cargo make verify-tla`.
If TLC reports a violation, inspect the trace with [`trace.py`](trace.py) and compare it against the Rust implementation before editing the model or code.

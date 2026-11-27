# Endgame split result protocol (TLA+)

Bounded TLC model check of the numerical return contract of an endgame split, composing the parts previously checked separately: move claims, stale alpha observations, PV re-search, locked score publication, worker retirement, and the owner's post-split stop check.

The existing [`split-point`](../split-point/README.md) suite checks task lifetime and explicitly excludes search values. [`parallel-move-list`](../parallel-move-list/README.md) checks unique dispatch and bounded postponement. This suite checks that their assumed delivery/retirement behavior, combined with the current score protocol, produces a sound fail-soft result. It complements the conditional, unbounded Lean contracts in [`SplitResult.lean`](../../lean/NeuralReversi/Threading/SplitResult.lean), [`SplitWindow.lean`](../../lean/NeuralReversi/Threading/SplitWindow.lean), and [`SplitReturn.lean`](../../lean/NeuralReversi/Threading/SplitReturn.lean).

## Rust correspondence

- [`search/pvs.rs`](../../../crates/reversi-core/src/search/pvs.rs), `search_split_point`: initial shared-alpha observation, NonPV scout, PV retry decision against **the scout's original alpha**, fresh alpha for the full-window retry, `split_point.lock()`, stop check, `best_score`/`alpha` updates, and cutoff marking.
- [`search/threading/thread.rs`](../../../crates/reversi-core/src/search/threading/thread.rs), `split`/`idle_loop`: owner waits for all participating workers before observing the result.
- [`search/pvs.rs`](../../../crates/reversi-core/src/search/pvs.rs), `search`: `should_stop_after_split()` rejects the returned tuple before the enclosing node accepts/stores it.

The model deliberately separates scout completion, retry decision, fresh-alpha load, retry completion, locked stop checking, publication, owner readback, and owner stop checking. A sibling can therefore raise alpha between these steps, and an external stop can happen between a successful locked stop check and score publication, or after the owner read but before its stop check.

## Run

```bash
python3 verification/tla/endgame-result/check.py --workers 2
python3 verification/tla/endgame-result/check.py --deep --workers 2
python3 verification/tla/endgame-result/check.py --mutants --workers 2
python3 verification/tla/endgame-result/check.py --all --workers 2
python3 verification/tla/check.py --suite endgame-result --all --workers 2
```

The default bounds are two workers and two remaining moves; `--deep` uses three remaining moves. Every initial state chooses a sound prefix result, PV or NonPV mode, and the applicable beta. The original alpha is chosen from `{-1, 0}`. NonPV beta is `originalAlpha + 1` (zero or one); PV beta is two or four. PV beta four exercises the maximum-score cutoff (`ScoreMax = 3`) without a beta cutoff. The negative-alpha cases also cover a stopped child returning dummy zero **above** its scout alpha, triggering a PV retry before the locked stop check rejects publication. NonPV also includes an empty prefix (`Sentinel = -2`) to cover a split before any serial move has completed.

Parent-perspective child true values from `-1..3` are materialized lazily on first `Bound` evaluation and folded into `nodeValue`, while `ReturnedBound` universally quantifies any unvalued moves over `-1..3`. Together with clearing dead per-worker scratch variables (`scoutAlpha`, `wideAlpha`, `score`, `moveVal`, `stopSeen`) and breaking initial worker symmetry via `w <= next` in `Claim(w)`, this is an exact state-space reduction of choosing all child values in `Init` while preserving `EventuallyCompletes` liveness soundness without TLC `SYMMETRY`.

Values and reports use the parent's perspective, after the negation of the child search return in Rust; `Bound` therefore uses the parent's window. The small scores are an ordered finite abstraction, not concrete scaled Rust scores. The model checks every permitted fail-soft child return, including inexact fail-low/fail-high bounds, instead of assuming all child searches return exact scores. When either `aborted` or local `cutoff` holds (matching Rust `thread.should_stop()`), `Scout(w)` and `Wide(w)` allow the child either to unwind early with dummy zero or to return a sound `Bound` value (e.g., a leaf or TT hit that completes without polling `should_stop()`). Setting `cutoff` in `Commit(w)` retires that worker immediately, matching `if cutoff { break; }` in `search_split_point`.

## Checked properties

- `ReturnedBound`: every accepted result is a valid fail-soft result for the maximum of the prefix value and all child values: fail-low returns an upper bound, fail-high returns a lower bound, and an in-window return equals that maximum.
- `ReturnedCoverage`: acceptance requires every move to have published, a beta-cutoff witness, or the known maximum score.
- `OwnerReadAfterDrain`: readback occurs only after every worker retires; exhausting the iterator alone is insufficient.
- `NoInterruptedPublication`: a worker never publishes when its own locked stop check observed cancellation. A stop that arrives **after** that check may precede the stores; the owner must still reject that interrupted split.
- `NoInterruptedAcceptance`: the owner rejects an interrupted split even when interruption happens after readback.
- `SearchWindow`: every refreshed alpha remains below beta.
- `LockOwnership`: only the worker holding the split lock can commit a score.
- `CutoffWitness`: every marked local cutoff retains a beta or maximum-score witness.
- `EventuallyCompletes`: under weak fairness of each worker and owner step, the owner and all workers finish.
- `TypeOK` and TLC's deadlock check validate the state domains and terminal behavior.

## Mutation checks

Each intentionally broken model must fail the named property. These are verification regressions, not changes to the Rust implementation.

| Mutant | Regression | Required counterexample |
| --- | --- | --- |
| `skip_pv_retry` | Publish a scout's fail-high lower bound as an exact PV result | `ReturnedBound` |
| `retry_against_new_alpha` | Decide whether to retry using a sibling's newer alpha, rather than the scout window's alpha | `ReturnedBound` |
| `read_before_workers_drain` | Read best score as soon as the iterator is exhausted, while a claimed move is still executing | `OwnerReadAfterDrain` |
| `publish_after_abort` | Omit the worker's locked stop check | `NoInterruptedPublication` |
| `accept_after_abort` | Omit the owner's post-split stop check | `NoInterruptedAcceptance` |

The fresh-alpha mutant is also covered by an unbounded integer counterexample family in [`SplitWindow/Regression.lean`](../../lean/NeuralReversi/Threading/SplitWindow/Regression.lean). For example, after a scout against alpha zero returns the lower bound one, a sibling can raise alpha to one while the first child's actual value is three. Skipping the retry because the scout no longer exceeds current alpha can incorrectly return one.

## Scope and assumptions

- **Bounded model checking:** success covers the configured finite states and interleavings. It is not an unbounded theorem about any number of threads or moves.
- **Child contract:** uninterrupted children obey the fail-soft contract and the modeled score cap. Stopped children (under either external `aborted` or local `cutoff`, matching `thread.should_stop()`) may unwind early and return the dummy zero used by Rust or still finish normally with a sound `Bound` score. ProbCut/selective search statistical error is outside this exact-search contract; apply it only when the child result contracts are sound, such as exact endgame search with sound TT bounds.
- **Dispatch and lifetime:** `Claim` abstracts an atomic unique iterator claim. It does not reproduce weak-CAS retries, deferred serial move-list rotation, in-flight hash collisions, nested slots, or panic unwinding. Their independent models remain necessary. At the next claim point, a stopped worker may retire immediately because any additional child result would be rejected by its subsequent stop check.
- **Memory ordering:** actions use sequential consistency. Relaxed alpha loads are represented by an earlier snapshot that may become stale as workers progress; arbitrary weak-memory read-from choices and release/acquire visibility need the separate Lean memory-order contracts.
- **Publication:** the split-point lock serializes publishers. Stop checking and the subsequent score/alpha/cutoff stores are separate actions; the global/ancestor stop writer is not protected by this lock. Root move/PV payloads, TT storage, and interrupted iterative-deepening fallback are outside this suite.
- **Cancellation:** `Stop` represents an external/global/ancestor stop, while `cutoff` represents this split's local numeric cutoff; both trigger `thread.should_stop()` for in-flight child searches and locked publication checks. Once the owner accepts the result, future searches and later stop requests are outside this modeled return boundary.

See [`../README.md`](../README.md) for the pinned TLC runner, logs, CI integration, and counterexample rendering.

# Quint -> TLC -> Lean witness replay suite

Bounded Quint models compiled to TLA+ (via Apalache `0.62.4`), model-checked with TLC, and replayed through the Lean 4 interleaving proofs under [`verification/lean/`](../../lean/README.md).
Each driver verifies the unmodified model's safety invariant in TLC and extracts positive and negative (mutant) JSON witnesses that the corresponding `Replay*.lean` script accepts or rejects in Lean.

## Run

Run from the repository root with Java 21 or later, Node.js, and the repository's Rust and Lean toolchains on `PATH`:

```bash
npm exec --yes --package=@informalsystems/quint@0.34.0 -- python3 verification/tla/lean-witnesses/check.py
npm exec --yes --package=@informalsystems/quint@0.34.0 -- python3 verification/tla/lean-witnesses/check.py --suite cut-boundary --suite worker-startup
```

Individual drivers can also be invoked directly (`check_<name>.py`).
Shared compilation, TLC witness extraction, and Lean replay helpers live in [`../quint_tlc.py`](../quint_tlc.py); generated TLA+, TLC logs, and JSON witnesses are written under `target/tla/<group>/`.

## Suites

| Suite (`--suite`) | Quint model | Lean replay | Focus and mutants |
| --- | --- | --- | --- |
| `score-reuse` | [`ScoreReuse.qnt`](ScoreReuse.qnt) | [`ReplayScoreReuse.lean`](ReplayScoreReuse.lean) | Relaxed `best_score` read after observing zero `helpers_mask` across slot reuse (including identical `i32` scores) and Rust MIR alignment; rejects weakened waits/resets, missing owner ordering, and misattributed score writers. |
| `nested-cutoff` | [`NestedCutoff.qnt`](NestedCutoff.qnt) | [`ReplayNestedCutoff.lean`](ReplayNestedCutoff.lean) | Parent cutoff flag propagation and cached poll after restoring a reused parent slot; rejects stale positive writes at the same address and false stops on a clear chain. |
| `call-returns` | [`CallReturns.qnt`](CallReturns.qnt) | [`ReplayCallReturns.lean`](ReplayCallReturns.lean) | Matching repeated identical child calls to per-worker call stacks, including overlapping reverse-order returns; rejects out-of-order and cross-worker return substitution. |
| `tt-readback` | [`TTReadback.qnt`](TTReadback.qnt) | [`ReplayTTReadback.lean`](ReplayTTReadback.lean) | Candidate write identities for a relaxed TT payload load across concurrent writes with duplicate values; rejects latest-write-only rules and missing same-worker upper bounds. |
| `parent-mask-zero` | [`ParentMaskZero.qnt`](ParentMaskZero.qnt) | [`ReplayParentMaskZero.lean`](ReplayParentMaskZero.lean) | Identifying which physical zero write an owner acquires on `helpers_mask` across ordinary and redundant resets; compares the generation checker against all valid atomic read sources. |
| `worker-startup` | [`WorkerStartup.qnt`](WorkerStartup.qnt) | [`ReplayWorkerStartup.lean`](ReplayWorkerStartup.lean) | Helper startup across reused split-point slots via `ActivePointer.splitDispatch`; rejects stale generations and wrong worker locks. |
| `owner-active-sync` | [`OwnerActiveSync.qnt`](OwnerActiveSync.qnt) | [`ReplayOwnerActiveSync.lean`](ReplayOwnerActiveSync.lean) | Synchronization between `ownerActive` and a subsequent foreign `book_into`; rejects unsynchronized races (`raceBook`), wrong worker locks (`wrongLock`), and early release (`earlyRelease`). |
| `split-lifecycle` | [`SplitLifecycle.qnt`](SplitLifecycle.qnt) | [`ReplaySplitLifecycle.lean`](ReplaySplitLifecycle.lean) | Unified split lifecycle: scout/wide α loads, `cutoff_epoch` hit, 32-bit `AbortState` generation wrap (`u32::MAX -> 0`), and slot reuse; rejects `backwardAlpha`, `reusedEpoch`, `missedWrappedAbort`, and `reuseScoreRead`. |
| `cut-boundary` | [`CutBoundary.qnt`](CutBoundary.qnt) | [`ReplayCutBoundary.lean`](ReplayCutBoundary.lean) | Key-mismatch SeqLock cut closure (`ReadPrefixLitmus`, HB-only vs RF+HB closure) and drained multi-session TT handoff; rejects `hbOnlyCut` and `undrainedHandoff`. |

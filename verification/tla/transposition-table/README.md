# Transposition table SeqLock protocol model (TLA+)

Bounded PlusCal/TLA+ model check of the lock-free SeqLock protocol in [`crates/reversi-core/src/transposition_table.rs`](../../../crates/reversi-core/src/transposition_table.rs).
Each 2-entry cluster (`CLUSTER_SIZE = 2`) stores split board words, packed `TTEntryData` metadata, and a 64-bit SeqLock sequence counter (`even` = idle/published, `odd` = write in progress).

See [`../README.md`](../README.md) for prerequisites, shared runner behavior, CI coverage, and `trace.py`. For machine-checked proofs of packing, replacement, clearing, and sampling, see [`verification/lean/NeuralReversi/TranspositionTable/`](../../lean/NeuralReversi/TranspositionTable/).

## Run

```bash
python3 verification/tla/transposition-table/check.py            # default bounds (NumWriters=2, NumReaders=1, MaxOps=2, MaxReaderOps=1, MaxRetries=4)
python3 verification/tla/transposition-table/check.py --deep     # MaxReaderOps=2
python3 verification/tla/transposition-table/check.py --mutants  # verify every mutant fails its target property
python3 verification/tla/transposition-table/check.py --all      # clean + deep + all mutants
```

## Model overview

- **Cluster layout**: 1 cluster of 2 slots (`Slots = {0, 1}`), each holding `seq` (even/odd), `board \in {0, 1, 2}` (`0` = empty/cleared), and `meta = [bound, depth, score, gen]` (`bound \in {"None", "Exact", "Lower", "Upper"}`), plus a ghost publish history `published[slot][evenSeq] = <<board, meta>>`.
- **Reader (`lookup` / `read_for_lookup`)**: scans slots `0` and `1`; reads `seq1` (rejects odd), reads `board` in a separate atomic step (fast-exiting on mismatch without `seq2`), reads `meta` in a separate step, then reads `seq2`; accepts only when `seq1 = seq2` and `meta.bound # "None"`.
- **Writer (`probe` + `save` / `save_exact` / `try_load_snapshot` / `seqlock_write`)**: selects a slot via replacement scoring (`depth - 8 * relative_age`), runs `save_exact` or `save` (`try_load_snapshot` + `ShouldReplace`), claims odd `seq` via CAS (at most `MaxRetries = 4` attempts), writes `board` and `meta` in separate steps, and publishes `seq := s1 + 2`.
- **Shared clear (`clear_shared`)**: iterates slots `0` and `1`; fast-skips already-empty slots, otherwise claims odd `seq` via CAS (at most `MaxRetries = 4`), writes `board := 0` and `meta := EmptyMeta`, and publishes `seq := s1 + 2`. Leaves a slot untouched if all retries contend.

## Checked properties

### Safety invariants

- `TypeOK`: all slot words, ghost histories, writer ownership sets, retry counters, and snapshots stay within their domains.
- `SingleWriterPerSlot`: at most one writer/clearer owns a slot at any time, and ownership is non-empty iff `seq[slot]` is odd.
- `NoTornRead`: every accepted `lookup` hit and `try_load_snapshot` matches `published[slot][seq]`.
- `BoundNoneNeverHits`: `lookup` never returns an entry with `bound = "None"`, even when querying `board = 0` on an empty or cleared slot.
- `BoundedRetries`: neither `save` nor `clear_shared` exceeds `MaxRetries` (`4`) attempts per slot.

### Liveness

- `EventuallyDone == <>AllOpsFinished`: under weak fairness (`fair process`), all bounded reader, writer, and clearer operations terminate without deadlock or livelock.

## Mutants

| Mutant | Simulated regression | Expected catch |
| --- | --- | --- |
| `skip_seq2_check` | `read_for_lookup` skips checking `seq1 = seq2` | `NoTornRead` |
| `accept_odd_seq` | `read_for_lookup` does not reject odd `seq1` | `NoTornRead` |
| `publish_before_payload` | `seqlock_write` publishes even `seq` before writing `meta` | `NoTornRead` |
| `ignore_bound_none` | `lookup` accepts `Bound::None` when `board = 0` matches an empty/cleared slot | `BoundNoneNeverHits` |
| `clear_without_cas` | `clear_shared` overwrites a slot without claiming odd `seq` via CAS | `SingleWriterPerSlot` |
| `exceed_retry_budget` | Retry loop checks `<= MaxRetries` instead of `< MaxRetries` | `BoundedRetries` |
| `livelock_reject_loop` | Non-`Exact` `save` loops forever on `!ShouldReplace` instead of returning | `Temporal` (`EventuallyDone`) |

## Scope and assumptions

- **Memory ordering**: PlusCal steps are sequentially consistent and do not model acquire/release fence reordering; relaxed payload readback is checked in [`../lean-witnesses/TTReadback.qnt`](../lean-witnesses/TTReadback.qnt).
- **No sequence wraparound**: `seq` is unbounded (`Nat`); in Rust, 64-bit `seq` increments by 2 per write and does not wrap during a single read.
- **Outside this model**: `penalize`, `duplicate`, `usage_rate`, hash-to-cluster indexing (`mul_hi64`), and multi-cluster allocation sizing (covered in Lean under `verification/lean/NeuralReversi/TranspositionTable/`).

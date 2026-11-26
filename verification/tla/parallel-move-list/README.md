# Parallel move dispatch and in-flight claim protocol (TLA+)

Bounded PlusCal/TLA+ model check of two interacting lock-free move-dispatch concurrency protocols in `crates/reversi-core`:

1. **`ConcurrentMoveIterator`** ([`crates/reversi-core/src/move_list/iterator.rs`](../../../crates/reversi-core/src/move_list/iterator.rs)): lock-free `AtomicUsize::try_update` weak-CAS loop distributing split-point moves across helper threads, including stale reads and bounded spurious weak-CAS failures.
2. **`InFlightTable` + `PvsPostpone`** ([`crates/reversi-core/src/search/threading/in_flight.rs`](../../../crates/reversi-core/src/search/threading/in_flight.rs), [`crates/reversi-core/src/search/pvs.rs`](../../../crates/reversi-core/src/search/pvs.rs), [`crates/reversi-core/src/move_list.rs`](../../../crates/reversi-core/src/move_list.rs)): direct-mapped hash-indexed in-flight claim table (`Claim::{Owned, Busy, Untracked}`, `Drop for Claim`) combined with ABDADA-style single postponement (`MoveList::postpone_move` when `!eldest && has_fresh_sibling`).

See [`../README.md`](../README.md) for prerequisites, shared runner behavior, CI coverage, and `trace.py`.

## Run

```bash
python3 verification/tla/parallel-move-list/check.py            # default bounds (N=2, NumIterMoves=3, SpuriousBudget=1)
python3 verification/tla/parallel-move-list/check.py --deep     # N=2, NumIterMoves=4, SpuriousBudget=2
python3 verification/tla/parallel-move-list/check.py --mutants  # verify every mutant fails its target property
python3 verification/tla/parallel-move-list/check.py --all      # clean + deep + all mutants
```

## Model overview

Each `fair process Worker \in 1..N` runs a split-point worker loop (`search_split_point` -> `search`):

- **`IterLoad` / `IterCAS`**: models `ConcurrentMoveIterator::next()` over `1..NumIterMoves` with up to `SpuriousBudget` spurious weak-CAS failures and stale-read retries (`iterIdx # obsIdx`). Once `iterIdx > NumIterMoves`, workers observe `None` without modifying `iterIdx`.
- **`OuterClaim` / `OuterDrop`**: models `search_split_point`'s outer `claim_child` on a 2-slot `InFlightTable` (`SlotOf(h) = ((h - 1) % 2) + 1`, so hashes `{1, 3}` collide in slot 1 and `{2, 4}` collide in slot 2).
- **`InnerClaim` / `InnerPostpone` / `InnerDrop`**: models `search`'s serial move loop over a 3-move `ChildMoves(curMove)` list. A `Claim::Busy` probe on a non-eldest move (`idx > 1`) with a remaining fresh sibling (`idx < freshEnd`) rotates `ml[idx..Len(ml)]` left by 1 and decrements `freshEnd` so each move is postponed at most once. `Claim::Untracked` collisions and eldest/already-postponed `Claim::Busy` probes are searched immediately without modifying the owner's slot.

## Checked properties

### Safety invariants

- `TypeOK`: all shared, ghost, and per-worker variables stay within their expected domains.
- `NoDuplicateIteratorClaim`: every move `m \in 1..NumIterMoves` is claimed by at most one worker (`iterClaims[m] <= 1`).
- `DrainedMeansAllClaimed`: whenever any worker observes the iterator drained (`None`), `iterIdx = NumIterMoves + 1` and every move `1..NumIterMoves` has been claimed.
- `ExclusiveInFlightOwnership`: at most one active `Claim::Owned` guard owns any `InFlightTable` slot at any time.
- `CollisionPreservesOwner`: while a worker holds `Claim::Owned(slot, hash)`, `inFlight[slot] = hash`, and unowned slots are `EMPTY`.
- `NoDroppedMove`: every completed iterator move searched its entire `ChildMoveSet(m)`, and when all workers finish, every iterator move was claimed and completed with all `InFlightTable` slots `EMPTY`.

### Liveness

- `EventuallyAllDone == <>(\A w \in Workers : pc[w] = "Done")`: guarantees lock-free iterator progress (despite stale reads and bounded spurious weak-CAS failures) and absence of circular-wait livelock when workers encounter each other's outer hashes.

## Mutants

| Mutant | Simulated regression | Expected catch |
| --- | --- | --- |
| `non_atomic_iterator_claim` | `IterCAS` writes `iterIdx := obsIdx + 1` without checking `iterIdx = obsIdx` | `NoDuplicateIteratorClaim` |
| `stale_claim_skips_move` | `IterCAS` advances `iterIdx` on a stale-read retry instead of reloading | `DrainedMeansAllClaimed` |
| `duplicate_owned_on_busy` | `InFlightTable::claim` returns `Claim::Owned` instead of `Claim::Busy` on a same-hash match | `ExclusiveInFlightOwnership` |
| `collision_clears_slot` | `Drop for Claim` clears `inFlight[slot]` on `Claim::Untracked` hash collisions | `CollisionPreservesOwner` |
| `drop_postponed_move` | Serial move loop stops at `idx > freshEnd`, dropping postponed moves at the tail | `NoDroppedMove` |
| `unbounded_postpone_livelock` | Removes the `idx < freshEnd` postponement bound, causing two workers to livelock re-postponing each other's hashes | `Temporal` (`EventuallyAllDone`) |

## Scope and assumptions

- **Memory ordering**: PlusCal steps are sequentially consistent and do not model `Relaxed` weak-memory visibility.
- **Bounds**: models a single split-point iterator (`NumIterMoves` moves) and a 2-slot `InFlightTable` with bounded spurious weak-CAS failures (`SpuriousBudget`).

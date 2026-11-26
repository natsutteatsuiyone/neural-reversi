#!/usr/bin/env python3
"""TLC model-checking suite for ConcurrentMoveIterator and InFlightTable+PvsPostpone."""

from __future__ import annotations

from pathlib import Path
import sys

TLA_DIR = Path(__file__).resolve().parents[1]
if str(TLA_DIR) not in sys.path:
    sys.path.insert(0, str(TLA_DIR))

from tlc_runner import SuiteSpec, run_suite_cli  # noqa: E402

SUITE = SuiteSpec(
    name="parallel-move-list",
    model_dir=Path(__file__).resolve().parent,
    tla_module="ConcurrentMoveList",
    mutants={
        "non_atomic_iterator_claim": "NoDuplicateIteratorClaim",
        "stale_claim_skips_move": "DrainedMeansAllClaimed",
        "duplicate_owned_on_busy": "ExclusiveInFlightOwnership",
        "collision_clears_slot": "CollisionPreservesOwner",
        "drop_postponed_move": "NoDroppedMove",
        "unbounded_postpone_livelock": "Temporal",
    },
    deep_overrides={"N": "2", "NumIterMoves": "4", "SpuriousBudget": "2"},
)


def main() -> int:
    return run_suite_cli(SUITE)


if __name__ == "__main__":
    sys.exit(main())

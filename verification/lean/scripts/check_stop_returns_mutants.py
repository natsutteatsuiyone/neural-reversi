"""Require an actual changed PVS return, cutoff poll or destructor to invalidate the projection."""

import argparse
from pathlib import Path

from cache_owner_mir import PASS, WORKER
from extract_stop_returns import render
from mir_mutants import assert_mutants_rejected


PVS = "search-pvs-search"
DROP = "search-threading-in_flight-{impl#1}-drop"
LEAF = "search-strategy-{impl#1}-shallow_search"
DEEP = "search-endgame-null_window_search"
MOVE = "search-endgame-search_move_nws_ec"
CASES = [
    (PVS, "_0 = const score::Score::ZERO;", "_0 = const score::Score::MAX;"),
    (PVS, "switchInt(move _533) -> [0: bb310, otherwise: bb309]", "switchInt(move _533) -> [0: bb309, otherwise: bb310]"),
    (PVS, "_0 = copy _121;", "_0 = copy _4;"),
    (PVS, "_121 = <Score as Neg>::neg(move _122)", "_121 = <Score as Neg>::neg(move _4)"),
    (PVS, "drop(_388) -> [return: bb437", "_0 = TranspositionTable::store() -> [return: bb437"),
    (PVS, "drop(_546) -> [return: bb434", "_0 = SearchContext::<'_>::update_root_move() -> [return: bb434"),
    (PVS, "_0 = Board::solve(move _136, move _137)", "_0 = Board::solve(move _136, move _3)"),
    (DROP, "_5 = &(*(*_3));", "_5 = &(*_1);"),
    (WORKER + "is_search_aborted", "0: bb4, otherwise: bb3", "0: bb3, otherwise: bb4"),
    (LEAF, "_0 = copy _121;", "_0 = const score::Score::ZERO;"),
    (LEAF, "_129 = copy _35;", "_129 = copy _128;"),
    (LEAF, "_130 = SubWithOverflow(copy _128, copy _129);", "_130 = SubWithOverflow(copy _129, copy _128);"),
    (LEAF, "_126 = Ge(move _127, const search::strategy::BOUNDARY_STORE_MIN_SUBTREE_NODES);",
     "_126 = Gt(move _127, const search::strategy::BOUNDARY_STORE_MIN_SUBTREE_NODES);"),
    (DEEP, "0: bb49, otherwise: bb48", "0: bb48, otherwise: bb49"),
    (MOVE, "0: bb3, otherwise: bb2", "0: bb2, otherwise: bb3"),
    ("score-{impl#5}-ZERO", "_0 = Score(const 0_i32);", "_0 = Score(const 1_i32);"),
]


def check(dump):
    original = render(dump)
    return assert_mutants_rejected(
        dump,
        [(f"reversi_core.{stem}{PASS}", old, replacement, stem) for stem, old, replacement in CASES],
        render,
        baseline=original,
        expected_errors=(ValueError, AssertionError),
        unique=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    args = parser.parse_args()
    print(f"Detected {check(args.mir)} PVS cancellation mutations")

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
    (PVS, "switchInt(move _530) -> [0: bb308, otherwise: bb307]", "switchInt(move _530) -> [0: bb307, otherwise: bb308]"),
    (PVS, "_0 = copy _120;", "_0 = copy _4;"),
    (PVS, "_120 = <Score as Neg>::neg(move _121)", "_120 = <Score as Neg>::neg(move _4)"),
    (PVS, "drop(_386) -> [return: bb435", "_0 = TranspositionTable::store() -> [return: bb435"),
    (PVS, "drop(_543) -> [return: bb432", "_0 = SearchContext::<'_>::update_root_move() -> [return: bb432"),
    (PVS, "_0 = Board::solve(move _135, move _136)", "_0 = Board::solve(move _135, move _3)"),
    (DROP, "_5 = &(*(*_3));", "_5 = &(*_1);"),
    (WORKER + "is_search_aborted", "0: bb4, otherwise: bb3", "0: bb3, otherwise: bb4"),
    (LEAF, "_0 = copy _134;", "_0 = const score::Score::ZERO;"),
    (LEAF, "_142 = copy _35;", "_142 = copy _141;"),
    (LEAF, "_143 = SubWithOverflow(copy _141, copy _142);", "_143 = SubWithOverflow(copy _142, copy _141);"),
    (LEAF, "_139 = Ge(move _140, const search::strategy::BOUNDARY_STORE_MIN_SUBTREE_NODES);",
     "_139 = Gt(move _140, const search::strategy::BOUNDARY_STORE_MIN_SUBTREE_NODES);"),
    (DEEP, "0: bb57, otherwise: bb56", "0: bb56, otherwise: bb57"),
    (MOVE, "0: bb3, otherwise: bb2", "0: bb2, otherwise: bb3"),
    ("score-{impl#5}-ZERO", "_0 = Score(const 0_i32);", "_0 = Score(const 1_i32);"),
]


def check(dump, *, baseline=None):
    original = render(dump) if baseline is None else baseline
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

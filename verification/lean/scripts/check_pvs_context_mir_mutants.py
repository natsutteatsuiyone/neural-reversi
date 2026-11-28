"""Check the context projection rejects wrong recursion operands, splices and publication fields."""

from pathlib import Path

from extract_pvs_context_mir import CTX, verify
from mir_mutants import check_mir_regression, run_mir_regression_cli


CASES = [
    (CTX + "new", "side_to_move: move _12", "side_to_move: move _22"),
    (CTX + "new", "n_empties: move _14", "n_empties: move _20"),
    (CTX + "update", "SubWithOverflow(copy ((*_1).10: u32)", "AddWithOverflow(copy ((*_1).10: u32)"),
    (CTX + "undo", "AddWithOverflow(copy ((*_1).10: u32)", "SubWithOverflow(copy ((*_1).10: u32)"),
    (CTX + "update_pass", "::switch_players(", "::increment_nodes("),
    (CTX + "ply", "_3 = copy ((*_1).10: u32);", "_3 = copy ((*_1).2: u64);"),
    (CTX + "from_split_task", "_3 = &((*_1).0: board::Board);", "_3 = &((*_1).4: board::Board);"),
    (CTX + "from_split_task", "_11 = &((*_1).5:", "_11 = &((*_1).6:"),
    ("search-threading-split_point-{impl#3}-new", "_25 = copy ((*_1).5: u32)", "_25 = copy ((*_1).6: u32)"),
    ("search-threading-split_point-{impl#3}-new", "PatternFeatures::o_feature(", "PatternFeatures::p_feature("),
    ("search-endgame-aspiration_search", "_13 = copy _7;", "_13 = const 0_u32;"),
    ("search-endgame-search_root", "_10 = &_4;", "_10 = &_1;"),
    ("search-pvs-search", "_395 = copy _365;", "_395 = copy _366;"),
    ("search-pvs-search", "_123 = &(*_124);", "_123 = &(*_2);"),
    ("search-pvs-search", "switchInt(move _248) -> [0: bb145, otherwise: bb157]", "switchInt(move _248) -> [0: bb157, otherwise: bb145]"),
    ("search-pvs-search_split_point", "_28 = copy ((*_23).2:", "_28 = copy ((*_23).1:"),
    ("search-pvs-search_split_point", "_56 = &(*_57);", "_56 = &(*_2);"),
    ("search-midgame-evaluate_depth2", "_74 = &_62;", "_74 = &_31;"),
    ("move_list-ordering-{impl#0}-evaluate_moves_by_search", "_98 = copy ((*_81).2:", "_98 = copy ((*_81).1:"),
]


def _cases():
    return [(f"reversi_core.{name}.SimplifyCfg-initial.after.mir", old, new, name) for name, old, new in CASES]


def check(dump, repo, *, baseline=None):
    return check_mir_regression(
        dump, repo, verify, _cases(), expected_errors=(ValueError, KeyError, IndexError, TypeError), baseline=baseline
    )


if __name__ == "__main__":
    run_mir_regression_cli(
        __doc__,
        check,
        "rejected {count} context MIR mutations",
        positional_dump=True,
        default_repo=Path(__file__).resolve().parents[3],
    )

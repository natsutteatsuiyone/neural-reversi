"""Check the context projection rejects wrong recursion operands, splices and publication fields."""

from pathlib import Path

from extract_pvs_context_mir import CTX, EL, verify
from mir_mutants import check_mir_regression, run_mir_regression_cli


CASES = [
    (CTX + "new", "side_to_move: move _10", "side_to_move: move _20"),
    (CTX + "update", "EmptyList::remove(", "EmptyList::restore("),
    (CTX + "undo", "EmptyList::restore(", "EmptyList::remove("),
    (CTX + "update_pass", "::switch_players(", "::increment_nodes("),
    (CTX + "undo_endgame", "EmptyList::restore(", "EmptyList::remove("),
    (CTX + "from_split_task", "_3 = &((*_1).5:", "_3 = &((*_1).4:"),
    ("search-threading-split_point-{impl#3}-new", "_27 = copy ((*_1).5: u32)", "_27 = copy ((*_1).6: u32)"),
    ("search-threading-split_point-{impl#3}-new", "PatternFeatures::o_feature(", "PatternFeatures::p_feature("),
    ("empty_list-{impl#4}-clone", "parity: move _8", "parity: const 0_u8"),
    (EL + "new", "_3 = const 0_u8", "_3 = const 1_u8"),
    (EL + "new", "(_4[_42].1: u8) = move _41", "(_4[_42].0: u8) = move _41"),
    (EL + "remove", "((*_37).0: u8) = move _36", "((*_37).1: u8) = move _36"),
    (EL + "restore", "((*_46).1: u8) = move _45", "((*_46).0: u8) = move _45"),
    (EL + "restore", "AddWithOverflow(copy ((*_1).1: u32)", "SubWithOverflow(copy ((*_1).1: u32)"),
    ("empty_list-PRESORTED", "_9 = Square::A6", "_9 = Square::A7"),
    ("search-endgame-aspiration_search", "_14 = copy _7;", "_14 = const 0_u32;"),
    ("search-endgame-search_root", "_10 = &_4;", "_10 = &_1;"),
    ("search-pvs-search", "_417 = copy _367;", "_417 = copy _368;"),
    ("search-pvs-search", "_124 = &(*_125);", "_124 = &(*_2);"),
    ("search-pvs-search", "_126 = copy _3;", "_126 = const 0_u32;"),
    ("search-pvs-search", "switchInt(move _250) -> [0: bb147, otherwise: bb159]", "switchInt(move _250) -> [0: bb159, otherwise: bb147]"),
    ("search-pvs-search_split_point", "_95 = copy ((*_23).2:", "_95 = copy ((*_23).1:"),
    ("search-pvs-search_split_point", "_56 = &(*_57);", "_56 = &(*_2);"),
    ("search-midgame-evaluate_depth2", "_75 = &_63;", "_75 = &_31;"),
    ("move_list-ordering-{impl#0}-evaluate_moves_by_search", "_115 = copy ((*_81).2:", "_115 = copy ((*_81).1:"),
]


def _cases():
    return [(f"reversi_core.{name}.SimplifyCfg-initial.after.mir", old, new, name) for name, old, new in CASES]


def check(dump, repo):
    return check_mir_regression(
        dump, repo, verify, _cases(), expected_errors=(ValueError, KeyError, IndexError, TypeError)
    )


if __name__ == "__main__":
    run_mir_regression_cli(
        __doc__,
        check,
        "rejected {count} context MIR mutations",
        positional_dump=True,
        default_repo=Path(__file__).resolve().parents[3],
    )

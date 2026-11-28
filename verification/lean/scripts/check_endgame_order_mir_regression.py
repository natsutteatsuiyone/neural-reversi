"""Mutation checks for compiler-derived move key arithmetic, hint assignment, loops and score origins."""

from extract_endgame_order_mir import IMPL, PRE, constants, endgame_sort_depths, render_weights, verify
from mir_common import PASS
from mir_mutants import check_mir_regression, run_mir_regression_cli


MUTATIONS = {
    (IMPL + "evaluate_moves", PASS): [
        ("move _23, move _24)", "move _23, move _23)"),
        ("otherwise: bb5", "otherwise: bb9"),
        ("_8 = Lt(move _10, move _11);", "_8 = Le(move _10, move _11);"),
        ("copy _12[_13]", "copy _12[0 of 64]"),
        ("0: bb7, otherwise: bb5", "0: bb5, otherwise: bb7"),
    ],
    (IMPL + "evaluate_moves_fast", PASS): [
        ("1: bb12, otherwise: bb1", "1: bb1, otherwise: bb12"),
        ("0: bb25, otherwise: bb24", "0: bb24, otherwise: bb25"),
        ("_36 = const move_list::ordering::TT_MOVE_VALUE;", "_36 = const 0_i32;"),
        ("0: bb20, 1: bb21", "0: bb21, 1: bb20"),
    ],
    (IMPL + "evaluate_moves_by_search", PASS): [
        ("0: bb44, otherwise: bb43", "0: bb43, otherwise: bb44"),
        ("((*_46).1: i32) = Neg(move _58);", "((*_46).1: i32) = copy _58;"),
        ("((*_81).1: i32) = const move_list::ordering::TT_MOVE_VALUE;", "((*_81).1: i32) = const 0_i32;"),
        ("_102 = <Score as Neg>::neg(", "_102 = <Score as Ord>::max("),
        ("AddWithOverflow(copy ((*_81).1: i32), copy _116)", "SubWithOverflow(copy ((*_81).1: i32), copy _116)"),
        ("_124 = copy (_125.1: bitboard::Bitboard);", "_124 = copy (_125.0: bitboard::Bitboard);"),
        ("_140 = SubWithOverflow(", "_140 = AddWithOverflow("),
        ("0: bb64, 1: bb65", "0: bb65, 1: bb64"),
    ],
    (PRE + "fast_move_value", PASS): [
        ("const 36_i32, copy _26", "const 63_i32, copy _26"),
        ("move _12, move _15", "move _12, move _8"),
    ],
    (PRE + "TT_MOVE_VALUE", "built"): [("Shl(const 1_i32, const 30_i32)", "Shl(const 1_i32, const 19_i32)")],
    ("eval-eval_cache-policy-select_way", PASS): [("move _60 as i16 (IntToInt)", "move _60 as i32 (IntToInt)")],
    ("eval-network-with_network_buffers", PASS): [
        ("const eval::network::NETWORK_BUFFERS;", "const eval::network::OTHER_BUFFERS;"),
        ("{ f: move _1 }", "{ f: move _2 }"),
    ],
    ("eval-network_small-{impl#2}-evaluate", PASS): [("move _47, move _48, move _49", "move _47, move _49, move _48")],
    ("search-correction_history-{impl#1}-correct", PASS): [("_0 = <Score as Ord>::clamp", "_0 = <Score as Ord>::max")],
    ("search-midgame-evaluate_depth1", PASS): [("_0 = copy _79;", "_0 = copy _4;")],
    ("search-midgame-evaluate_depth2", PASS): [("_0 = copy _48;", "_0 = copy _4;")],
}


def check(dump, repo, *, baseline=None):
    # Derive the mutants from the current values so that tuning them keeps the check valid.
    weight = constants(dump)[1]["squareWeight"]
    depth = endgame_sort_depths(dump)[0]
    mutations = {
        **MUTATIONS,
        (PRE + "SQUARE_VALUE_WEIGHT", "built"): [(f" = const {weight}_i32;", f" = const {weight + 1}_i32;")],
        (IMPL + "evaluate_moves-ENDGAME_MIN_SORT_DEPTH", "built"): [(f"[const {depth}_u32,", f"[const {depth + 1}_u32,")],
    }
    cases = [
        (f"reversi_core.{stem}.{phase}.after.mir", old, new, stem)
        for (stem, phase), changes in mutations.items()
        for old, new in changes
    ]
    return check_mir_regression(dump, repo, lambda d, r: (verify(d, r), render_weights(d)), cases, baseline=baseline)


def main():
    run_mir_regression_cli(__doc__, check, "{count} endgame ordering mutants detected")


if __name__ == "__main__":
    main()

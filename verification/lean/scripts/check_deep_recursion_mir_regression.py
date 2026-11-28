"""Reject changes to the normal deep NWS, singleton, ETC, recursive poll and publication paths."""

from extract_deep_recursion_mir import constants, render_thresholds, verify
from mir_common import PASS
from mir_mutants import check_mir_regression, run_mir_regression_cli


MUTATIONS = {
    f"search-endgame-search_move_nws_ec.{PASS}": [
        ("_10 = &(*_7);", "_10 = &(*_6);"),
        ("_16 = copy _4;", "_16 = copy _3;"),
        ("_14 = Le(move _15,", "_14 = Ge(move _15,"),
        ("_44 = copy _5;", "_44 = copy _4;"),
        ("_48 = copy _4;", "_48 = copy _3;"),
        ("_53 = copy _5;", "_53 = copy _4;"),
        ("_57 = copy _4;", "_57 = copy _3;"),
        ("_0 = const search::endgame::SPLIT_CUTOFF_SCORE;", "_0 = copy _5;"),
    ],
    f"search-endgame-null_window_search.{PASS}": [
        ("add(move _55, const 1_i32)", "add(move _55, const 2_i32)"),
        ("_69 = copy _52;", "_69 = copy _3;"),
        ("_77 = copy _54;", "_77 = copy _3;"),
        ("_81 = copy _7;", "_81 = copy _62;"),
        ("_91 = copy _3;", "_91 = copy _54;"),
        ("_98 = copy _52;", "_98 = copy _3;"),
        ("_131 = copy _7;", "_131 = copy _6;"),
        ("_139 = copy _54;", "_139 = copy _3;"),
        ("_151 = copy _133;", "_151 = copy _3;"),
        ("Ge(move _161,", "Le(move _161,"),
        ("_184 = copy _54;", "_184 = copy _3;"),
        ("<DiscScore as PartialOrd>::le(", "<DiscScore as PartialOrd>::ge("),
        ("evaluate_moves(move _196, move _197)", "evaluate_moves(move _197, move _196)"),
        ("_211 = copy _7;", "_211 = copy _6;"),
        ("_219 = copy _54;", "_219 = copy _3;"),
        ("_230 = copy _213;", "_230 = copy _3;"),
        ("_221 = &(*_5);", "_221 = &(*_4);"),
        ("_0 = copy _194;", "_0 = copy _3;"),
        ("_240 = copy _194;", "_240 = copy _3;"),
    ],
    "search-endgame-SPLIT_CUTOFF_SCORE.built": [("0_i32", "1_i32")],
    "score-DISC_SCORE_MAX.built": [("64_i32", "63_i32")],
}


def check(dump, repo, *, baseline=None):
    # Derive the threshold mutants from the current values so that tuning them keeps the check valid.
    limits = constants(dump)
    mutations = {**MUTATIONS, **{
        f"search-endgame-{name}.built": [(f"{limits[name]}_u32", f"{limits[name] + 1}_u32")]
        for name in ("DEPTH_TO_SHALLOW_SEARCH", "MIN_EC_ETC_EMPTIES")
    }}
    cases = [
        (f"reversi_core.{stem}.after.mir", old, new, stem)
        for stem, checks in mutations.items()
        for old, new in checks
    ]
    return check_mir_regression(dump, repo, lambda d, r: (verify(d, r), render_thresholds(d)), cases, baseline=baseline)


def main():
    run_mir_regression_cli(__doc__, check, "{count} deep recursion mutants detected")


if __name__ == "__main__":
    main()

"""Reject changes to the normal deep NWS, singleton, ETC, recursive poll and publication paths."""

from extract_deep_recursion_mir import constants, render_thresholds, verify
from mir_common import PASS
from mir_mutants import check_mir_regression, run_mir_regression_cli


MUTATIONS = {
    f"search-endgame-search_move_nws_ec.{PASS}": [
        ("_10 = &(*_7);", "_10 = &(*_6);"),
        ("_14 = copy _4;", "_14 = copy _3;"),
        ("Le(move _17,", "Ge(move _17,"),
        ("_23 = copy _5;", "_23 = copy _4;"),
        ("_33 = &(*_7);", "_33 = &(*_6);"),
        ("_37 = copy _4;", "_37 = copy _5;"),
        ("_0 = copy _15;", "_0 = copy _5;"),
    ],
    f"search-endgame-null_window_search.{PASS}": [
        ("_46 = copy _29;", "_46 = copy _31;"), ("_54 = copy _31;", "_54 = copy _3;"),
        ("_67 = copy _3;", "_67 = copy _31;"), ("_75 = copy _3;", "_75 = copy _31;"),
        ("_0 = copy _77;", "_0 = copy _3;"), ("_110 = copy _85;", "_110 = copy _87;"),
        ("_123 = copy _104;", "_123 = copy _3;"), ("_129 = copy _6;", "_129 = copy _39;"),
        ("Ge(move _133,", "Le(move _133,"), ("_153 = &(*_154);", "_153 = &(*_2);"),
        ("_156 = copy _31;", "_156 = copy _3;"), ("<DiscScore as PartialOrd>::le(", "<DiscScore as PartialOrd>::ge("),
        ("_165 = copy _158;", "_165 = copy _3;"), ("evaluate_moves(move _168, move _169)", "evaluate_moves(move _169, move _168)"),
        ("_188 = copy _31;", "_188 = copy _3;"), ("_196 = copy _182;", "_196 = copy _31;"),
        ("_199 = copy _182;", "_199 = copy _31;"), ("_202 = &(*_5);", "_202 = &(*_4);"),
        ("_209 = copy _166;", "_209 = copy _3;"), ("_0 = copy _166;", "_0 = copy _3;"),
    ],
    "search-endgame-SPLIT_CUTOFF_SCORE.built": [("0_i32", "1_i32")],
    "score-DISC_SCORE_MAX.built": [("64_i32", "63_i32")],
}


def check(dump, repo):
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
    return check_mir_regression(dump, repo, lambda d, r: (verify(d, r), render_thresholds(d)), cases)


def main():
    run_mir_regression_cli(__doc__, check, "{count} deep recursion mutants detected")


if __name__ == "__main__":
    main()

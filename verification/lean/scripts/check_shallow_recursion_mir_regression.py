"""Detect changes to the three source shallow-search routines in the pinned compiler output."""

from extract_shallow_recursion_mir import verify
from mir_common import PASS
from mir_mutants import check_mir_regression, run_mir_regression_cli


MUTATIONS = {
    "shallow_search": [
        ("_53 = copy _30;", "_53 = copy _3;"), ("_65 = copy _3;", "_65 = copy _30;"),
        ("move _94, move _95, move _96", "move _94, move _96, move _95"),
        ("const 18374403900871474942_u64", "const 9187201950435737471_u64"),
        ("_165 = LegalMoves::corners(", "_165 = LegalMoves::non_corners("),
        ("_212 = &mut _97;", "_212 = &mut _3;"), ("_222 = copy _97;", "_222 = copy _3;"),
    ],
    "shallow_search_moves": [
        ("_28 = <DiscScore as PartialOrd>::ge(", "_28 = <DiscScore as PartialOrd>::lt("),
        ("_37 = copy _5;", "_37 = copy _21;"), ("(*_6) = move _40;", "(*_6) = move _21;"),
    ],
    "shallow_search_move": [
        ("Eq(move _16, const 4_u32)", "Eq(move _16, const 5_u32)"),
        ("_27 = copy _12;", "_27 = copy _4;"),
        ("move _52, move _53, move _54, move _55", "move _53, move _52, move _54, move _55"),
        ("_61 = copy _12;", "_61 = copy _4;"), ("_62 = copy _47;", "_62 = copy _12;"),
        ("undo_endgame(move _74, move _75)", "update_endgame(move _74, move _75)"),
    ],
}


def check(dump, repo):
    cases = [
        (f"reversi_core.search-endgame-{method}.{PASS}.after.mir", old, new, method)
        for method, checks in MUTATIONS.items()
        for old, new in checks
    ]
    return check_mir_regression(dump, repo, verify, cases)


def main():
    run_mir_regression_cli(__doc__, check, "{count} shallow recursion mutants detected")


if __name__ == "__main__":
    main()

"""Mutation checks for the fixed-leaf MIR verifier on a live compiler dump."""

from extract_fixed_leaf_mir import constant, verify
from mir_mutants import check_mir_regression, run_mir_regression_cli


BODY = "reversi_core.search-strategy-{impl#1}-shallow_search.SimplifyCfg-initial.after.mir"
CHANGES = [
    ("solver argument swap", "move _83, move _84, move _85)", "move _83, move _85, move _84)"),
    ("wrong next predecessor", "_93 = copy _88;", "_93 = copy _86;"),
    ("wrong quartet element", "_102 = copy (_104.2:", "_102 = copy (_104.1:"),
    ("wrong threshold", "_74 = copy _7;", "_74 = copy _4;"),
    ("wrong final return", "_0 = copy _121;", "_0 = copy _4;"),
    ("bypassed terminal call", "_42 = Board::final_disc_diff", "_42 = Board::disc_score"),
    ("unexpected effect", "_68 = &(*_2);", "(*_1) = move _2;\n        _68 = &(*_2);"),
    ("wrong dispatch arm", "3: bb39, 4: bb40", "3: bb38, 4: bb40"),
    ("reachable table path", "0: bb18, otherwise: bb2", "0: bb2, otherwise: bb18"),
]


def check(dump, repo):
    # Both lowered limits route some of the 0-4 empty arms away from the fixed solvers.
    nws, shallow = constant(dump, "DEPTH_TO_NWS"), constant(dump, "DEPTH_TO_SHALLOW_SEARCH")
    cases = [(BODY, old, new, label) for label, old, new in CHANGES] + [
        ("reversi_core.search-endgame-DEPTH_TO_NWS.built.after.mir", f"{nws}_u32", "3_u32", "TT boundary changed"),
        ("reversi_core.search-endgame-DEPTH_TO_SHALLOW_SEARCH.built.after.mir", f"{shallow}_u32", "2_u32",
         "shallow depth changed"),
    ]
    return check_mir_regression(
        dump, repo, verify, cases, expected_errors=(ValueError,), compare_baseline=False
    )


def main():
    run_mir_regression_cli(__doc__, check, "{count} fixed-leaf MIR mutations rejected")


if __name__ == "__main__":
    main()

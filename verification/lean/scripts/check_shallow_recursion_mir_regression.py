"""Detect changes to the three source shallow-search routines in the pinned compiler output."""

from extract_shallow_recursion_mir import verify
from mir_common import PASS
from mir_mutants import check_mir_regression, run_mir_regression_cli


MUTATIONS = {
    "shallow_search": [
        ("_97 = copy _75;", "_97 = copy _3;"),
        ("_110 = copy _3;", "_110 = copy _75;"),
        ("move _93, move _94, move _96", "move _93, move _96, move _94"),
        ("const 18374403900871474942_u64", "const 9187201950435737471_u64"),
        ("_208 = LegalMoves::corners(", "_208 = LegalMoves::non_corners("),
        ("_192 = &mut _148;", "_192 = &mut _3;"),
        ("_265 = copy _148;", "_265 = copy _3;"),
        ("_199 = copy (_125.1: bitboard::Bitboard);", "_199 = copy _123;"),
        ("_128 = copy _123;", "_128 = copy _149;"),
        ("stability_cutoff(move _116, const EMPTIES, move _117)", "stability_cutoff(move _116, const 4_u32, move _117)"),
        ("_100 = copy _6;", "_100 = copy _37;"),
    ],
    "shallow_search_moves": [
        ("_29 = <DiscScore as PartialOrd>::ge(", "_29 = <DiscScore as PartialOrd>::lt("),
        ("_38 = copy _8;", "_38 = copy _21;"),
        ("(*_6) = move _41;", "(*_6) = move _22;"),
        ("_26 = copy _5;", "_26 = copy _4;"),
    ],
    "shallow_search_move": [
        ("_27 = copy _9;", "_27 = copy (_4.2: score::DiscScore);"),
        ("_51 = sort_last4(move _52, move _53)", "_51 = sort_last4(move _53, move _52)"),
        ("_44 = copy (_4.1: bitboard::Bitboard);", "_44 = copy (_4.0: search::endgame::empties::Empties);"),
        ("_12 = copy (_4.0: search::endgame::empties::Empties);", "_12 = copy (_4.1: bitboard::Bitboard);"),
        ("_79 = copy _11;", "_79 = copy (_4.0: search::endgame::empties::Empties);"),
        ("move _59, move _60, move _61, move _62", "move _60, move _59, move _61, move _62"),
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

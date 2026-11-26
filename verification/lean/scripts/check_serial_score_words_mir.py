"""Check the overflow-checked Score operations used at a recursive PVS boundary."""

import argparse
from pathlib import Path

from mir_common import checked, blocks
from extract_endgame_order_mir import read
from extract_header import ordinary


# All normal blocks, including the conditions and destinations of overflow assertions.
# The release-build implementations are separately checked by check_score_operators.
OPERATIONS = {
    "score-{impl#8}-neg": (
        "(_1: Score) -> Score {",
        [
            ["_3 = copy (_1.0: i32);", "_4 = Eq(copy _3, const i32::MIN);",
             'assert(!move _4, "attempt to negate `{}`, which would overflow", copy _3) -> [success: bb1, unwind: bb2];'],
            ["_2 = Neg(move _3);", "_0 = Score(move _2);", "return;"],
        ],
    ),
    "score-{impl#11}-add": (
        "(_1: Score, _2: i32) -> Score {",
        [
            ["_4 = copy (_1.0: i32);", "_5 = copy _2;", "_6 = AddWithOverflow(copy _4, copy _5);",
             'assert(!move (_6.1: bool), "attempt to compute `{} + {}`, which would overflow", move _4, move _5) '
             '-> [success: bb1, unwind: bb2];'],
            ["_3 = move (_6.0: i32);", "_0 = Score(move _3);", "return;"],
        ],
    ),
    "score-{impl#34}-partial_cmp": (
        "(_1: &Score, _2: &Score) -> Option<std::cmp::Ordering> {",
        [
            ["_4 = &(*_1);", "_5 = &(*_2);", "_3 = <Score as Ord>::cmp(move _4, move _5) -> [return: bb1, unwind: bb2];"],
            ["_0 = Option::<std::cmp::Ordering>::Some(move _3);", "return;"],
        ],
    ),
    "score-{impl#35}-cmp": (
        "(_1: &Score, _2: &Score) -> std::cmp::Ordering {",
        [
            ["_4 = &((*_1).0: i32);", "_3 = &(*_4);", "_6 = &((*_2).0: i32);", "_5 = &(*_6);",
             "_0 = <i32 as Ord>::cmp(move _3, move _5) -> [return: bb1, unwind: bb2];"],
            ["return;"],
        ],
    ),
}


def operation(stem, source):
    signature, expected = OPERATIONS[stem]
    checked(any(line.startswith("fn score::<impl at ") and line.endswith(signature)
                for line in source.splitlines()), f"{stem}: wrong Score operands")
    seen = {pc: ordinary(instructions) for pc, instructions in blocks(source).items()}
    checked(seen == dict(enumerate(expected)), f"{stem}: changed normal Score operation or overflow assertion")


def check(dump):
    sources = {stem: read(dump, stem) for stem in OPERATIONS}
    for stem, source in sources.items():
        operation(stem, source)
    mutants = [
        ("score-{impl#8}-neg", "const i32::MIN", "const 0_i32"),
        ("score-{impl#8}-neg", "assert(!move _4", "assert(move _4"),
        ("score-{impl#11}-add", "AddWithOverflow(", "SubWithOverflow("),
        ("score-{impl#11}-add", "_3 = move (_6.0: i32);", "_3 = copy _2;"),
        ("score-{impl#34}-partial_cmp", "cmp(move _4, move _5)", "cmp(move _5, move _4)"),
        ("score-{impl#35}-cmp", "<i32 as Ord>::cmp", "<u32 as Ord>::cmp"),
    ]
    for stem, before, after in mutants:
        source = sources[stem]
        checked(source.count(before) == 1, f"{stem}: missing Score mutation anchor {before}")
        try:
            operation(stem, source.replace(before, after))
        except ValueError:
            continue
        raise ValueError(f"{stem}: accepted Score mutation {after}")
    return len(mutants)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    args = parser.parse_args()
    print(f"checked signed PVS score operators and rejected {check(args.mir)} checked-MIR mutations")

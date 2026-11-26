"""Check that fixed endgame solvers can mutate only the node counter through their context."""

import argparse
from pathlib import Path
import re

from extract_endgame_order_mir import read
from extract_header import ordinary
from mir_common import checked, blocks, CALL

PREFIX = "reversi_core.search-endgame-solve-"
SUFFIX = ".SimplifyCfg-initial.after.mir"
INCREMENT = "SearchContext::<'_>::increment_nodes"
REB = re.compile(r"(_\d+) = &mut \(\*(_\d+)\);")
LOCAL = re.compile(r"\b_\d+\b")


def checked_counter(source):
    cfg = {pc: ordinary(ops) for pc, ops in blocks(source).items()}
    checked(cfg == {
        0: ["_2 = AddWithOverflow(copy ((*_1).2: u64), const 1_u64);",
            'assert(!move (_2.1: bool), "attempt to compute `{} + {}`, which would overflow", '
            'copy ((*_1).2: u64), const 1_u64) -> [success: bb1, unwind: bb2];'],
        1: ["((*_1).2: u64) = move (_2.0: u64);", "_0 = const ();", "return;"]
    }, "increment_nodes has an unverified context effect")


def callees(source):
    ops = [op for block in blocks(source).values() for op in ordinary(block)]
    borrows = [m.groups() for op in ops if (m := REB.fullmatch(op))]
    refs = {"_1"}
    while new := {dest for dest, src in borrows if src in refs} - refs:
        refs.update(new)

    forwarded = set()
    for op in ops:
        uses = [reg for reg in LOCAL.findall(op) if reg in refs]
        if not uses:
            continue
        if (borrow := REB.fullmatch(op)) and borrow[2] in refs:
            continue
        call = re.fullmatch(CALL, op)
        checked(call and len(uses) == 1 and call[3].split(", ")[0] == "move " + uses[0],
                f"unverified use of fixed solver context: {op}")
        if call[2] != INCREMENT:
            forwarded.add(call[2])
    return forwarded


def check(dump):
    sources = {}
    for path in dump.glob(PREFIX + "solve*" + SUFFIX):
        name = path.name.removeprefix(PREFIX).removesuffix(SUFFIX)
        if not re.fullmatch(r"solve[1-4](?:_\w+)?", name):
            continue
        source = path.read_text()
        signature = re.search(r"^fn ((?:solve::)?" + name + r")\(_1: &mut SearchContext<'_>,", source, re.M)
        if signature:
            sources[signature[1]] = source
    counter = read(dump, "search-context-{impl#0}-increment_nodes")
    checked_counter(counter)

    def traverse(bodies):
        visited = set()
        pending = {"solve::solve1", "solve2", "solve3", "solve4"}
        while pending:
            name = pending.pop()
            checked(name in bodies, f"missing verified body for context recipient {name}")
            visited.add(name)
            pending.update(callees(bodies[name]) - visited)

    traverse(sources)
    original = sources["solve::solve1"]
    borrow = next(m for line in original.splitlines() if (m := REB.fullmatch(line.strip())) and m[2] == "_1")
    counter_write = "((*_1).2: u64) = move (_2.0: u64);"
    bad_store = "(((*{ref}).10: empty_list::EmptyList).1: u32) = const 0_u32;"
    mutations = [
        (original, INCREMENT, "SearchContext::<'_>::clear_empties", False),
        (original, borrow[0], bad_store.format(ref="_1") + "\n        " + borrow[0], False),
        (original, borrow[0], borrow[0] + "\n        " + bad_store.format(ref=borrow[1]), False),
        (original, borrow[0], borrow[0] + f"\n        _200 = &raw mut (*{borrow[1]});", False),
        (counter, counter_write, counter_write + "\n        " + bad_store.format(ref="_1"), True),
    ]
    for source, old, new, counter_only in mutations:
        checked(source.count(old) == 1, "missing fixed solver effect mutation anchor")
        mutated = source.replace(old, new)
        try:
            if counter_only:
                checked_counter(mutated)
            else:
                traverse({**sources, "solve::solve1": mutated})
        except ValueError:
            continue
        raise ValueError(f"accepted fixed solver context effect mutation {new}")
    return len(mutations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    args = parser.parse_args()
    print(f"checked transitive fixed solver context effects and rejected {check(args.mir)} MIR mutations")

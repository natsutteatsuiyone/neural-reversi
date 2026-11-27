"""Check polled solve_root paths and reject mutations to their actual MIR edges."""

import argparse
import itertools
from pathlib import Path
import re

import extract_outer
from mir_common import CALL, GOTO, blocks, checked


ABORT = "threading::thread::Thread::is_search_aborted"
COMPARE = "<Selectivity as PartialOrd>::ge"


def check_source(source):
    paths = extract_outer.polled_paths(source)
    expected = {("entry", ()): ("line", ("beginPhase", "prepareEndgame"))}
    for has_line, publish in itertools.product((False, True), repeat=2):
        effects = ["setPvIndex"] if has_line else (["iterationCompleted"] if publish else []) + ["returnResult"]
        expected[("line", (has_line, publish))] = ("level" if has_line else "returned", tuple(effects))
    for item, above, publish, abort, clock in itertools.product((False, True), repeat=5):
        key = ("level", (item, above, publish, abort, clock))
        if not item or above:
            expected[key] = ("afterLine", ())
        else:
            effects = ["setLevel", "query", "nextWindow", "pollAbort"]
            if abort:
                expected[key] = ("afterLine", tuple(effects + ["recordAbort"]))
            else:
                effects += ["sortRows"] + (["lineCompleted"] if publish else []) + ["clockCheck"]
                expected[key] = ("afterLine" if clock else "level", tuple(effects))
    for abort, timeout in itertools.product((False, True), repeat=2):
        effects = ["pollAbort"] + ([] if abort else ["timeCheck"])
        if abort or timeout:
            effects += ["returnResult"]
        expected[("afterLine", (abort, timeout))] = ("returned" if abort or timeout else "line", tuple(effects))
    checked(paths.keys() == expected.keys(), "polled outer case coverage changed")
    for case, wanted in expected.items():
        checked(paths[case] == wanted, f"polled outer {case}: expected {wanted}, saw {paths[case]}")
    return paths


def read(dump):
    return (dump / (extract_outer.ROOT + "solve_root" + extract_outer.PASS)).read_text(encoding="utf-8")


def swap_blocks(source, a, b):
    checked(a != b and a != 0 and b != 0, "cannot swap entry or identical blocks")
    mapping = {str(a): f"bb{b}", str(b): f"bb{a}"}
    return re.sub(rf"\bbb({a}|{b})\b", lambda m: mapping[m[1]], source)


def check_abort_classification(source, expected_paths):
    cfg = blocks(source)
    names, _, _, _ = extract_outer.locations(cfg, source)
    before, after = names["before_abort"], names["after_abort"]
    swapped = swap_blocks(source, before, after)
    swapped_cfg = blocks(swapped)
    swapped_names, _, _, _ = extract_outer.locations(swapped_cfg, swapped)
    checked(swapped_names["before_abort"] == after and swapped_names["after_abort"] == before,
            "abort site classification depends on basic block numbering")
    checked(check_source(swapped) == expected_paths,
            "polled outer paths changed after swapping abort basic block numbers")


def check(dump):
    source = read(dump)
    paths = check_source(source)
    check_abort_classification(source, paths)
    return len(paths)


def branch_for_call(cfg, site):
    call = re.fullmatch(CALL, cfg[site][-1])
    checked(call, f"bb{site} is not a call")
    pattern = (r"switchInt\(move " + re.escape(call[1])
               + r"\) -> \[0: bb(\d+), otherwise: bb(\d+)\];")
    choices = [(pc, match) for pc, instructions in cfg.items()
               if (match := re.fullmatch(pattern, instructions[-1]))]
    checked(len(choices) == 1, f"ambiguous branch after bb{site}")
    pc, match = choices[0]
    return pc, int(match[1]), int(match[2])


def first_call(cfg, pc):
    visited = set()
    while pc not in visited:
        visited.add(pc)
        end = cfg[pc][-1]
        if call := re.fullmatch(CALL, end):
            return call[2]
        edge = re.fullmatch(GOTO, end)
        checked(edge, f"bb{pc} has no straight path to the guarded call")
        pc = int(edge[1] or edge[2])
    raise ValueError("cycle before guarded call")


def replace_edge(source, cfg, pc, no, yes):
    old = cfg[pc][-1]
    changed, count = re.subn(r"\[0: bb\d+, otherwise: bb\d+\]", f"[0: bb{no}, otherwise: bb{yes}]", old)
    checked(count == 1 and changed != old, f"bb{pc}: ineffective branch mutant")
    pattern = re.compile(rf"(^    bb{pc}: \{{\n)(.*?)(^    \}})", re.M | re.S)
    match = pattern.search(source)
    checked(match and match[2].count(old) == 1, f"bb{pc}: ambiguous mutation site")
    body = match[2].replace(old, changed)
    return source[:match.start(2)] + body + source[match.end(2):]


def check_mutants(dump):
    source = read(dump)
    check_source(source)
    cfg = blocks(source)
    names, _, _, _ = extract_outer.locations(cfg, source)
    before, before_no, before_yes = branch_for_call(cfg, names["before_abort"])
    after, after_no, after_yes = branch_for_call(cfg, names["after_abort"])
    guarded = {}
    for pc, instructions in cfg.items():
        call = re.fullmatch(CALL, instructions[-1])
        if call and call[2] == COMPARE:
            branch, no, yes = branch_for_call(cfg, pc)
            guarded[first_call(cfg, yes)] = (branch, no, yes)
    checked(set(guarded) == {extract_outer.MARK, extract_outer.CAPTURE}, "unknown reportability guards")
    mark, mark_no, mark_yes = guarded[extract_outer.MARK]
    capture, capture_no, capture_yes = guarded[extract_outer.CAPTURE]

    mutations = [
        ("before-abort-bypass", before, before_no, before_no, True),
        ("before-abort-inverted", before, before_yes, before_no, False),
        ("after-abort-bypass", after, after_no, after_no, True),
        ("after-abort-inverted", after, after_yes, after_no, False),
        ("line-capture-unguarded", mark, mark_yes, mark_yes, False),
        ("normal-final-capture-unguarded", capture, capture_yes, capture_yes, False),
    ]
    for name, pc, no, yes, old_accepts in mutations:
        mutant = replace_edge(source, cfg, pc, no, yes)
        if old_accepts:
            # These regressions specifically demonstrate the former blind spot:
            # forcing both abort reads to false accepts the changed true edge.
            extract_outer.check_ladder_source(mutant)
        try:
            check_source(mutant)
        except ValueError:
            continue
        raise ValueError(f"accepted outer MIR mutation {name}")
    return len(mutations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    args = parser.parse_args()
    print(f"verified {check(args.mir)} polled outer cases and rejected {check_mutants(args.mir)} MIR mutants")

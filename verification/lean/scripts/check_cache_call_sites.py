"""Reject stale cache call-site ordinals even when two sites call the same Rust method."""

from pathlib import Path
import re
import tempfile

from extract_cache_owner import END, routing
from mir_common import BLOCK, checked
import extract_deep_recursion_mir as deep
import extract_shallow_recursion_mir as shallow


def check(dump):
    cases = [
        ("shallow_search", shallow.SHALLOW, shallow.verify_node, [(5, 6), (5, 9), (4, 10)]),
        ("null_window_search", deep.DEEP_SIG, deep.verify_node, [(1, 5), (2, 6), (3, 7), (4, 8)]),
    ]
    total = 0
    for name, signature, verify_node, pairs in cases:
        stem = END + name
        file = f"reversi_core.{stem}.{shallow.PASS}.after.mir"
        source = (dump / file).read_text(encoding="utf-8")
        blocks = {int(m[1]): m for m in re.finditer(BLOCK, source, re.M)}
        calls = routing(dump, stem)
        for early, late in pairs:
            a, c, f = calls[early]
            b, d, g = calls[late]
            checked((c, f) == (d, g), "site mutant must preserve callee and borrow")
            first, second = blocks[a], blocks[b]
            checked(first.end() < second.start(), "unexpected block layout")
            # Keep both call sites and all their control-flow successors. Exchanging
            # their positions in the compiler listing exchanges their assigned ordinals.
            changed = (source[:first.start()] + second[0] + source[first.end():second.start()]
                       + first[0] + source[second.end():])
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)
                (target / file).write_text(changed, encoding="utf-8")
                rows = routing(target, stem)
                checked([row[1:] for row in rows] == [row[1:] for row in calls], "changed receiver graph")
                sites = {pc: i for i, (pc, _, _) in enumerate(rows)}
                try:
                    verify_node(shallow.body(target, name, signature), sites)
                except ValueError as error:
                    if "cache call-site order" not in str(error):
                        raise
                else:
                    raise ValueError(f"accepted swapped {name} sites {early} and {late}")
            total += 1
    return total

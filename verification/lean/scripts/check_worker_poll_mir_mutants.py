"""Check mutations that change a completed worker's iterator/lock/exit chronology."""

from pathlib import Path

import extract_worker_poll_mir as extract
from mir_mutants import assert_mutants_rejected


def check(dump, repo):
    expected = extract.render(dump, repo)
    cases = [
        ("iterator tag", "switchInt(move _22) -> [1: bb7, otherwise: bb98]", "switchInt(move _22) -> [0: bb7, otherwise: bb98]"),
        ("omit the lock", "_96 = SplitPoint::lock(move _97)", "_96 = SplitPoint::state(move _97)"),
        ("continue when stopped", "switchInt(move _101) -> [0: bb52, otherwise: bb48]", "switchInt(move _101) -> [0: bb48, otherwise: bb52]"),
        ("exit after a retry", "StorageDead(_20);\n        goto -> bb4;", "StorageDead(_20);\n        goto -> bb101;"),
        ("repoll after cutoff", "StorageDead(_20);\n        goto -> bb101;", "StorageDead(_20);\n        goto -> bb4;"),
        ("false cutoff selector", "_177 = copy _127;", "_177 = copy _101;"),
        ("empty iterator enters a move", "1: bb7, otherwise: bb98", "1: bb7, otherwise: bb8"),
    ]
    return assert_mutants_rejected(
        dump,
        [(extract.MIR, before, after, name) for name, before, after in cases],
        lambda d: extract.render(d, repo),
        baseline=expected,
    )


if __name__ == "__main__":
    import sys
    print(f"rejected {check(Path(sys.argv[1]), Path.cwd())} worker loop projection mutants")

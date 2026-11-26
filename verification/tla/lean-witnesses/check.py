"""Run all Quint -> TLC -> Lean witness replay drivers."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

WITNESS_SUITES: tuple[tuple[str, str], ...] = (
    ("call-returns", "check_call_returns.py"),
    ("cut-boundary", "check_cut_boundary.py"),
    ("nested-cutoff", "check_nested_cutoff.py"),
    ("owner-active-sync", "check_owner_active_sync.py"),
    ("parent-mask-zero", "check_parent_mask_zero.py"),
    ("score-reuse", "check_score_reuse.py"),
    ("split-lifecycle", "check_split_lifecycle.py"),
    ("tt-readback", "check_tt_readback.py"),
    ("worker-startup", "check_worker_startup.py"),
)

SUITE_MAP = dict(WITNESS_SUITES)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--suite",
        action="append",
        choices=list(SUITE_MAP),
        help="run only the specified witness suite (repeatable; defaults to all 9)",
    )
    ap.add_argument("--quint", default="quint", help="Quint CLI executable")
    ap.add_argument(
        "--apalache-jar",
        type=Path,
        help="use an existing Apalache distribution's JAR",
    )
    args = ap.parse_args(argv)

    selected = list(dict.fromkeys(args.suite)) if args.suite else list(SUITE_MAP)
    extra: list[str] = ["--quint", args.quint]
    if args.apalache_jar is not None:
        extra.extend(["--apalache-jar", str(args.apalache_jar)])

    results: list[tuple[str, bool]] = []
    for name in selected:
        script = HERE / SUITE_MAP[name]
        print(f"\n== [lean-witnesses/{name}] {script.name}", flush=True)
        proc = subprocess.run([sys.executable, str(script), *extra])
        results.append((name, proc.returncode == 0))

    print("\n=== Lean Witnesses Summary ===")
    all_ok = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} {status}")
        all_ok &= passed
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

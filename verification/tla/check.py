"""Top-level runner for all TLA+ / PlusCal concurrency verification suites."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

TLA_DIR = Path(__file__).resolve().parent
if str(TLA_DIR) not in sys.path:
    sys.path.insert(0, str(TLA_DIR))

from tlc_runner import SuiteSpec, run_suite

SUITE_NAMES = (
    "split-point",
    "thread-pool",
    "stop-abort",
    "transposition-table",
    "parallel-move-list",
    "endgame-result",
)


def load_suite(name: str) -> SuiteSpec:
    """Dynamically import SUITE from verification/tla/<name>/check.py."""
    check_path = TLA_DIR / name / "check.py"
    if not check_path.is_file():
        raise FileNotFoundError(f"suite {name!r} is missing {check_path}")

    mod_name = f"_tla_suite_{name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, check_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {check_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    suite = getattr(module, "SUITE", None)
    if not isinstance(suite, SuiteSpec):
        raise TypeError(
            f"{check_path} must export SUITE as a SuiteSpec (got {type(suite).__name__})"
        )
    return suite


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run TLA+ concurrency verification suites")
    ap.add_argument(
        "--suite",
        action="append",
        choices=SUITE_NAMES,
        help="run only the specified suite (repeatable; defaults to all suites)",
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--mutants",
        action="store_true",
        help="run every mutant in each selected suite; each must fail",
    )
    g.add_argument(
        "--deep",
        action="store_true",
        help="run deep overrides for each selected suite (skips suites without deep_overrides)",
    )
    g.add_argument(
        "--all",
        action="store_true",
        help="run clean, deep (when configured), and every mutant for each selected suite",
    )
    ap.add_argument(
        "--workers",
        default="auto",
        help="TLC worker threads (default: auto)",
    )
    args = ap.parse_args(argv)

    selected = list(dict.fromkeys(args.suite)) if args.suite else list(SUITE_NAMES)
    if args.all:
        mode = "all"
    elif args.deep:
        mode = "deep"
    elif args.mutants:
        mode = "mutants"
    else:
        mode = "clean"

    results: list[tuple[str, bool]] = []
    for name in selected:
        try:
            suite = load_suite(name)
        except Exception as exc:
            print(f"== [{name}] FAIL: {exc}", file=sys.stderr)
            results.append((name, False))
            continue
        passed = run_suite(suite, mode=mode, workers=args.workers)
        results.append((name, passed))

    print("\n=== TLA+ Summary ===")
    all_ok = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} {status}")
        all_ok &= passed
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

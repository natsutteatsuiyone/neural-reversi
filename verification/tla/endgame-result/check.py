#!/usr/bin/env python3
"""Bounded endgame split score/stop/owner return protocol checks."""

from __future__ import annotations

from pathlib import Path
import sys

TLA_DIR = Path(__file__).resolve().parents[1]
if str(TLA_DIR) not in sys.path:
    sys.path.insert(0, str(TLA_DIR))

from tlc_runner import SuiteSpec, run_suite_cli  # noqa: E402

SUITE = SuiteSpec(
    name="endgame-result",
    model_dir=Path(__file__).resolve().parent,
    tla_module="EndgameResult",
    mutants={
        "skip_pv_retry": "ReturnedBound",
        "retry_against_new_alpha": "ReturnedBound",
        "read_before_workers_drain": "OwnerReadAfterDrain",
        "publish_after_abort": "NoInterruptedPublication",
        "accept_after_abort": "NoInterruptedAcceptance",
    },
    deep_overrides={"NumMoves": "3"},
    pluscal=False,
)


def main() -> int:
    return run_suite_cli(SUITE)


if __name__ == "__main__":
    sys.exit(main())

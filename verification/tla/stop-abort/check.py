"""TLC model-checking suite for StopToken, AbortState, and timer_loop."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlc_runner import SuiteSpec, run_suite_cli  # noqa: E402

MUTANTS = {
    "wrap_keeps_old_abort_marker": "FreshRunNotSpuriouslyAborted",
    "attach_skips_pre_stopped_check": "StoppedTokenReachesAttached",
    "detach_does_not_remove": "NoStaleAbortLeak",
    "skip_timer_join": "NoStaleAbortLeak",
    "unscoped_timer_abort": "NoStaleAbortLeak",
    "request_abort_for_ignores_gen": "NoStaleAbortLeak",
    "request_abort_for_clobbers_abort": "StoppedTokenReachesAttached",
    "stop_timer_skips_flag": "Deadlock",
}

DEEP_OVERRIDES = {
    "MaxRuns": "9",
    "NumTokens": "2",
}

SUITE = SuiteSpec(
    name="stop-abort",
    model_dir=Path(__file__).resolve().parent,
    tla_module="StopAbort",
    mutants=MUTANTS,
    deep_overrides=DEEP_OVERRIDES,
)


def main(argv: list[str] | None = None) -> int:
    return run_suite_cli(SUITE, argv)


if __name__ == "__main__":
    sys.exit(main())

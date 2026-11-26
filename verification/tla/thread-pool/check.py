"""TLC model-checking suite for ThreadPool lifecycle, Lazy SMP <-> YBWC switching, and Arc shutdown."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tlc_runner import SuiteSpec, run_suite_cli  # noqa: E402

MUTANTS = {
    "skip_wait_ready": "ReadyBeforeSearch",
    "skip_stop_helpers_before_split": "NoHelperSplitOverlap",
    "exit_before_zero_refs": "NoExitWhileRefHeld",
    "drop_pool_before_clear_thinking": "NoUseAfterExit",
    "drop_last_ref_self_join": "NoSelfJoin",
    "exit_skips_notify": "Deadlock",
    "exit_skips_main_msg": "Deadlock",
    "worker_ignores_exit_flag": "Temporal",
}

DEEP_OVERRIDES = {"N": "3", "MaxSearches": "3"}

SUITE = SuiteSpec(
    name="thread-pool",
    model_dir=Path(__file__).resolve().parent,
    tla_module="ThreadPool",
    mutants=MUTANTS,
    deep_overrides=DEEP_OVERRIDES,
)


def main() -> int:
    return run_suite_cli(SUITE)


if __name__ == "__main__":
    sys.exit(main())

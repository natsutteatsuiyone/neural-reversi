from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tlc_runner import SuiteSpec, run_suite_cli  # noqa: E402

MUTANTS = {
    "late_join_no_recheck": "BookedOnlyIntoLiveSplitPoints",
    "leave_keeps_all_helpers_searching": "BookedOnlyIntoLiveSplitPoints",
    "reset_bit_before_merge_counters": "TaskAliveWhileUsed",
    "can_join_ignores_searching": "BookingMatchesThread",
    "leave_keeps_active": "ActiveChainAlive",
    "cutoff_check_only_own_split_point": "CutoffCacheSound",
    "leave_no_cache_reset": "CutoffCacheNoFalseStop",
    "finalize_skips_clear": "CleanTeardown",
    "hold_lock_across_idle_loop": "Deadlock",
    "unwind_skips_helper_bit_reset": "Deadlock",
    "unwind_clears_slot_before_helpers_leave": "BookedOnlyIntoLiveSplitPoints",
    "unwind_keeps_active": "ActiveChainAlive",
    "unwind_can_join_ignores_failed": "Deadlock",
}

DEEP_OVERRIDES = {"N": "2", "Budget": "3"}

SUITE = SuiteSpec(
    name="split-point",
    model_dir=Path(__file__).resolve().parent,
    tla_module="SplitPoint",
    mutants=MUTANTS,
    deep_overrides=DEEP_OVERRIDES,
)


def main() -> int:
    return run_suite_cli(SUITE)


if __name__ == "__main__":
    sys.exit(main())

"""Mutate the slot projection fixtures to check saved, restored, and reset slot identities."""

from extract_slot_frame_mir import render
from extract_worker_frame_mir import SUFFIX, TH
from mir_mutants import check_mir_projection, run_mir_test_cli


CASES = [
    (TH + "initialize_split_point", "(*_104) = move _102;", "(*_104) = move _83;"),
    (TH + "finalize_split_point", "((*_27).11:", "((*_27).10:"),
    (TH + "finalize_split_point", "fetch_sub(move _21, const 1_usize", "fetch_sub(move _21, const 2_usize"),
    (TH + "search_active_split_point", "AtomicBitSet::reset(move _70, move _73)", "AtomicBitSet::reset(move _70, const 0_usize)"),
    (TH + "new-{closure#0}", "_0 = Arc::<SplitPoint>::new", "_0 = Arc::<SplitPoint>::clone"),
    (TH + "abandon_active_split", "AtomicBitSet::reset(move _33, move _36)", "AtomicBitSet::reset(move _33, const 0_usize)"),
    ("reversi_core.search-threading-split_point-{impl#3}-new", "parent_split_point: move _30", "parent_split_point: move _23"),
    (TH + "split", "Thread::finalize_split_point(move _27, move _30)", "Thread::finalize_split_point(move _27, move _27)"),
    (TH + "abandon_active_split", "_22 = Ne(", "_22 = Eq("),
    (TH + "abandon_active_split", "AbortState::request_abort(", "AbortState::is_aborted("),
]


def check(dump, expected):
    return check_mir_projection(
        dump, expected, render, SUFFIX, CASES, file_filter=lambda p: "search-threading-" in p.name
    )


if __name__ == "__main__":
    run_mir_test_cli(__doc__, check, "{count} slot projection mutations detected")

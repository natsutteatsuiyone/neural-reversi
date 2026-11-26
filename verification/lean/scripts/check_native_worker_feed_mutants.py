"""Reject ticket substitution, off-by-one storage, and wrong specialized child call sites."""

from pathlib import Path

from extract_native_worker_feed import ARRAY, ITER, WORKER, verify
from mir_mutants import check_mir_regression, run_mir_regression_cli


CASES = [
    (ITER + "from_offset", "_5 = copy _2;", "_5 = const 0_usize;"),
    ("search-threading-thread-{impl#1}-initialize_split_point", "_15 = copy (_4.7: usize);", "_15 = const 0_usize;"),
    ("search-threading-split_point-{impl#5}-set_move_iter", "(*_6) = move _4;", "(*_6) = move _5;"),
    ("search-threading-split_point-{impl#5}-move_iter", "((*_1).2: std::cell::SyncUnsafeCell", "((*_1).3: std::cell::SyncUnsafeCell"),
    (ITER + "next", "_21 = &(((*_1).0:", "_21 = &(((*_1).1:"),
    (ITER + "next", "_22 = copy _4;", "_22 = copy _2;"),
    (ITER + "next", "_24 = copy _4;", "_24 = const 0_usize;"),
    (ITER + "next", "_4 = copy _17;", "_4 = copy _2;"),
    (ITER + "next", "_10 = std::sync::atomic::Ordering::Relaxed;", "_10 = std::sync::atomic::Ordering::Acquire;"),
    (ITER + "next-{closure#0}", "_3 = Lt(move _4, move _5);", "_3 = Le(move _4, move _5);"),
    (ITER + "next-{closure#0}-{closure#0}", "AddWithOverflow(copy _2, const 1_usize)", "AddWithOverflow(copy _2, const 2_usize)"),
    (ARRAY + "get_unchecked_ref", "_14 = copy _2;", "_14 = const 0_usize;"),
    (WORKER, "_21 = &(*_3);", "_21 = &(*_2);"),
    (WORKER, "_23 = copy (((_20 as Some).0: (&move_list::Move, usize)).0:", "_23 = copy (((_20 as Some).0: (&move_list::Move, usize)).1:"),
    (WORKER, "_54 = search::<NonPV, EndGameStrategy>", "_54 = search::<PV, EndGameStrategy>"),
]


def _cases():
    return [(f"reversi_core.{name}.SimplifyCfg-initial.after.mir", old, new, name) for name, old, new in CASES]


def check(dump, repo):
    return check_mir_regression(
        dump, repo, verify, _cases(), expected_errors=(ValueError, KeyError, IndexError, TypeError)
    )


if __name__ == "__main__":
    run_mir_regression_cli(
        __doc__,
        check,
        "rejected {count} worker ticket and return-site mutations",
        positional_dump=True,
        default_repo=Path(__file__).resolve().parents[3],
    )

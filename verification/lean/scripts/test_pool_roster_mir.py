"""Mutate the production body fixtures to check the cursor and thread identity projection."""

from extract_pool_roster_mir import render
from extract_worker_frame_mir import PL, SUFFIX, TH
from mir_mutants import check_mir_projection, run_mir_test_cli


CASES = [
    (PL + "create_worker_threads", "start: const 1_usize", "start: const 0_usize"),
    (PL + "create_main_thread", "const 0_usize,", "const 1_usize,"),
    (TH + "new", "idx: move _14", "idx: move _23"),
    (PL + "assign_helpers_to_split_point", "book_into(move _42, move _45", "book_into(move _45, move _45"),
    (TH + "try_late_join", "book_into(move _97", "book_into(move _98"),
    (PL + "add_thread", "Vec::<Arc<threading::thread::Thread>>::push", "Vec::<Arc<threading::thread::Thread>>::insert"),
]


def check(dump, expected):
    return check_mir_projection(
        dump, expected, render, SUFFIX, CASES, file_filter=lambda p: "search-threading-" in p.name
    )


if __name__ == "__main__":
    run_mir_test_cli(__doc__, check, "{count} roster projection mutations detected")

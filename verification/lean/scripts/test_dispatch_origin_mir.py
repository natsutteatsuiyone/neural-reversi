"""Check worker dispatch, job publication and the preceding callback's release."""

from extract_dispatch_origin_mir import render
from extract_worker_frame_mir import SUFFIX, TH
from mir_mutants import check_mir_projection, run_mir_test_cli


CASES = [
    (TH + "idle_loop", "run_helper_job(move _45, move _46)", "run_helper_job(move _45, move _45)"),
    (TH + "idle_loop", ">>::take(move _35)", ">>::take(move _40)"),
    (TH + "run_helper_job", "_11 = (move _12,);", "_11 = (move _10,);"),
    (TH + "book_helper_job", ">>::Some(move _20)", ">>::None"),
    (TH + "book_helper_job", "const true, move _31", "const false, move _31"),
    (TH + "run_helper_job-{impl#0}-drop", "const false, move _13", "const true, move _13"),
    (TH + "search_active_split_point", "Thread::try_late_join(move _81)", "Thread::try_late_join(move _999)"),
]


def check(dump, expected):
    return check_mir_projection(
        dump, expected, render, SUFFIX, CASES, file_filter=lambda p: "search-threading-" in p.name
    )


if __name__ == "__main__":
    run_mir_test_cli(__doc__, check, "{count} dispatch source mutations detected")

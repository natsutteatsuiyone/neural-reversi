"""Mutate the actual PVS gate, captured operands, and incoming scheduler calls."""

from extract_guard_receiver_mir import DROP, PVS, SCOUT, render
from extract_worker_frame_mir import SUFFIX, TH
from mir_mutants import check_mir_projection, run_mir_test_cli


CASES = [
    (PVS, "Thread::can_split(move _503)", "Thread::can_split(move _512)"),
    (PVS, "[0: bb309, otherwise: bb302]", "[0: bb302, otherwise: bb309]"),
    (PVS, "Thread::split(move _511", "Thread::split(move _512"),
    (SCOUT, "move _64, move _66, move _67", "move _64, move _64, move _67"),
    (TH + "search_active_split_point", "move _37, move _38, move _39", "move _37, move _38, move _34"),
    (TH + "search_active_split_point", "bb40 (cleanup): {\n        drop(_79) -> [return: bb43",
     "bb40 (cleanup): {\n        drop(_79) -> [return: bb13"),
    (DROP, "Thread::abandon_active_split", "Thread::detach_split"),
    (TH + "initialize_split_point", "{ thread: move _109, sp: move _110 }",
     "{ thread: move _110, sp: move _110 }"),
    (TH + "search_active_split_point-{impl#0}-drop", "((*_1).0: &search::threading::thread::Thread)",
     "((*_1).1: &search::threading::thread::Thread)"),
]


def check(dump, expected):
    return check_mir_projection(
        dump,
        expected,
        render,
        SUFFIX,
        CASES,
        file_filter=lambda p: "search-threading-" in p.name or p.name in (PVS + SUFFIX, SCOUT + SUFFIX),
    )


if __name__ == "__main__":
    run_mir_test_cli(__doc__, check, "{count} first-split receiver projection mutations detected")

"""Fault injection into validated rustc MIR. Takes full dump and existing extractor paths on PYTHONPATH."""

from pathlib import Path
import sys

from cache_owner_frontier import inspect
from cache_owner_mir import PASS, POOL, WORKER
from cache_owner_queue import EXECUTE
from extract_cache_owner import NODES, render, targets
from mir_mutants import assert_mutants_rejected


def check(dump, repo):
    changes = [
        (WORKER + "new", "SyncUnsafeCell::<EndGameCaches>::new(move _26)", "SyncUnsafeCell::<EndGameCaches>::new(move _99)"),
        (WORKER + "endgame_caches", ".13: std::cell::SyncUnsafeCell", ".12: std::cell::SyncUnsafeCell"),
        (POOL + "add_thread", "::push(move _19, move _20)", "::push(move _19, move _99)"),
        (POOL + "create_worker_threads", "{ thread: move _15 }", "{ thread: move _99 }"),
        (WORKER + "dispatch_search", "::<Root>(move _20, move _21, move _22, move _23", "::<Root>(move _20, move _21, move _22, move _99"),
        ("search-endgame-null_window_search", "((*_4).0: search::endgame::cache::EndGameCache", "((*_4).1: search::endgame::cache::EndGameCache"),
        ("search-endgame-{impl#0}-fail_low_score", "((*_1).0: search::endgame::cache::EndGameCache", "((*_1).1: search::endgame::cache::EndGameCache"),
        ("search-pvs-endgame_cache_cutoff", "_12 = &(*_8);", "_12 = &(*_99);"),
        ("search-endgame-search_move_nws_ec", "((*_6).1: search::endgame::cache::EndGameCache", "((*_6).0: search::endgame::cache::EndGameCache"),
        ("search-endgame-search_move_nws_ec", "_46 = &mut ((*_6).1:", "_333 = copy _6 as *mut () (PtrToPtr);\n        _46 = &mut ((*_6).1:"),
        (EXECUTE, "ThreadPool::start_thinking(move _10, move _13", "ThreadPool::start_thinking(move _10, move _99"),
        (POOL + "start_thinking", "_48 = move _12;", "_48 = move _99;"),
        (WORKER + "main_thread_loop", "search::search_root(move _32, move _33)", "search::search_root(move _99, move _33)"),
        (POOL + "start_helpers", "RangeFrom::<usize> { start: const 1_usize }", "RangeFrom::<usize> { start: const 0_usize }"),
        (WORKER + "book_helper_job", "(*_22) = move _19;", "(*_22) = move _99;"),
        (WORKER + "idle_loop", "_47 = &_44;", "_47 = &_99;"),
        (WORKER + "run_helper_job", "(move _12,)", "(move _99,)"),
    ]
    assert_mutants_rejected(
        dump,
        [(f"reversi_core.{stem}{PASS}", needle, value, stem) for stem, needle, value in changes],
        lambda d: render(d, repo),
        unique=False,
    )
    forbidden = [
        "_10 = threading::thread::Thread::endgame_caches(move _5) -> [return: bb1, unwind: bb2];",
        "_4 = &((*_3).13: std::cell::SyncUnsafeCell<search::endgame::EndGameCaches>);",
        "_4 = search::endgame::shallow_search as fn() (PointerCoercion(ReifyFnPointer, Implicit));",
        "_4 = search::endgame::shallow_search(move _5, move _6) -> [return: bb1, unwind: bb2];",
    ]
    for line in forbidden:
        try:
            inspect("foreign", ["    bb0: {", "        " + line], targets, NODES)
        except ValueError:
            continue
        raise AssertionError("accepted outside entry: " + line)
    return len(changes) + len(forbidden)


if __name__ == "__main__":
    print(f"{check(Path(sys.argv[1]), Path.cwd())} routed operand and external-entry mutations rejected")

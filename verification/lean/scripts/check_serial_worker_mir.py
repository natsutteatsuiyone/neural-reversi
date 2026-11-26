"""Check the one-worker branches used by the serial endgame model.

The pool roster, guarded split call, private drop guards and cutoff reader are
checked by extract_pool_roster_mir, extract_guard_receiver_mir and
check_cutoff_poll_mir. This check supplies their single-worker initial state and
checks the short circuits that keep the worker out of those scheduler paths.
"""

from argparse import ArgumentParser
from pathlib import Path
import re

from cache_owner_mir import Body, CLOSURE, PASS, POOL, THREAD, WORKER
from mir_common import checked
from check_move_iterator_mir import call, jump
from extract_cache_boot import record, sequence
from extract_header import ordinary
from mir_mutants import assert_mutants_rejected


FLIGHT = "std::option::Option<std::sync::Arc<search::threading::in_flight::InFlightTable>>"
ACTIVE = "std::cell::SyncUnsafeCell<std::option::Option<std::sync::Arc<search::threading::split_point::SplitPoint>>>"
ROSTER = f"std::vec::Vec<std::sync::Arc<search::{THREAD}>>"


def check_blocks(body, expected):
    for pc, want in expected.items():
        have = ordinary(body.cfg[pc])
        checked(have == want, f"{body.stem} bb{pc}: {have} differs from {want}")


def constructors(dump):
    pool = Body(dump, POOL + "new-{closure#0}")
    then, = [fn for _, fn, _, _ in pool.calls if re.fullmatch(
        rf"core::bool::<impl bool>::then::<Arc<InFlightTable>, {CLOSURE}>", fn)]
    sequence(pool, [then, "ThreadPool::init"])
    _, _, args, _ = pool.named(then)
    predicate = pool.resolve(args[0], {"Gt(move _20, const 1_usize)": "parallel"})
    checked(predicate == "parallel" and pool.defs["_20"] == ["copy (*(_1.0: &usize))"],
            "serial pool allocation predicate")
    local, = [reg for reg, ds in pool.defs.items() if any(
        isinstance(d, str) and d.startswith("ThreadPool { ") for d in ds)]
    data = record(pool, local, "ThreadPool")
    checked(list(data)[2] == "size" and pool.resolve(data["size"], {"(_1.0: &usize)": "size"}) == "size",
            "serial pool stored size")
    checked(list(data)[7] == "in_flight" and pool.resolve(data["in_flight"], pool.produced(then, "option", {})) == "option",
            "serial pool stored in-flight option")

    add = Body(dump, POOL + "add_thread")
    clone = "<Option<Arc<InFlightTable>> as Clone>::clone"
    _, _, inputs, _ = add.named(clone)
    _, _, args, _ = add.named(THREAD + "::new")
    checked(add.resolve(inputs[0], {f"((*_1).7: {FLIGHT})": "poolOption"}) == "poolOption" and
            add.resolve(args[5], add.produced(clone, "threadOption", {})) == "threadOption",
            "serial pool option forwarded to worker")
    worker = Body(dump, WORKER + "new")
    data = record(worker, "_0", THREAD)
    for field, slot in (("helper_job_aborted", 7), ("local_chain_cutoff", 10), ("pool_size", 11),
                        ("in_flight", 12), ("active_split_point", 17)):
        checked(list(data)[slot] == field, f"serial worker field {field}")
    for field in ("pool_size", "in_flight"):
        checked(worker.resolve(data[field], {worker.arg(field): field}) == field, f"serial worker stored {field}")
    none = "SyncUnsafeCell::<Option<Arc<SplitPoint>>>::new"
    _, _, args, _ = worker.named(none)
    checked(worker.resolve(args[0], {"Option::<Arc<SplitPoint>>::None": "none"}) == "none" and
            worker.resolve(data["active_split_point"], worker.produced(none, "cell", {})) == "cell",
            "serial worker initial split slot")
    for field in ("helper_job_aborted", "local_chain_cutoff"):
        reg = data[field].removeprefix("move ")
        checked(worker.defs[reg] == [("call", "Atomic::<bool>::new", ["const false"])], f"serial worker initial {field}")


def guards(dump):
    check_blocks(Body(dump, WORKER + "can_split"), {
        0: ["_3 = copy ((*_1).11: usize);", "_2 = move _3 as u32 (IntToInt);", "FakeRead(ForLet(None), _2);",
            "_6 = copy _2;", "_5 = Le(move _6, const 1_u32);", "switchInt(move _5) -> [0: bb2, otherwise: bb1];"],
        1: ["_0 = const false;", jump(24)],
        24: ["return;"],
    })
    claim = Body(dump, WORKER + "claim")
    closure, = [m[1] for _, line in claim.ops if (m := re.fullmatch(rf"_6 = ({CLOSURE}) \{{ child: move _7 \}};", line))]
    check_blocks(claim, {
        0: [f"_4 = &((*_1).12: {FLIGHT});", call("_3 = Option::<Arc<InFlightTable>>::as_ref(move _4)", 1, 4)],
        1: ["_5 = Claim::<'_>::Untracked;", "_7 = &(*_2);", f"_6 = {closure} {{ child: move _7 }};",
            call(f"_0 = Option::<&Arc<InFlightTable>>::map_or::<Claim<'_>, {closure}>(move _3, move _5, move _6)", 2, 3)],
        2: ["return;"],
    })
    check_blocks(Body(dump, WORKER + "active_split_point"), {
        0: [f"_4 = &((*_1).17: {ACTIVE});", call("_3 = SyncUnsafeCell::<Option<Arc<SplitPoint>>>::get(move _4)", 1, 2)],
        1: ["_2 = &(*_3);", "_0 = &(*_2);", "return;"],
    })
    check_blocks(Body(dump, WORKER + "has_active_split_point"), {
        0: ["_4 = &(*_1);", call(f"_3 = {THREAD}::active_split_point(move _4)", 1, 3)],
        1: ["_2 = &(*_3);", call("_0 = Option::<Arc<SplitPoint>>::is_some(move _2)", 2, 3)],
        2: ["return;"],
    })


def empty_helper_suffixes(dump):
    # On a one-entry roster both loops see None on the first next after [1..].
    arc = f"Arc<{THREAD}>"
    index = f"<Vec<{arc}> as Index<std::ops::RangeFrom<usize>>>::index"
    into = f"<&[{arc}] as IntoIterator>::into_iter"
    next_ = f"<std::slice::Iter<'_, {arc}> as Iterator>::next"
    suffix = "std::ops::RangeFrom::<usize> { start: const 1_usize }"
    check_blocks(Body(dump, POOL + "start_helpers"), {
        0: [f"_6 = &((*_1).0: {ROSTER});", f"_7 = {suffix};", call(f"_5 = {index}(move _6, move _7)", 1, 15)],
        1: ["_4 = &(*_5);", call(f"_3 = {into}(move _4)", 2, 15)],
        2: ["PlaceMention(_3);", "_8 = move _3;", jump(3)],
        3: ["falseUnwind -> [real: bb4, unwind: bb15];"],
        4: ["_13 = &mut _8;", "_12 = &mut (*_13);", call(f"_11 = {next_}(move _12)", 5, 15)],
        5: ["PlaceMention(_11);", "_14 = discriminant(_11);", "switchInt(move _14) -> [0: bb7, 1: bb8, otherwise: bb6];"],
        7: ["falseEdge -> [real: bb9, imaginary: bb8];"],
        9: ["_0 = const ();", "return;"],
    })
    check_blocks(Body(dump, POOL + "stop_helpers"), {
        0: [f"_4 = &((*_1).0: {ROSTER});", f"_5 = {suffix};", call(f"_3 = {index}(move _4, move _5)", 1, 22)],
        1: ["_2 = &(*_3);", "FakeRead(ForLet(None), _2);", "_8 = copy _2;", call(f"_7 = {into}(move _8)", 2, 22)],
        2: ["PlaceMention(_7);", "_9 = move _7;", jump(3)],
        3: ["falseUnwind -> [real: bb4, unwind: bb22];"],
        4: ["_14 = &mut _9;", "_13 = &mut (*_14);", call(f"_12 = {next_}(move _13)", 5, 22)],
        5: ["PlaceMention(_12);", "_15 = discriminant(_12);", "switchInt(move _15) -> [0: bb7, 1: bb8, otherwise: bb6];"],
        7: ["falseEdge -> [real: bb9, imaginary: bb8];"],
        9: ["_6 = const ();", "_23 = copy _2;", call(f"_22 = {into}(move _23)", 12, 22)],
        12: ["PlaceMention(_22);", "_24 = move _22;", jump(13)],
        13: ["falseUnwind -> [real: bb14, unwind: bb22];"],
        14: ["_28 = &mut _24;", "_27 = &mut (*_28);", call(f"_26 = {next_}(move _27)", 15, 22)],
        15: ["PlaceMention(_26);", "_29 = discriminant(_26);", "switchInt(move _29) -> [0: bb17, 1: bb18, otherwise: bb16];"],
        17: ["falseEdge -> [real: bb19, imaginary: bb18];"],
        19: ["_0 = const ();", "return;"],
    })


def effects(dump):
    # The already checked guard coverage controls these protected entrypoints.
    # The active-slot setter and helper abort flag cannot be introduced elsewhere.
    expected = {
        f".17: {ACTIVE})": {WORKER + "active_split_point", WORKER + "active_split_point_mut"},
        f".12: {FLIGHT})": {WORKER + "claim"},
        f".7: {FLIGHT})": {POOL + "add_thread"},
        ".7: std::sync::atomic::Atomic<bool>)": {WORKER + name for name in
            ("is_search_aborted", "abort_helper_job", "wait_for_helper_job")},
        ".10: std::sync::atomic::Atomic<bool>)": {WORKER + name for name in
            ("reset_cutoff_cache", "split_point_cutoff_occurred", "should_stop_after_split")},
        "Thread::active_split_point_mut(": {WORKER + name for name in
            ("book_into", "initialize_split_point", "finalize_split_point", "detach_split")},
        "Thread::abort_helper_job(": {POOL + "stop_helpers"},
        "Thread::wait_for_helper_job(": {POOL + "stop_helpers"},
        "Thread::book_helper_job(": {POOL + "start_helpers"},
    }
    found = {token: set() for token in expected}
    for path in dump.glob("reversi_core.*" + PASS):
        source = path.read_text(encoding="utf-8")
        for token, sites in found.items():
            if token in source:
                sites.add(path.name.removeprefix("reversi_core.").removesuffix(PASS))
    for token, sites in found.items():
        checked(sites == expected[token], f"serial state access {token}: {sites} differs from {expected[token]}")


def check(dump):
    constructors(dump)
    guards(dump)
    empty_helper_suffixes(dump)
    effects(dump)


def check_mutants(dump):
    variants = (
        (constructors, POOL + "new-{closure#0}", "Gt(move _20, const 1_usize)", "Ge(move _20, const 1_usize)"),
        (constructors, WORKER + "new", "_24 = move _6;", "_24 = Option::<Arc<InFlightTable>>::None;"),
        (guards, WORKER + "can_split", "Le(move _6, const 1_u32)", "Lt(move _6, const 1_u32)"),
        (guards, WORKER + "claim", "_5 = Claim::<'_>::Untracked;", "_5 = Claim::<'_>::Busy;"),
        (guards, WORKER + "has_active_split_point", ">::is_some(move _2)", ">::is_none(move _2)"),
        (empty_helper_suffixes, POOL + "stop_helpers", "start: const 1_usize", "start: const 0_usize"),
        (effects, WORKER + "detach_split", "Thread::reset_cutoff_cache(move _8)", "Thread::abort_helper_job(move _8)"),
    )
    for verify, stem, old, new in variants:
        assert_mutants_rejected(dump, [(f"reversi_core.{stem}{PASS}", old, new, stem)], verify)
    return len(variants)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    args = parser.parse_args()
    check(args.mir)
    print(f"verified serial worker state and rejected {check_mutants(args.mir)} MIR mutants")

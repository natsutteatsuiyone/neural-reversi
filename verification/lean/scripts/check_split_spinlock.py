"""Follow the split guard to its mutex, successful CAS, and releasing store."""

import re

from check_move_iterator_mir import RELAXED, call, compare, graph, jump
from mir_common import CALL, checked
from extract_split_activation_mir import THREAD, drop_of
from extract_split_scheduler_mir import Body, edges
from mir_mutants import assert_mutants_rejected

SUFFIX = ".SimplifyCfg-initial.after.mir"
SLOT = "reversi_core.search-threading-split_point-{impl#5}-lock"
WORKER = "reversi_core.search-threading-thread-{impl#1}-"
LOCK = "reversi_core.util-spinlock-{impl#0}-"
DROP = "reversi_core.util-spinlock-{impl#1}-drop"
FLAG = "&((*_1).0: std::sync::atomic::Atomic<bool>)"
ACQUIRE = "std::sync::atomic::Ordering::Acquire"
RELEASE = "std::sync::atomic::Ordering::Release"


def check_snapshot(dump):
    body = Body(dump, "search_active_split_point", ["worker"])
    active = (THREAD + "active_split_point", "worker")
    slot = ("unSome", active)
    task = ("SplitPoint::task", slot)
    body.call(THREAD + "active_split_point", ["worker"], "read")
    body.call("<Option<Arc<SplitPoint>> as Clone>::clone", [active], "clone")
    body.call(THREAD + "dispatch_search", ["worker", ("SearchContext::<'_>::from_split_task", task),
        task, ("SplitPoint::move_iter", slot), slot], "dispatch")
    # Label each guard with its MIR local so the drop must match its own acquisition.
    locks = []
    for pc, lines in body.cfg.items():
        site = re.fullmatch(CALL, lines[-1])
        if site and site[2] == THREAD + "lock":
            checked(body.val(site[3]) == "worker", "state lock belongs to another worker")
            guard = site[1]
            locks.append(guard)
            body.marks[pc] = [("lock", guard)]
            for end in drop_of(body, guard):
                body.marks[end] = [("drop", guard)]
    checked(len(set(locks)) == 2, "worker snapshot and detach need distinct guards")
    path = body.projection()
    checked(any(path == [("lock", g), "read", "clone", ("drop", g), "dispatch",
                         ("lock", h), ("drop", h)] for g, h in (locks, locks[::-1])),
            "worker must clone the dispatched slot under its own state lock")


def check(dump):
    def body(name):
        return graph((dump / (name + SUFFIX)).read_text(encoding="utf-8"))

    for name, field in [(SLOT, 0), (WORKER + "lock", 1)]:
        compare(name, body(name), [
            [f"_2 = &((*_1).{field}: util::spinlock::SpinLock);", call("_0 = SpinLock::lock(move _2)", 1, 2)],
            ["return;"], ["unwind"]])
    check_snapshot(dump)
    compare("guard constructor", body(LOCK + "lock"), [
        ["_4 = &(*_1);", call("_3 = SpinLock::try_acquire(move _4)", 1, 6)],
        ["switchInt(move _3) -> [0: bb3, otherwise: bb2];"], ["_2 = const ();", jump(5)],
        ["_6 = &(*_1);", call("_5 = SpinLock::lock_slow(move _6)", 4, 6)],
        ["_2 = const ();", jump(5)], ["_7 = &(*_1);", "_0 = SpinLockGuard::<'_>(move _7);", "return;"],
        ["unwind"]])
    compare("fast acquisition", body(LOCK + "try_acquire"), [
        [f"_3 = {FLAG};", f"_4 = {RELAXED};", call("_2 = Atomic::<bool>::load(move _3, move _4)", 1, 7)],
        ["switchInt(move _2) -> [0: bb3, otherwise: bb2];"], ["_0 = const false;", jump(6)],
        [f"_7 = {FLAG};", f"_8 = {ACQUIRE};", f"_9 = {RELAXED};",
         call("_6 = Atomic::<bool>::compare_exchange(move _7, const false, const true, move _8, move _9)", 4, 7)],
        ["_5 = &_6;", call("_0 = Result::<bool, bool>::is_ok(move _5)", 5, 7)], [jump(6)], ["return;"], ["unwind"]])

    slow = body(LOCK + "lock_slow")
    compare("successful slow acquisition", {i: slow[i+16] for i in range(4)}, [
        [f"_25 = {FLAG};", f"_26 = {ACQUIRE};", f"_27 = {RELAXED};",
         call("_24 = Atomic::<bool>::compare_exchange_weak(move _25, const false, const true, move _26, move _27)", 17, 36)],
        ["_23 = &_24;", call("_22 = Result::<bool, bool>::is_ok(move _23)", 18, 36)],
        ["switchInt(move _22) -> [0: bb20, otherwise: bb19];"], ["_0 = const ();", "return;"]])
    checked([pc for pc, ls in slow.items() if ls[-1] == "return;"] == [19], "slow lock returned elsewhere")
    incoming = {17: [], 18: [], 19: []}
    for pc, ls in slow.items():
        if ls[-1] != "unwind":
            for dst in edges(ls[-1]):
                if dst in incoming:
                    incoming[dst].append(pc)
    checked(incoming == {17: [16], 18: [17], 19: [18]}, "slow return can bypass a successful CAS")

    compare("spin guard drop", body(DROP), [
        ["_3 = &(*((*_1).0: &util::spinlock::SpinLock));", call("_2 = SpinLock::unlock(move _3)", 1, 2)],
        ["_0 = const ();", "return;"], ["unwind"]])
    compare("spin guard release", body(LOCK + "unlock"), [
        [f"_3 = {FLAG};", f"_4 = {RELEASE};", call("_2 = Atomic::<bool>::store(move _3, const false, move _4)", 1, 2)],
        ["_0 = const ();", "return;"], ["unwind"]])


def check_mutants(dump):
    edits = [
        (SLOT, "((*_1).0:", "((*_1).1:"),
        (WORKER + "lock", "((*_1).1:", "((*_1).0:"),
        (WORKER + "search_active_split_point", "return: bb5, unwind: bb47", "return: bb6, unwind: bb47"),
        (WORKER + "search_active_split_point", "drop(_7) -> [return: bb8", "drop(_57) -> [return: bb8"),
        (LOCK + "lock", "0: bb3, otherwise: bb2", "0: bb2, otherwise: bb3"),
        (LOCK + "try_acquire", ACQUIRE, RELAXED),
        (LOCK + "lock_slow", "goto -> bb21;", "goto -> bb19;"),
        (LOCK + "lock_slow", "0: bb20, otherwise: bb19", "0: bb19, otherwise: bb20"),
        (DROP, "SpinLock::unlock(move _3)", "SpinLock::try_acquire(move _3)"),
        (LOCK + "unlock", RELEASE, RELAXED),
        (LOCK + "unlock", "move _3, const false", "move _3, const true"),
    ]
    return assert_mutants_rejected(
        dump,
        [(name + SUFFIX, old, new, name) for name, old, new in edits],
        check,
        unique=False,
    )

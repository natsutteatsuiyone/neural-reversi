"""Check the active-chain walk, cached stop poll and publication of a split cutoff."""

import argparse
from pathlib import Path

from check_move_iterator_mir import call, compare, graph, jump


def switch(reg, zero, nonzero):
    return f"switchInt(move _{reg}) -> [0: bb{zero}, otherwise: bb{nonzero}];"


def check(dump):
    suffix = ".SimplifyCfg-initial.after.mir"

    def body(thread, name, blocks):
        ns = "thread-{impl#1}" if thread else "split_point-{impl#0}"
        path = dump / ("reversi_core.search-threading-" + ns + "-" + name + suffix)
        src = path.read_text(encoding="utf-8")
        for qualifier in ("std::sync::atomic::", "std::sync::", "util::align::",
                          "search::threading::split_point::", "threading::thread::"):
            src = src.replace(qualifier, "")
        compare(name, graph(src), blocks)

    def typed_field(reg, obj, index, ty):
        return f"_{reg} = &((*_{obj}).{index}: {ty});"

    def order(reg, name):
        return f"_{reg} = Ordering::{name};"

    flag = "Align64<Atomic<bool>>"
    epoch = "Align64<Atomic<u64>>"
    glob = f"Arc<{epoch}>"
    done = ["return;"]
    unwind = ["unwind"]

    for name, operation, ordering, result in [
        ("clear_cutoff", "store(move _3, const false, move _6)", "Release", "_0 = const ();"),
        ("mark_cutoff", "swap(move _3, const true, move _6)", "AcqRel", "_0 = Not(move _2);"),
    ]:
        body(False, name, [
            [typed_field(5, 1, 4, flag), call(f"_4 = <{flag} as Deref>::deref(move _5)", 1, 3)],
            ["_3 = &(*_4);", order(6, ordering), call(f"_2 = Atomic::<bool>::{operation}", 2, 3)],
            [result, "return;"], unwind])
    body(False, "cutoff", [
        [typed_field(4, 1, 4, flag), call(f"_3 = <{flag} as Deref>::deref(move _4)", 1, 3)],
        ["_2 = &(*_3);", order(5, "Relaxed"), call("_0 = Atomic::<bool>::load(move _2, move _5)", 2, 3)],
        done, unwind])
    body(True, "mark_split_point_cutoff", [
        ["_4 = &(*_2);", call("_3 = SplitPointState::mark_cutoff(move _4)", 1, 8)], [switch(3, 6, 2)],
        [typed_field(10, 1, 8, glob), call(f"_9 = <{glob} as Deref>::deref(move _10)", 3, 8)],
        ["_8 = &(*_9);", call(f"_7 = <{epoch} as Deref>::deref(move _8)", 4, 8)],
        ["_6 = &(*_7);", order(11, "Release"), call("_5 = Atomic::<u64>::fetch_add(move _6, const 1_u64, move _11)", 5, 8)],
        ["_0 = const ();", jump(7)], ["_0 = const ();", jump(7)], done, unwind])
    body(True, "reset_cutoff_cache", [
        [typed_field(3, 1, 10, "Atomic<bool>"), order(4, "Relaxed"),
         call("_2 = Atomic::<bool>::store(move _3, const false, move _4)", 1, 7)],
        [typed_field(6, 1, 9, "Atomic<u64>"), typed_field(13, 1, 8, glob),
         call(f"_12 = <{glob} as Deref>::deref(move _13)", 2, 7)],
        ["_11 = &(*_12);", call(f"_10 = <{epoch} as Deref>::deref(move _11)", 3, 7)],
        ["_9 = &(*_10);", order(14, "Relaxed"), call("_8 = Atomic::<u64>::load(move _9, move _14)", 4, 7)],
        [call("_7 = core::num::<impl u64>::wrapping_sub(move _8, const 1_u64)", 5, 7)],
        [order(15, "Relaxed"), call("_5 = Atomic::<u64>::store(move _6, move _7, move _15)", 6, 7)],
        ["_0 = const ();", "return;"], unwind])
    body(True, "split_point_cutoff_occurred", [
        [typed_field(4, 1, 10, "Atomic<bool>"), order(5, "Relaxed"), call("_3 = Atomic::<bool>::load(move _4, move _5)", 1, 17)],
        [switch(3, 3, 2)], ["_0 = const true;", jump(16)],
        ["_2 = const ();", typed_field(12, 1, 8, glob), call(f"_11 = <{glob} as Deref>::deref(move _12)", 4, 17)],
        ["_10 = &(*_11);", call(f"_9 = <{epoch} as Deref>::deref(move _10)", 5, 17)],
        ["_8 = &(*_9);", order(13, "Acquire"), call("_7 = Atomic::<u64>::load(move _8, move _13)", 6, 17)],
        ["_16 = copy _7;", typed_field(18, 1, 9, "Atomic<u64>"), order(19, "Relaxed"),
         call("_17 = Atomic::<u64>::load(move _18, move _19)", 7, 17)],
        ["_15 = Ne(move _16, move _17);", switch(15, 14, 8)],
        [typed_field(21, 1, 9, "Atomic<u64>"), "_22 = copy _7;", order(23, "Relaxed"),
         call("_20 = Atomic::<u64>::store(move _21, move _22, move _23)", 9, 17)],
        ["_25 = &(*_1);", call("_24 = Thread::cutoff_occurred(move _25)", 10, 17)],
        [typed_field(27, 1, 10, "Atomic<bool>"), "_28 = copy _24;", order(29, "Relaxed"),
         call("_26 = Atomic::<bool>::store(move _27, move _28, move _29)", 11, 17)],
        ["_30 = copy _24;", switch(30, 13, 12)], ["_0 = const true;", jump(16)],
        ["_14 = const ();", jump(15)], ["_14 = const ();", jump(15)], ["_0 = const false;", jump(16)], done, unwind])
    body(True, "should_stop", [
        ["_4 = &(*_1);", call("_3 = Thread::split_point_cutoff_occurred(move _4)", 1, 6)],
        [switch(3, 3, 2)], ["_0 = const true;", jump(5)],
        ["_2 = const ();", "_6 = &(*_1);", call("_0 = Thread::is_search_aborted(move _6)", 4, 6)],
        [jump(5)], done, unwind])
    body(True, "should_stop_after_split", [
        ["_4 = &(*_1);", call("_3 = Thread::cutoff_occurred(move _4)", 1, 7)], [switch(3, 4, 2)],
        [typed_field(7, 1, 10, "Atomic<bool>"), order(8, "Relaxed"),
         call("_6 = Atomic::<bool>::store(move _7, const true, move _8)", 3, 7)],
        ["_0 = const true;", jump(6)], ["_2 = const ();", "_9 = &(*_1);", call("_0 = Thread::is_search_aborted(move _9)", 5, 7)],
        [jump(6)], done, unwind])
    body(True, "cutoff_occurred", [
        ["_7 = &(*_1);", call("_6 = Thread::active_split_point(move _7)", 1, 8)],
        ["_5 = &(*_6);", call("_4 = Option::<Arc<SplitPoint>>::as_ref(move _5)", 2, 8)],
        ["_8 = discriminant(_4);", "switchInt(move _8) -> [1: bb3, otherwise: bb5];"],
        ["falseEdge -> [real: bb4, imaginary: bb5];"],
        ["_3 = copy ((_4 as Some).0: &Arc<SplitPoint>);", "_9 = &(*_1);", "_10 = &(*_3);",
         call("_0 = Thread::cutoff_occurred_slow(move _9, move _10)", 6, 8)],
        ["_0 = const false;", jump(7)], [jump(7)], done, unwind])
    body(True, "cutoff_occurred_slow", [
        ["_4 = copy _2;", "_3 = Option::<&Arc<SplitPoint>>::Some(move _4);", jump(1)],
        ["falseUnwind -> [real: bb2, unwind: bb15];"],
        ["_7 = discriminant(_3);", "switchInt(move _7) -> [1: bb3, otherwise: bb13];"],
        ["falseEdge -> [real: bb4, imaginary: bb13];"],
        ["_8 = copy ((_3 as Some).0: &Arc<SplitPoint>);", "_12 = &(*_8);", call("_11 = <Arc<SplitPoint> as Deref>::deref(move _12)", 5, 15)],
        ["_10 = &(*_11);", call("_9 = SplitPoint::state(move _10)", 6, 15)],
        ["_15 = &(*_9);", call("_14 = SplitPointState::cutoff(move _15)", 7, 15)],
        [switch(14, 9, 8)], ["_0 = const true;", jump(14)],
        ["_13 = const ();", "_22 = &(*_8);", call("_21 = <Arc<SplitPoint> as Deref>::deref(move _22)", 10, 15)],
        ["_20 = &(*_21);", call("_19 = SplitPoint::task(move _20)", 11, 15)],
        [typed_field(18, 19, 11, "std::option::Option<Arc<SplitPoint>>"), call("_17 = Option::<Arc<SplitPoint>>::as_ref(move _18)", 12, 15)],
        ["_3 = move _17;", "_6 = const ();", jump(1)], ["_5 = const ();", "_0 = const false;", jump(14)], done, unwind])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    check(parser.parse_args().mir)
    print("verified split cutoff publication, cached poll and active ancestor walk")

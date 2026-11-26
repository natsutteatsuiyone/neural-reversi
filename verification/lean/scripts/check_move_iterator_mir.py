"""Check the production concurrent move iterator and the toolchain's relaxed try_update loop."""

import argparse
from pathlib import Path
import re
import subprocess
import tempfile


ITER = "move_list-iterator-{impl#1}-"
ARRAY = "move_list-move_array-{impl#0}-"
MOVE_ARRAY = "&(((*_1).0: move_list::MoveList).0: move_list::move_array::MoveArray)"
WORD = "std::sync::atomic::Atomic<usize>"
RELAXED = "std::sync::atomic::Ordering::Relaxed"
ORDER = "std::intrinsics::AtomicOrdering::Relaxed"
CELL = "std::cell::UnsafeCell<std::sync::atomic::private::AlignWORD<usize>>"
UPDATE = "{update}"
INC = "{increment}"


def graph(source):
    blocks = {}
    for match in re.finditer(r"^    bb(\d+)( \(cleanup\))?: \{\n(.*?)^    \}", source, re.M | re.S):
        index = int(match[1])
        if index in blocks:
            raise ValueError(f"duplicate bb{index}")
        lines = []
        for line in map(str.strip, match[3].splitlines()):
            if re.fullmatch(r"Storage(?:Live|Dead)\(_\d+\);|PlaceMention\(_\d+\);|"
                            r"FakeRead\(For(?:Let|MatchedPlace)\(None\), _\d+\);", line):
                continue
            lines.append(line)
        if match[2]:
            if lines != ["resume;"]:
                raise ValueError(f"unrecognized cleanup bb{index}: {lines}")
            lines = ["unwind"]
        blocks[index] = lines
    return blocks


def compare(name, body, expected):
    if len(body) != len(expected) or set(body) != set(range(len(expected))):
        raise ValueError(f"{name}: unexpected block numbers {list(body)}")
    for pc, instructions in enumerate(expected):
        if body[pc] != instructions:
            raise ValueError(f"{name} bb{pc}: expected {instructions}, got {body[pc]}")


def call(assignment, target, unwind):
    return f"{assignment} -> [return: bb{target}, unwind: bb{unwind}];"


def jump(target):
    return f"goto -> bb{target};"


def increment(value, pair, result, target, unwind):
    checked_add = [f"{pair} = AddWithOverflow(copy {value}, const 1_usize);",
                   f'assert(!move ({pair}.1: bool), "attempt to compute `{{}} + {{}}`, which would overflow", '
                   f'move {value}, const 1_usize) -> [success: bb{target}, unwind: bb{unwind}];']
    return checked_add, f"{result} = move ({pair}.0: usize);"


def check_iterator(dump):
    def source(stem):
        path = dump / f"reversi_core.{stem}.SimplifyCfg-initial.after.mir"
        return path.read_text(encoding="utf-8").replace("\\", "/")

    def closure(stem):
        text = source(ITER + stem)
        match = re.search(r"^fn iterator::<impl at [^\n]+>::next(?:::\{closure#0\})+"
                          r"\(_1: (?:&mut )?(\{closure@(?:[^{}]+/)?iterator\.rs:\d+:\d+: \d+:\d+\})", text, re.M)
        if not match:
            raise ValueError(f"{stem}: missing closure identity")
        return match[1]

    outer = closure("next-{closure#0}")
    inner = closure("next-{closure#0}-{closure#0}")
    if outer == inner:
        raise ValueError("update and increment closures have the same identity")

    def cfg(stem):
        return graph(source(stem).replace(outer, UPDATE).replace(inner, INC))

    compare("from_offset", cfg(ITER + "from_offset"), [
        ["_3 = move _1;", "_5 = copy _2;", call("_4 = Atomic::<usize>::new(move _5)", 1, 2)],
        ["_0 = ConcurrentMoveIterator { move_list: move _3, current: move _4 };", "return;"], ["unwind"]])
    compare("update closure", cfg(ITER + "next-{closure#0}"), [
        ["_4 = copy _2;", "_5 = copy (*((*_1).0: &usize));", "_3 = Lt(move _4, move _5);", "_7 = &_2;",
         f"_6 = {INC} {{ index: move _7 }};",
         call(f"_0 = core::bool::<impl bool>::then::<usize, {INC}>(move _3, move _6)", 1, 2)],
        ["return;"], ["unwind"]])
    inner_body = cfg(ITER + "next-{closure#0}-{closure#0}")
    old = "_2 = copy (*(_1.0: &usize));"
    if len(inner_body) == 1:
        compare("increment closure", inner_body, [[old, "_0 = Add(move _2, const 1_usize);", "return;"]])
    else:
        add, value = increment("_2", "_3", "_0", 1, 2)
        compare("increment closure", inner_body, [[old, *add], [value, "return;"], ["unwind"]])

    body = cfg(ITER + "next")
    checked = len(body) == 14
    finish = 12 if checked else 11
    unwind = finish + 1
    expected = [
        [f"_3 = {MOVE_ARRAY};", call("_2 = MoveArray::len(move _3)", 1, unwind)],
        [f"_8 = &((*_1).1: {WORD});", f"_9 = {RELAXED};", f"_10 = {RELAXED};", "_12 = &_2;",
         f"_11 = {UPDATE} {{ len: move _12 }};",
         call(f"_7 = Atomic::<usize>::try_update::<{UPDATE}>(move _8, move _9, move _10, move _11)", 2, unwind)],
        [call("_6 = Result::<usize, usize>::ok(move _7)", 3, unwind)],
        [call("_5 = <Option<usize> as Try>::branch(move _6)", 4, unwind)],
        ["_13 = discriminant(_5);", "switchInt(move _13) -> [0: bb7, 1: bb6, otherwise: bb5];"],
        ["unreachable;"], ["falseEdge -> [real: bb8, imaginary: bb7];"],
        ["_17 = copy ((_5 as Continue).0: usize);", "_4 = copy _17;", f"_21 = {MOVE_ARRAY};", "_22 = copy _4;",
         call("_20 = MoveArray::get_unchecked_ref(move _21, move _22)", 10, unwind)],
        ["_14 = copy ((_5 as Break).0: std::option::Option<!>);", "_16 = copy _14;",
         call("_0 = <Option<(&move_list::Move, usize)> as FromResidual<Option<!>>>::from_residual(move _16)", 9, unwind)],
        [jump(finish)]]
    prepare = ["_19 = &(*_20);", "_24 = copy _4;"]
    deliver = ["_18 = (move _19, move _23);", "_0 = Option::<(&move_list::Move, usize)>::Some(move _18);", jump(finish)]
    if checked:
        add, value = increment("_24", "_25", "_23", 11, unwind)
        expected.extend([[*prepare, *add], [value, *deliver]])
    else:
        expected.append([*prepare, "_23 = Add(move _24, const 1_usize);", *deliver])
    compare("next", body, [*expected, ["return;"], ["unwind"]])

    compare("MoveArray::len", cfg(ARRAY + "len"), [["_0 = copy ((*_1).1: usize);", "return;"]])
    compare("MoveArray::as_ptr", cfg(ARRAY + "as_ptr"), [
        ["_4 = &((*_1).0: [std::mem::MaybeUninit<move_list::Move>; 35]);",
         "_3 = move _4 as &[std::mem::MaybeUninit<move_list::Move>] (PointerCoercion(Unsize, Implicit));",
         call("_2 = core::slice::<impl [MaybeUninit<move_list::Move>]>::as_ptr(move _3)", 1, 3)],
        [call("_0 = std::ptr::const_ptr::<impl *const MaybeUninit<move_list::Move>>::cast::<move_list::Move>(move _2)", 2, 3)],
        ["return;"], ["unwind"]])
    access = cfg(ARRAY + "get_unchecked_ref")
    enabled = access[0][0]
    if enabled not in ("_4 = const true;", "_4 = const false;"):
        raise ValueError("MoveArray::get_unchecked_ref: unexpected debug guard")
    compare("MoveArray::get_unchecked_ref", access, [
        [enabled, "switchInt(move _4) -> [0: bb4, otherwise: bb1];"],
        ["_7 = copy _2;", "_8 = copy ((*_1).1: usize);", "_6 = Lt(move _7, move _8);",
         "switchInt(move _6) -> [0: bb3, otherwise: bb2];"],
        ["_5 = const ();", "_3 = const ();", jump(5)], ['_9 = panic(const "assertion failed: index < self.len") -> bb8;'],
        ["_3 = const ();", jump(5)], ["_13 = &(*_1);", call("_12 = MoveArray::as_ptr(move _13)", 6, 8)],
        ["_14 = copy _2;", call("_11 = std::ptr::const_ptr::<impl *const move_list::Move>::add(move _12, move _14)", 7, 8)],
        ["_10 = &(*_11);", "_0 = &(*_10);", "return;"], ["unwind"]])


def check_atomic(source):
    if "fn update(_1: &Atomic<usize>, _2: fn(usize) -> Option<usize>) -> Result<usize, usize>" not in source:
        raise ValueError("unexpected atomic probe parameters")
    layouts = set(re.findall(r"atomic::private::Align(\d+)<usize>", source))
    if layouts not in ({"4"}, {"8"}):
        raise ValueError(f"unexpected atomic word layout {layouts}")
    body = graph(source.replace(f"Align{next(iter(layouts))}<usize>", "AlignWORD<usize>"))
    cell = f"&raw const ((*_1).0: {CELL})"
    compare("core::AtomicUsize::try_update", body, [
        [f"_9 = {cell};", "_8 = copy _9 as *const usize (PtrToPtr);",
         f"_7 = atomic_load::<usize, {ORDER}, false>(move _8) -> [return: bb6, unwind unreachable];"],
        ["return;"], ["_4 = copy _7;", "_3 = copy _2(move _4) -> [return: bb7, unwind unreachable];"],
        ["_6 = copy ((_3 as Some).0: usize);", f"_11 = {cell};", "_10 = copy _11 as *mut usize (PtrToPtr);",
         f"_14 = atomic_cxchgweak::<usize, {ORDER}, {ORDER}>(move _10, move _7, move _6) -> [return: bb8, unwind unreachable];"],
        ["unreachable;"], ["_0 = Result::<usize, usize>::Err(move _7);", jump(1)], [jump(2)],
        ["_5 = discriminant(_3);", "switchInt(move _5) -> [1: bb3, 0: bb5, otherwise: bb4];"],
        ["_12 = copy (_14.0: usize);", "_13 = copy (_14.1: bool);", "switchInt(copy _13) -> [0: bb10, otherwise: bb9];"],
        ["_0 = Result::<usize, usize>::Ok(copy _12);", jump(1)],
        ["_0 = Result::<usize, usize>::Err(copy _12);", "_7 = copy ((_0 as Err).0: usize);", jump(2)]])


def compile_atomic(repo):
    # The opaque function pointer keeps the return/retry paths visible after the stdlib is inlined.
    probe = """use std::sync::atomic::{AtomicUsize, Ordering};
pub fn update(a: &AtomicUsize, f: fn(usize) -> Option<usize>) -> Result<usize,usize> {
    a.try_update(Ordering::Relaxed, Ordering::Relaxed, f)
}
"""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.rs"
        mir = Path(tmp) / "probe.mir"
        src.write_text(probe, encoding="utf-8")
        subprocess.run(["rustc", "--crate-type=lib", "--emit=mir", "-O", "-Cpanic=abort", "-Zmir-opt-level=3",
                        "-Zinline-mir=yes", "-Zinline-mir-threshold=1000", "-Zinline-mir-hint-threshold=1000",
                        "-Zinline-mir-forwarder-threshold=1000", str(src), "-o", str(mir)], cwd=repo, check=True)
        check_atomic(mir.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True, help="pinned SimplifyCfg-initial dump of reversi-core")
    args = parser.parse_args()
    check_iterator(args.mir)
    compile_atomic(Path(__file__).resolve().parents[3])
    print("verified the concurrent move iterator, move array, and relaxed try_update intrinsics")


if __name__ == "__main__":
    main()

"""Check possible uses of the evaluator's unique reference to root move storage."""

import re

from mir_common import blocks, CALL, checked


def verify_value_writes(source):
    signature, = re.findall(r"^fn .+\(_1: &mut MoveList, .+$", source, re.M)
    checked(signature.endswith(" -> () {"), "unexpected ordering function result")
    decl = dict(re.findall(r"^\s*let (?:mut )?(_\d+): (.*);$", source, re.M))
    decl["_1"] = "&mut move_list::MoveList"
    # Iterator, Option and reborrow variants all retain the capability to mutate the move array.
    capability = {local for local, ty in decl.items()
                  if "IterMut<'_, move_list::Move>" in ty or "&mut move_list::Move" in ty}
    elements = {local for local, ty in decl.items() if ty == "&mut move_list::Move"}
    functions = {
        "MoveList::iter_mut", "<std::slice::IterMut<'_, move_list::Move> as IntoIterator>::into_iter",
        "<std::slice::IterMut<'_, move_list::Move> as Iterator>::next",
        "ordering::<impl MoveList>::evaluate_moves_fast",
        "ordering::<impl MoveList>::evaluate_moves_by_search::<NT, SS>",
    }
    for instructions in blocks(source).values():
        for ins in instructions:
            used = set(re.findall(r"\b_\d+\b", ins)) & capability
            if not used:
                continue
            if re.fullmatch(r"Storage(?:Live|Dead)\(_\d+\);|PlaceMention\(_\d+\);|FakeRead\(For(?:Let|MatchedPlace)\(None\), _\d+\);", ins):
                continue
            if m := re.fullmatch(CALL, ins):
                checked(m[2] in functions, f"move storage escapes through {ins}")
                args = m[3].split(", ")
                checked(re.fullmatch(r"(?:move|copy) _\d+", args[0]), f"unsupported ordering receiver {ins}")
                allowed = {m[1], args[0].split()[1]}
                checked(used <= allowed, f"mutable move arguments outside their receiver {ins}")
                continue
            if m := re.fullmatch(r"(_\d+) = (?:&mut \(\*(_\d+)\)|&mut (_\d+)|(?:move|copy) (_\d+));", ins):
                checked(m[1] in capability and (m[2] or m[3] or m[4]) in capability, f"unexpected alias {ins}")
                continue
            if m := re.fullmatch(r"(_\d+) = move \(\((_\d+) as Some\)\.0: &mut move_list::Move\);", ins):
                checked(m[1] in elements and m[2] in capability, f"unexpected next result {ins}")
                continue
            if m := re.fullmatch(r"(_\d+) = discriminant\((_\d+)\);", ins):
                checked(m[1] not in capability and m[2] in capability, f"unexpected discriminant {ins}")
                continue
            if m := re.fullmatch(r"\(\(\*(_\d+)\)\.1: i32\) = (.*);", ins):
                checked(m[1] in elements and used == {m[1]}, f"unexpected write to move {ins}")
                continue
            if m := re.fullmatch(r"(_\d+) = (?:copy \(\*(_\d+)\)|(?:copy |&)\(\(\*(_\d+)\)\.(?:0: bitboard::Bitboard|1: i32|2: square::Square)\));", ins):
                checked(m[1] not in capability and (m[2] or m[3]) in elements, f"unsupported record read {ins}")
                continue
            if m := re.fullmatch(r"(_\d+) = (?:Add|Sub)(?:WithOverflow)?\(copy \(\(\*(_\d+)\)\.1: i32\), (?:move|copy) (_\d+)\);", ins):
                checked(used == {m[2]} and m[2] in elements, f"ordering arithmetic alias {ins}")
                continue
            if m := re.fullmatch(r'assert\(!move \(_\d+\.1: bool\), "attempt to compute [^"]+which would overflow", copy \(\(\*(_\d+)\)\.1: i32\), (?:move|copy) _\d+\) -> \[success: bb\d+, unwind: bb\d+\];', ins):
                checked(used == {m[1]} and m[1] in elements, f"unexpected assertion payload {ins}")
                continue
            raise ValueError(f"unknown use of move storage: {ins}")


def verify_evaluators(directory):
    for function in ("evaluate_moves", "evaluate_moves_fast", "evaluate_moves_by_search"):
        path = directory / f"reversi_core.move_list-ordering-{{impl#0}}-{function}.SimplifyCfg-initial.after.mir"
        verify_value_writes(path.read_text(encoding="utf-8"))


def sequence(path, kind, args, length=0, greater=False, returns="unit"):
    """Follow a normal return, recording permutations applied to the live array."""
    from extract_single import GOTO

    cfg = blocks(path.read_text(encoding="utf-8"))
    pcs = set()
    pc = 0
    env = dict(zip(("_1", "_2", "_3", "_4"), args))
    effects = []

    def value(op):
        if m := re.fullmatch(r"const (\d+)_usize", op):
            return int(m[1])
        if op in ("const true", "const false"):
            return op == "const true"
        if op == "const ()":
            return "unit"
        if op == "copy ((*_1).1: usize)":
            checked(env["_1"] == "array", "length of different array")
            return length
        if op == "&mut ((*_1).0: move_list::move_array::MoveArray)":
            checked(env["_1"] == "list", "different MoveList")
            return "array"
        if op == "&mut ((*_1).0: [std::mem::MaybeUninit<move_list::Move>; 35])":
            checked(env["_1"] == "array", "different MoveArray")
            return "data"
        if m := re.fullmatch(r"move (_\d+) as &mut \[std::mem::MaybeUninit<move_list::Move>\] \(PointerCoercion\(Unsize, Implicit\)\)", op):
            checked(env[m[1]] == "data", "unexpected slice base")
            return "capacitySlice"
        if op == "copy ((*_2).1: i32)":
            checked(env["_2"] == "move", "key of different object")
            return "value"
        if op == "&(*((*_1).0: &impl Fn(&Move) -> i32))":
            checked(env["_1"] == "reverseKey", "invalid key capture")
            return "key"
        if m := re.fullmatch(r"Reverse::<i32>\(move (_\d+)\)", op):
            checked(env[m[1]] == "value", "invalid key ordering")
            return "reversed"
        if m := re.fullmatch(r"(?:move |copy |&(?:mut )?)(_\d+)|&(?:mut )?\(\*(_\d+)\)", op):
            return env[m[1] or m[2]]
        if m := re.fullmatch(r"\(move (_\d+),\)", op):
            return ("tuple", env[m[1]])
        if re.fullmatch(r"\{closure@[^}]+\}", op):
            return "key"
        if m := re.fullmatch(r"\{closure@[^}]+\} \{ key: move (_\d+) \}", op):
            checked(env[m[1]] == "key", "unknown closure for unstable sort")
            return "reverseKey"
        if m := re.fullmatch(r"std::ops::RangeFrom::<usize> \{ start: move (_\d+) \}", op):
            return ("range", env[m[1]])
        if m := re.fullmatch(r"Lt\((.+), (.+)\)", op):
            l, r = value(m[1]), value(m[2])
            if type(l) is int and type(r) is int:
                return l < r
            checked(l == ("keyValue", args[1]) and r == ("keyValue", args[2]), "unknown comparison operands")
            return greater
        raise ValueError(f"{kind} unsupported reorder expression {op}")

    while pc not in pcs:
        pcs.add(pc)
        *statements, end = cfg[pc]
        for instruction in statements:
            if instruction == "PlaceMention(((*_1).1: usize));":
                continue
            if re.fullmatch(r"(?:FakeRead\(ForLet\(None\), _\d+\)|Storage(?:Live|Dead)\(_\d+\));", instruction):
                continue
            m = re.fullmatch(r"(_\d+) = (.+);", instruction)
            checked(m, f"unsupported {kind} statement {instruction}")
            env[m[1]] = value(m[2])
        if m := re.fullmatch(CALL, end):
            fn = m[2]
            values = [value(v) for v in m[3].split(", ") if v]
            if re.fullmatch(r"MoveArray::sort_by_key_desc::<\{closure@[^}]+\}>", fn):
                checked(values == ["array", "key"], "wrong sort receiver")
                effects.append("sortArray")
                result = "unit"
            elif fn == "MoveArray::as_mut_slice":
                checked(values == ["array"], "wrong slice")
                result = "wholeSlice"
            elif fn == "MoveArray::as_mut_ptr":
                checked(values == ["array"], "wrong pointer base")
                result = "base"
            elif fn == "MoveArray::len":
                checked(values == ["array"], "unexpected initialized length")
                result = "length"
            elif fn == "std::slice::from_raw_parts_mut::<'_, move_list::Move>":
                checked(values == ["base", "length"], "initialized slice differs")
                result = "wholeSlice"
            elif fn == "core::slice::<impl [MaybeUninit<move_list::Move>]>::as_mut_ptr":
                checked(values == ["capacitySlice"], "slice pointer from different storage")
                result = "rawData"
            elif fn == "std::ptr::mut_ptr::<impl *mut MaybeUninit<move_list::Move>>::cast::<move_list::Move>":
                checked(values == ["rawData"], "unexpected element pointer cast")
                result = "base"
            elif fn == "<MoveArray as DerefMut>::deref_mut":
                checked(values == ["array"], "unexpected move array")
                result = "wholeSlice"
            elif fn == "core::slice::<impl [move_list::Move]>::iter_mut":
                checked(values == ["wholeSlice"], "iterator into different slice")
                result = "iterator"
            elif fn == "std::ptr::mut_ptr::<impl *mut move_list::Move>::add":
                base, index = values
                checked(base == "base" and 0 <= index < length, "candidate pointer outside initialized prefix")
                result = ("ptr", index)
            elif fn == "<impl Fn(&Move) -> i32 as Fn<(&move_list::Move,)>>::call":
                checked(values[0] == "key" and values[1][0] == "tuple", "wrong key")
                item = values[1][1]
                if item == "move":
                    result = "value"
                else:
                    checked(item[0] == "ptr", "key received different record")
                    result = ("keyValue", item[1])
            elif fn == "std::ptr::swap::<move_list::Move>":
                a, b = values
                checked(a[0] == b[0] == "ptr", "swap does not refer to array")
                effects.append(("swap", a[1], b[1]))
                result = "unit"
            elif fn == "MoveArray::compare_swap_unchecked::<&impl Fn(&Move) -> i32>":
                checked(values[0] == "array" and values[3] == "key", "invalid compare-swap")
                effects.append(("compare", values[1], values[2]))
                result = "unit"
            elif re.fullmatch(r"core::slice::<impl \[move_list::Move\]>::sort_unstable_by_key::<Reverse<i32>, \{closure@[^}]+\}>", fn):
                checked(values == ["wholeSlice", "reverseKey"], "unstable sort of different slice")
                effects.append("unstableSort")
                result = "unit"
            elif fn == "<[move_list::Move] as IndexMut<std::ops::RangeFrom<usize>>>::index_mut":
                checked(values == ["wholeSlice", ("range", "idx")], "unexpected postponed suffix")
                result = "suffix"
            elif fn == "core::slice::<impl [move_list::Move]>::rotate_left":
                checked(values == ["suffix", 1], "unexpected rotation")
                effects.append("postpone")
                result = "unit"
            else:
                raise ValueError(f"{kind} unsupported {fn}")
            env[m[1]] = result
            pc = int(m[4])
        elif m := re.fullmatch(r"switchInt\((.*)\) -> \[((?:\d+: bb\d+, )*otherwise: bb\d+)\];", end):
            branches = dict(re.findall(r"(\w+): bb(\d+)", m[2]))
            key = value(m[1])
            if key == "many":
                checked(set(branches) <= {"0", "1", "2", "3", "otherwise"}, "unmodeled sorting length branch")
            pc = int(branches.get(str(int(key)) if isinstance(key, int) else str(key), branches["otherwise"]))
        elif m := re.fullmatch(GOTO, end):
            pc = int(m[1] or m[2])
        elif m := re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];", end):
            checked(env[m[1]] == "key", "unexpected destructor")
            pc = int(m[2])
        elif end == "return;":
            checked(env["_0"] == returns, f"unexpected {kind} result")
            return effects
        else:
            raise ValueError(f"unexpected reorder terminator {end}")
    raise ValueError("unsupported reorder loop")


def verify_reordering(directory):
    def mir(name):
        return directory / f"reversi_core.move_list-{name}.SimplifyCfg-initial.after.mir"

    accessors = [
        ("{impl#2}-iter_mut", ["list"], "iterator"),
        ("move_array-{impl#4}-deref_mut", ["array"], "wholeSlice"),
        ("move_array-{impl#0}-len", ["array"], "length"),
        ("move_array-{impl#0}-as_mut_ptr", ["array"], "base"),
        ("move_array-{impl#0}-as_mut_slice", ["array"], "wholeSlice"),
        ("ordering-{impl#0}-sort-{closure#0}", ["key", "move"], "value"),
        ("move_array-{impl#0}-sort_by_key_desc-{closure#0}", ["reverseKey", "move"], "reversed"),
    ]
    for name, args, returns in accessors:
        checked(not sequence(mir(name), name, args, length="length", returns=returns), "accessor side effects")
    checked(sequence(mir("ordering-{impl#0}-sort"), "sort", ["list"]) == ["sortArray"], "wrong sorting dispatch")
    checked(sequence(mir("{impl#2}-postpone_move"), "postpone", ["list", "idx"]) == ["postpone"], "wrong rotation")
    sort = mir("move_array-{impl#0}-sort_by_key_desc")
    swap = mir("move_array-{impl#0}-compare_swap_unchecked")
    for length in (0, 1, 2, 3, "many"):
        ops = sequence(sort, "sortArray", ["array", "key"], length)
        expected = {0: [], 1: [], 2: [("compare", 0, 1)], 3: [("compare", 0, 1), ("compare", 1, 2),
                    ("compare", 0, 1)], "many": ["unstableSort"]}[length]
        checked(ops == expected, "sort operations changed")
        for op in ops:
            if isinstance(op, tuple):
                _, a, b = op
                for comparison in (False, True):
                    changed = sequence(swap, "swap", ["array", a, b, "key"], length, comparison)
                    checked(changed == ([("swap", a, b)] if comparison else []), "invalid conditional swap")


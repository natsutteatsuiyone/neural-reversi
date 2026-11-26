"""Translate the normal Rust MIR branches of the two-slot lookup and its seqlock reader."""

import argparse
from pathlib import Path
import re

from mir_common import PASS, Block, parse_blocks as blocks, checked, normal_target, split_call
from extract_tt_reader_mir import layout, read_method


def dependencies(dump, repo):
    fields = layout(repo)
    src = (repo / "crates/reversi-core/src/transposition_table.rs").read_text(encoding="utf-8")
    src = re.sub(r"//[^\n]*", "", src)
    checked(re.search(r"const CLUSTER_SIZE: usize = 2;", src), "cluster size")
    checked(re.search(r"pub struct TranspositionTable\s*\{\s*entries: AlignedBuffer<TTEntry>,\s*cluster_count: u64,", src),
            "table layout")
    path = dump / f"reversi_core.transposition_table-same_board_bits.{PASS}.after.mir"
    compare = blocks(path, "transposition_table::same_board_bits", "fn same_board_bits(_1: u64, _2: u64, _3: u64, _4: u64) -> bool {")
    checked(compare == {0: Block(["_7 = copy _1;", "_8 = copy _3;", "_6 = BitXor(move _7, move _8);",
                                  "_10 = copy _2;", "_11 = copy _4;", "_9 = BitXor(move _10, move _11);",
                                  "_5 = BitOr(move _6, move _9);", "_0 = Eq(move _5, const 0_u64);"],
                                 "return;", False)}, "same_board_bits")
    occupied = read_method(dump, "is_occupied", "(_1: &TTEntryData) -> bool")
    checked(occupied == {0: Block(["_2 = copy ((*_1).2: u8);", "_0 = Ne(move _2, const 0_u8);"],
                                  "return;", False)}, "is_occupied")
    unpack = read_method(dump, "from_u64", "(_1: u64) -> TTEntryData")
    checked(unpack == {
        0: Block(["_2 = copy _1;"], "_0 = std::intrinsics::transmute::<u64, TTEntryData>(move _2) -> [return: bb1, unwind: bb2];", False),
        1: Block([], "return;", False), 2: Block([], "resume;", True)}, "from_u64")
    index = read_method(dump, "get_cluster_idx", "(_1: &TranspositionTable, _2: u64) -> usize")
    checked(index == {
        0: Block(["_5 = copy _2;", "_6 = copy ((*_1).1: u64);"], "_4 = mul_hi64(move _5, move _6) -> [return: bb1, unwind: bb3];", False),
        1: Block(["_3 = move _4 as usize (IntToInt);", "_7 = MulWithOverflow(copy _3, const transposition_table::CLUSTER_SIZE);"],
                 'assert(!move (_7.1: bool), "attempt to compute `{} * {}`, which would overflow", move _3, const transposition_table::CLUSTER_SIZE) -> [success: bb2, unwind: bb3];', False),
        2: Block(["_0 = move (_7.0: usize);"], "return;", False),
        3: Block([], "resume;", True)}, "get_cluster_idx")
    return fields


def lower(body, fields, env, atomic, stop=None):
    visited = set()

    def walk(bb, env, actions, ancestors, indent):
        checked(bb not in ancestors, f"unexpected cycle at bb{bb}")
        visited.add(bb)
        ancestors = ancestors | {bb}
        block = body[bb]
        if stop and bb == stop[0]:
            return [indent + "(" + ", ".join(env[r] for r in stop[1]) + ")"]
        for statement in block.statements:
            if re.fullmatch(r"PlaceMention\(_\d+\);", statement):
                continue
            lhs, rhs = statement.removesuffix(";").split(" = ", 1)
            if m := re.fullmatch(r"&\(\(\*_1\)\.(\d+): std::sync::atomic::Atomic<u64>\)", rhs):
                checked(atomic, "read on non-entry")
                value = fields[int(m[1])]
            elif rhs == "&((*_1).0: util::aligned_buffer::AlignedBuffer<transposition_table::TTEntry>)":
                checked(not atomic, "entries on non-table")
                value = "entries"
            elif m := re.fullmatch(r"std::sync::atomic::Ordering::(Acquire|Relaxed)", rhs):
                value = "." + m[1].lower()
            elif m := re.fullmatch(r"(?:copy |move |&\(\*|&)(_\d+)\)?", rhs):
                value = env[m[1]]
            elif m := re.fullmatch(r"BitAnd\(move (_\d+), const 1_u64\)", rhs):
                value = f"({env[m[1]]} &&& 1)"
            elif m := re.fullmatch(r"(Eq|Ne)\(move (_\d+), (?:const 0_u64|move (_\d+))\)", rhs):
                op = "=" if m[1] == "Eq" else "≠"
                value = f"{env[m[2]]} {op} {env[m[3]] if m[3] else '0'}"
            elif rhs == "Option::<TTEntryData>::None":
                value = "none"
            elif m := re.fullmatch(r"discriminant\((_\d+)\)", rhs):
                value = ("discriminant", m[1])
            elif m := re.fullmatch(r"copy \(\((_\d+) as Some\)\.0: transposition_table::TTEntryData\)", rhs):
                value = env[(m[1], "some")]
            elif m := re.fullmatch(r"Option::<TTEntryData>::Some\(move (_\d+)\)", rhs):
                value = f"some {env[m[1]]}"
            elif m := re.fullmatch(r"AddWithOverflow\(copy (_\d+), const (\d+)_usize\)", rhs):
                value = ("add", env[m[1]], int(m[2]))
            elif m := re.fullmatch(r"move \((_\d+)\.0: usize\)", rhs):
                kind, x, y = env[m[1]]
                checked(kind == "add", "projected non-addition")
                value = f"({x} + {y})"
            else:
                raise ValueError(f"bb{bb}: unsupported {statement}")
            env[lhs] = value

        term = block.end
        if m := split_call(term):
            dest, name, args, nxt, cleanup = m.groups()
            args = [env[re.fullmatch(r"move (_\d+)", x)[1]] for x in args.split(", ")]
            if name == "Atomic::<u64>::load":
                loc, mode = args
                env[dest] = f"(w {len(actions)})"
                actions = actions + [f".load {loc} {mode}"]
            elif name == "fence":
                mode, = args
                actions = actions + [f".fence {mode}"]
            elif name == "same_board_bits":
                env[dest] = "sameBoardBits " + " ".join(args)
            elif name == "TTEntryData::from_u64":
                env[dest] = f"(Packing.unpack endian {args[0]})"
            elif name == "TTEntryData::is_occupied":
                env[dest] = args[0] + ".bound ≠ 0"
            elif name == "core::bool::<impl bool>::then_some::<TTEntryData>":
                guard, value = args
                env[dest] = f"(if {guard} then some {value} else none)"
            elif name == "TranspositionTable::get_cluster_idx":
                receiver, key = args
                checked(receiver == "table", "cluster receiver")
                env[dest] = f"(Clear.getClusterIdx {key} clusters)"
            elif name in ("Board::player", "Board::opponent"):
                checked(args == ["board"], "board receiver")
                env[dest] = name.split("::")[1]
            elif name == "Bitboard::bits":
                env[dest], = args
            elif name == "<AlignedBuffer<TTEntry> as Deref>::deref":
                checked(args == ["entries"], "storage receiver")
                env[dest] = "slice"
            elif name == "core::slice::<impl [TTEntry]>::get_unchecked::<usize>":
                receiver, idx = args
                checked(receiver == "slice", "indexed array")
                env[dest] = idx
            elif name == "TTEntry::read_for_lookup":
                env[dest] = "read " + " ".join(args)
            else:
                raise ValueError(f"bb{bb}: unsupported call {name}")
            return walk(normal_target(body, nxt, cleanup), env, actions, ancestors, indent)
        if m := re.fullmatch(r"goto -> bb(\d+);|falseEdge -> \[real: bb(\d+), imaginary: bb\d+\];", term):
            return walk(normal_target(body, m[1] or m[2]), env, actions, ancestors, indent)
        if m := re.fullmatch(r'assert\(!move \((_\d+)\.1: bool\), "attempt to compute `\{\} \+ \{\}`, which would overflow", move (_\d+), const (\d+)_usize\) -> \[success: bb(\d+), unwind: bb(\d+)\];', term):
            checked(env[m[1]] == ("add", env[m[2]], int(m[3])), "index overflow operands")
            return walk(normal_target(body, m[4], m[5]), env, actions, ancestors, indent)
        if m := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[([01]): bb(\d+), otherwise: bb(\d+)\];", term):
            cond = env[m[1]]
            yes, no = (m[4], m[3]) if m[2] == "0" else (m[3], m[4])
            with_hit = env.copy()
            if isinstance(cond, tuple):
                checked(cond[0] == "discriminant" and m[2] == "1", "Option discriminant")
                with_hit[(cond[1], "some")] = "hit"
                cond = f"let some hit := {env[cond[1]]}"
            else:
                checked(m[2] == "0", "Bool branch")
            return ([indent + "if " + cond + " then"]
                    + walk(normal_target(body, yes), with_hit, actions, ancestors, indent + "  ")
                    + [indent + "else"]
                    + walk(normal_target(body, no), env.copy(), actions, ancestors, indent + "  "))
        checked(term == "return;", f"unknown terminator {term}")
        result = env["_0"]
        if atomic:
            return [indent + "⟨[" + ", ".join(actions) + "], " + result + "⟩"]
        return [indent + result]

    output = walk(0, env, [], set(), "  ")
    if not stop:
        checked(all(i in visited or b == Block([], "resume;", True) for i, b in body.items()), "untranslated block")
    else:
        checked(visited == set(range(stop[0] + 1)), "untranslated probe receiver computation")
    return output


def probe_receivers(dump, repo, fields):
    body = read_method(dump, "probe", "(_1: &TranspositionTable, _2: &Board, _3: u64) -> TTProbeResult<'_>")
    src = (repo / "crates/reversi-core/src/transposition_table.rs").read_text(encoding="utf-8")
    limit = re.search(r"^\s*const PROBE_RETRIES: u32 = (\d+);$", src, re.MULTILINE)
    checked(limit, "probe retry limit")
    setup = body[10].statements
    pair = re.fullmatch(r"_15 = \(move (_\d+), move (_\d+)\);", setup[0])
    checked(pair and setup[1:] == [
        "PlaceMention(_15);", "_13 = copy (_15.0: &transposition_table::TTEntry);",
        "_14 = copy (_15.1: &transposition_table::TTEntry);", "_28 = Option::<&TTEntry>::None;",
        "_31 = std::ops::Range::<u32> { start: const 0_u32, end: const transposition_table::TranspositionTable::PROBE_RETRIES };"],
        "probe receiver setup")
    checked(body[10].end == "_30 = <std::ops::Range<u32> as IntoIterator>::into_iter(move _31) -> [return: bb11, unwind: bb65];",
            "probe retry range")
    refs = [stmt for i, block in body.items() if i != 10 for stmt in block.statements
            if re.match(r"_(?:13|14|15) =", stmt)]
    checked(not refs, "mutable slot aliases in probe")
    loads = [(i, block) for i, block in body.items() if "TTEntry::try_load_snapshot(" in block.end]
    checked([i for i, _ in loads] == [17, 30], "probe snapshot call sites")
    checked(body[17].statements == ["_40 = const false;", "_41 = Option::<TTEntryData>::None;", "_44 = &(*_13);"]
            and body[17].end == "_43 = TTEntry::try_load_snapshot(move _44) -> [return: bb19, unwind: bb65];",
            "first probe read")
    checked(body[30].statements == ["_70 = &(*_14);"]
            and body[30].end == "_69 = TTEntry::try_load_snapshot(move _70) -> [return: bb31, unwind: bb65];",
            "second probe read")
    for i, statements, end in [
        (19, ["PlaceMention(_43);", "_45 = discriminant(_43);"], "switchInt(move _45) -> [0: bb22, 1: bb21, otherwise: bb20];"),
        (20, ["FakeRead(ForMatchedPlace(None), _43);"], "unreachable;"),
        (21, [], "falseEdge -> [real: bb23, imaginary: bb22];"),
        (22, ["_40 = const true;"], "goto -> bb30;"),
        (23, [f"_{46+i} = copy (((_43 as Some).0: (u64, u64, u64, transposition_table::TTEntryData)).{i}: {ty});"
              for i, ty in enumerate(["u64", "u64", "u64", "transposition_table::TTEntryData"])] + ["_52 = &_49;"],
         "_51 = TTEntryData::is_occupied(move _52) -> [return: bb24, unwind: bb65];"),
        (24, [], "switchInt(move _51) -> [0: bb26, otherwise: bb25];"),
        (25, ["_58 = copy _46;", "_59 = copy _47;", "_60 = copy _7;", "_61 = copy _10;"],
         "_57 = same_board_bits(move _58, move _59, move _60, move _61) -> [return: bb27, unwind: bb65];"),
        (26, ["_55 = &(*_13);", "_54 = TTSlot::<'_>(move _55);", "_0 = TTProbeResult::<'_>::Miss { slot: move _54 };"], "goto -> bb62;"),
        (27, [], "switchInt(move _57) -> [0: bb29, otherwise: bb28];"),
        (28, ["_63 = copy _49;", "_65 = &(*_13);", "_64 = TTSlot::<'_>(move _65);",
              "_0 = TTProbeResult::<'_>::Hit { data: move _63, slot: move _64 };"], "goto -> bb62;"),
        (29, ["_67 = copy _49;", "_66 = Option::<TTEntryData>::Some(move _67);", "_41 = move _66;"], "goto -> bb30;"),
        (62, [], "goto -> bb63;"), (63, [], "goto -> bb64;"), (64, [], "return;"),
    ]:
        checked(body[i] == Block(statements, end, False), f"probe second-slot selection bb{i}")
    stop = (10, pair.groups())
    env = {"_1": "table", "_2": "board", "_3": "key"}
    locations = lower(body, fields, env.copy(), False, stop)
    checked(lower(body, fields, env.copy(), False, (10, ("_7", "_10"))) == ["  (player, opponent)"], "query bitboards")
    return locations, int(limit[1])


def render(dump, repo):
    fields = dependencies(dump, repo)
    slot = read_method(dump, "read_for_lookup", "(_1: &TTEntry, _2: u64, _3: u64) -> Option<TTEntryData>")
    cluster = read_method(dump, "lookup", "(_1: &TranspositionTable, _2: &Board, _3: u64) -> Option<TTEntryData>")
    locations, limit = probe_receivers(dump, repo, fields)
    return "\n".join([
        "import NeuralReversi.TranspositionTable.ReaderMir.Syntax", "import NeuralReversi.TranspositionTable.Clear",
        "import NeuralReversi.TranspositionTable.ProbeRetry", "",
        "namespace NeuralReversi.TranspositionTable.ReaderMir", "", "open SeqLock (Word)", "",
        "def lookupRun (endian : Bool) (player opponent : Word) (w : Nat → Word) : List AtomicAction × Option EntryData :=",
        *lower(slot, fields, {"_2": "player", "_3": "opponent"}, True), "",
        "def lookupCluster (clusters key player opponent : Word) (read : Nat → Word → Word → Option EntryData) : Option EntryData :=",
        *lower(cluster, fields, {"_1": "table", "_2": "board", "_3": "key"}, False), "",
        "def probeSlots (clusters key : Word) : Nat × Nat :=", *locations, "",
        "def probeSecond (query : ProbeRetry.Key) : Option (ProbeRetry.Key × EntryData) → Bool",
        "  | none => true",
        "  | some (stored,data) => data.bound != 0 && !sameBoardBits stored.1 stored.2 query.1 query.2", "",
        f"def probeRetries : Nat := {limit}", "",
        "end NeuralReversi.TranspositionTable.ReaderMir", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mir", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    program = render(args.mir, args.repo)
    if args.check:
        checked(args.check.read_text(encoding="utf-8") == program, f"generated lookup differs from {args.check}")
    else:
        print(program, end="")

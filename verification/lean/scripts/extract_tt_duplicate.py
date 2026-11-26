"""Follow the successful allocation, entry copy and return in TranspositionTable::duplicate."""

import re

from mir_common import CALL, checked
from extract_tt_reader_mir import layout
from extract_tt_writer import read_cfg


def check_copy(cfg):
    registers = {}
    calls = {
        "AlignedBuffer::<TTEntry>::len": (("entries",), "len"),
        "AlignedBuffer::<TTEntry>::zeroed": (("len",), "dest"),
        "AlignedBuffer::<TTEntry>::as_ptr": (("entries",), "source.ptr"),
        "AlignedBuffer::<TTEntry>::as_mut_slice": (("dest",), "dest.slice"),
        "core::slice::<impl [TTEntry]>::as_mut_ptr": (("dest.slice",), "dest.ptr"),
        "std::ptr::copy_nonoverlapping::<TTEntry>": (("source.ptr", "dest.ptr", "len"), "copied"),
        "Atomic::<u8>::load": (("generation", "relaxed"), "observed"),
        "Atomic::<u8>::new": (("observed",), "new.generation"),
    }
    fields = {
        "&((*_1).0: util::aligned_buffer::AlignedBuffer<transposition_table::TTEntry>)": "entries",
        "copy ((*_1).1: u64)": "clusters",
        "&((*_1).2: std::sync::atomic::Atomic<u8>)": "generation",
        "std::sync::atomic::Ordering::Relaxed": "relaxed",
        "const ()": "()",
    }
    operations = []
    visited = set()
    pc = 0
    while pc not in visited:
        visited.add(pc)
        *assignments, term = cfg[pc]
        for line in assignments:
            dest, expr = line.rstrip(";").split(" = ", 1)
            if expr in fields:
                registers[dest] = fields[expr]
            elif m := re.fullmatch(r"(?:copy |move |&mut )(_\d+)|&mut \(\*(_\d+)\)", expr):
                registers[dest] = registers[m[1] or m[2]]
            elif m := re.fullmatch(r"TranspositionTable \{ entries: move (_\d+), cluster_count: move (_\d+), generation: move (_\d+) \}", expr):
                checked(tuple(registers[arg] for arg in m.groups()) == ("dest", "clusters", "new.generation"),
                        "duplicate returned fields")
                registers[dest] = "result"
            else:
                raise ValueError(f"unexpected duplicate statement {line}")
        if term == "return;":
            checked(registers.get("_0") == "result" and operations == list(calls),
                    "duplicate allocation, copy, generation or return")
            checked(visited == set(cfg), "unvisited normal duplicate block")
            return
        if m := re.fullmatch(CALL, term):
            dest, name, args, pc = m.groups()
            checked(name in calls, f"unexpected duplicate call {name}")
            actual = tuple(registers[arg.removeprefix("move ")] for arg in args.split(", "))
            expected, registers[dest] = calls[name]
            checked(actual == expected, f"duplicate {name} operands")
            operations.append(name)
        elif m := re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];", term):
            checked(registers[m[1]] == "dest" and registers.get("_0") == "result", "duplicate drop")
            pc = m[2]
        else:
            raise ValueError(f"unexpected duplicate terminator {term}")
        pc = int(pc)
    raise ValueError("cycle in duplicate")


def zeroed_return(cfg):
    # The prefix computes the entry count. Every successful return must use
    # the very buffer allocated at that count, with no intervening entry store.
    expected = {
        8: ["_14 = move (_16.0: usize);", "_18 = copy _14;",
            "_17 = AlignedBuffer::<TTEntry>::zeroed(move _18) -> [return: bb9, unwind: bb13];"],
        9: ["_20 = copy _2;", "_19 = move _20 as u64 (IntToInt);",
            "_21 = Atomic::<u8>::new(const 0_u8) -> [return: bb10, unwind: bb12];"],
        10: ["_0 = TranspositionTable { entries: move _17, cluster_count: move _19, generation: move _21 };",
             "drop(_17) -> [return: bb11, unwind: bb13];"],
        11: ["return;"],
    }
    checked(set(cfg) == set(range(12)) and all(cfg[bb] == block for bb, block in expected.items()),
            "new table did not return its zeroed buffer")
    for bb in range(8):
        term = cfg[bb][-1]
        targets = [int(n) for n in re.findall(r"bb(\d+)", term)]
        checked(targets and not any(n in (9, 10, 11) for n in targets), "new bypassed zeroed allocation")


def clear_writes(cfg, locations):
    # The body below is run for each Some returned by iter_mut on this table.
    header = {
        0: ["_6 = &mut ((*_1).0: util::aligned_buffer::AlignedBuffer<transposition_table::TTEntry>);",
            "_5 = <AlignedBuffer<TTEntry> as DerefMut>::deref_mut(move _6) -> [return: bb1, unwind: bb15];"],
        1: ["_4 = &mut (*_5);", "_3 = core::slice::<impl [TTEntry]>::iter_mut(move _4) -> [return: bb2, unwind: bb15];"],
        2: ["_2 = <std::slice::IterMut<'_, TTEntry> as IntoIterator>::into_iter(move _3) -> [return: bb3, unwind: bb15];"],
        3: ["_7 = move _2;", "goto -> bb4;"],
        4: ["falseUnwind -> [real: bb5, unwind: bb15];"],
        5: ["_12 = &mut _7;", "_11 = &mut (*_12);",
            "_10 = <std::slice::IterMut<'_, TTEntry> as Iterator>::next(move _11) -> [return: bb6, unwind: bb15];"],
        6: ["_13 = discriminant(_10);", "switchInt(move _13) -> [0: bb8, 1: bb9, otherwise: bb7];"],
        7: ["unreachable;"],
        8: ["falseEdge -> [real: bb10, imaginary: bb9];"],
        10: ["_0 = const ();", "return;"],
    }
    checked(all(cfg.get(bb) == lines for bb, lines in header.items()), "exclusive clear iterator")
    pc = 9
    visited = set(header)
    pending = None
    writes = []
    while pc not in visited:
        visited.add(pc)
        lines = cfg[pc]
        if pending is None:
            checked(lines[0] == "_15 = move ((_10 as Some).0: &mut transposition_table::TTEntry);",
                    "exclusive clear did not use the iterated entry")
        else:
            dest, field = pending
            store = re.fullmatch(r"\(\*" + dest + r"\) = const (\d+)_u64;", lines[0])
            checked(store, "exclusive clear write through get_mut")
            writes.append((field, int(store[1])))
        if lines[1:] == ["_9 = const ();", "_8 = const ();", "goto -> bb4;"]:
            checked(visited == set(cfg), "unvisited exclusive clear block")
            return writes
        checked(len(lines) == 3, "extra exclusive clear operation")
        target = re.fullmatch(r"(_\d+) = &mut \(\(\*_15\)\.(\d+): std::sync::atomic::Atomic<u64>\);", lines[1])
        call = re.fullmatch(CALL, lines[2])
        checked(target and call and call[2] == "Atomic::<u64>::get_mut" and call[3] == "move " + target[1],
                "exclusive clear target")
        pending = call[1], locations[int(target[2])]
        pc = int(call[4])
    raise ValueError("exclusive clear loop did not finish the entry")


def render(dump, repo):
    check_copy(read_cfg(dump, "{impl#7}", "duplicate", "(_1: &TranspositionTable) -> TranspositionTable"))
    locations = layout(repo)
    zeroed_return(read_cfg(dump, "{impl#7}", "new", "(_1: usize) -> TranspositionTable"))
    clear = clear_writes(read_cfg(dump, "{impl#7}", "clear", "(_1: &mut TranspositionTable) -> ()"), locations)
    stores = ", ".join(f"({loc.strip('()')},{value})" for loc, value in clear)
    offsets = [f"  | {loc.strip('()')} => {index}" for index, loc in enumerate(locations)]
    fields_at = [f"  | {i} => {loc.strip('()')}" for i, loc in enumerate(locations)]
    return """import NeuralReversi.TranspositionTable.Duplicate

namespace NeuralReversi.TranspositionTable.Duplicate

def wordIndex : SeqLock.Location → Fin 4
""" + "\n".join(offsets) + """

def locationAt : Fin 4 → SeqLock.Location
""" + "\n".join(fields_at) + """

def zeroEntries (before : Bytes) (dest len : Nat) : Bytes :=
  fun addr => if dest ≤ addr ∧ addr < dest + len*32 then 0 else before addr

def copyEntries (before : Bytes) (source dest len : Nat) : Bytes :=
  copy (zeroEntries before dest len) source dest (len*32)

def exclusiveStores : List (SeqLock.Location × SeqLock.Word) :=
  [""" + stores + """]

def clearEntry (prior : SeqLock.Location → SeqLock.Word) : SeqLock.Location → SeqLock.Word :=
  exclusiveStores.foldl (fun words (loc,value) => fun l => if l = loc then value else words l) prior

end NeuralReversi.TranspositionTable.Duplicate
"""


def check_mutants(dump, repo):
    def mutated(cfg, bb, old, new):
        changed = [line.replace(old, new) for line in cfg[bb]]
        checked(changed != cfg[bb], f"missing allocation mutation in bb{bb}")
        return {**cfg, bb: changed}

    def rejected(check, cfg):
        try:
            check(cfg)
        except ValueError:
            return
        raise ValueError("accepted incorrect table initialization or copy")

    clone = read_cfg(dump, "{impl#7}", "duplicate", "(_1: &TranspositionTable) -> TranspositionTable")
    rejected(check_copy, mutated(clone, 5, "move _8, move _10", "move _10, move _8"))
    rejected(check_copy, mutated(clone, 5, "_14 = copy _2", "_14 = const 1_usize"))
    new = read_cfg(dump, "{impl#7}", "new", "(_1: usize) -> TranspositionTable")
    rejected(zeroed_return, mutated(new, 7, "success: bb8", "success: bb9"))
    clear = read_cfg(dump, "{impl#7}", "clear", "(_1: &mut TranspositionTable) -> ()")
    locations = layout(repo)
    rejected(lambda cfg: clear_writes(cfg, locations), mutated(clear, 6, "1: bb9", "1: bb8"))
    writes = clear_writes(clear, locations)
    for cfg in (mutated(clear, 12, "0_u64", "1_u64"),
                mutated(clear, 13, "(*_15).2", "(*_15).0")):
        checked(clear_writes(cfg, locations) != writes, "exclusive store mutation lost in translation")
    return 6

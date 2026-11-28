"""Translate a stable seqlock_write and each caller's request into Lean."""

import re

from mir_common import blocks, CALL, GOTO, checked
from extract_tt_reader_mir import layout


SUFFIX = ".SimplifyCfg-initial.after.mir"
SKIP = re.compile(r"(?:Storage(?:Live|Dead)|PlaceMention)\(.+\);|FakeRead\(For(?:Let|MatchedPlace)\(None\), .+\);")


def read_cfg(dump, owner, method, signature):
    path = dump / f"reversi_core.transposition_table-{owner}-{method}{SUFFIX}"
    source = path.read_text(encoding="utf-8")
    header = re.search(r"^fn transposition_table::<impl at [^>]+>::" + method
                       + re.escape(signature) + r" \{$", source, re.M)
    checked(header, f"unexpected signature for {method}")
    return {bb: [line for line in lines if not SKIP.fullmatch(line)]
            for bb, lines in blocks(source).items()}


def validate_pack(dump):
    cfg = read_cfg(dump, "{impl#3}", "to_u64", "(_1: TTEntryData) -> u64")
    checked(cfg == {0: ["_2 = copy _1;",
                        "_0 = std::intrinsics::transmute::<TTEntryData, u64>(move _2) -> [return: bb1, unwind: bb2];"],
                    1: ["return;"]}, "TTEntryData::to_u64")


def stable_write(cfg, locations, success):
    values = {"_1": "slot", "_2": "seq.val", "_3": "player", "_4": "opponent", "_5": "entry"}
    actions = []

    def value(expr):
        if expr in ("const true", "const false"):
            return expr == "const true"
        if expr == "const ()":
            return "()"
        if m := re.fullmatch(r"const (\d+)_u64", expr):
            return m[1]
        if m := re.fullmatch(r"(?:copy|move) (_\d+)", expr):
            return values[m[1]]
        if m := re.fullmatch(r"(?:copy \(\*(_\d+)\)|&(_\d+))", expr):
            return values[m[1] or m[2]]
        if m := re.fullmatch(r"\((move _\d+), (move _\d+)\)", expr):
            return (value(m[1]), value(m[2]))
        if m := re.fullmatch(r"copy \((_\d+)\.([01]): &u64\)", expr):
            return values[m[1]][int(m[2])]
        if m := re.fullmatch(r"&\(\(\*(_\d+)\)\.(\d+): std::sync::atomic::Atomic<u64>\)", expr):
            checked(values[m[1]] == "slot", "wrong atomic object")
            return locations[int(m[2])]
        if m := re.fullmatch(r"std::sync::atomic::Ordering::(Acquire|Release|Relaxed|AcqRel)", expr):
            return "." + m[1][0].lower() + m[1][1:]
        if m := re.fullmatch(r"(BitAnd|BitOr|Eq)\((.*), (.*)\)", expr):
            left, right = value(m[2]), value(m[3])
            if m[1] == "Eq":
                checked((left, right) == ("(seq.val &&& 1)", "0"), "even sequence precondition")
                return True
            op = {"BitAnd": "&&&", "BitOr": "|||"}[m[1]]
            return f"({left} {op} {right})"
        raise ValueError(f"unknown seqlock_write value {expr}")

    pc = 0
    visited = set()
    while pc not in visited:
        visited.add(pc)
        *statements, end = cfg[pc]
        for statement in statements:
            m = re.fullmatch(r"(_\d+) = (.+);", statement)
            checked(m, f"unexpected seqlock_write statement {statement}")
            values[m[1]] = value(m[2])
        if m := re.fullmatch(CALL, end):
            dest, name, args, next_pc = m.groups()
            args = [value(arg) for arg in args.split(", ")]
            if name == "Atomic::<u64>::compare_exchange":
                actions.append(".cas " + " ".join(args))
                result = ("casResult", success)
            elif name == "Result::<u64, u64>::is_err":
                (cas,) = args
                checked(cas[0] == "casResult", "missing CAS result")
                result = not cas[1]
            elif name == "fence":
                (mode,) = args
                actions.append(f".fence {mode}")
                result = "()"
            elif name == "TTEntryData::to_u64":
                (entry,) = args
                result = f"(Packing.pack endian {entry})"
            elif name == "Atomic::<u64>::store":
                actions.append(".store " + " ".join(args))
                result = "()"
            elif name == "core::num::<impl u64>::wrapping_add":
                left, right = args
                result = f"({left} + {right})"
            else:
                raise ValueError(f"unexpected seqlock_write call {name}")
            values[dest] = result
            pc = int(next_pc)
        elif m := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];", end):
            condition = values[m[1]]
            checked(isinstance(condition, bool), "non-Boolean switch")
            pc = int(m[3] if condition else m[2])
        elif m := re.fullmatch(GOTO, end):
            pc = int(m[1] or m[2])
        else:
            checked(end == "return;" and isinstance(values["_0"], bool), f"unexpected exit {end}")
            return "⟨[" + ", ".join(actions) + "], " + str(values["_0"]).lower() + "⟩"
    raise ValueError("unexpected loop in seqlock_write")


def render(dump, repo):
    fields = layout(repo)
    validate_pack(dump)
    body = read_cfg(dump, "{impl#6}", "seqlock_write",
                    "(_1: &TTEntry, _2: u64, _3: u64, _4: u64, _5: TTEntryData) -> bool")
    return "\n".join([
        "module", "public import NeuralReversi.TranspositionTable.WriterMir.Syntax", "",
        "@[expose] public section", "", "namespace NeuralReversi.TranspositionTable.WriterMir", "", "open SeqLock (Word)", "",
        "def writeStable (endian : Bool) (seq : { v : Word // v &&& 1 = 0 })",
        "    (player opponent : Word) (entry : EntryData) (casSuccess : Bool) : List Atom × Bool :=",
        "  if casSuccess then", "    " + stable_write(body, fields, True),
        "  else", "    " + stable_write(body, fields, False), "",
        "end NeuralReversi.TranspositionTable.WriterMir", ""])

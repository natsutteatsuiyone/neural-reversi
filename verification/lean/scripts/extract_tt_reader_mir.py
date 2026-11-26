"""Extract the normal atomic traces and returned tuple of TTEntry::try_load_snapshot."""

import re

from mir_common import (
    run_extractor_cli,
    PASS,
    Block,
    parse_blocks as blocks,
    checked,
    normal_target,
    split_call,
)


def read_method(dump, method, signature):
    paths = list(dump.glob(f"reversi_core.transposition_table-{{impl#*}}-{method}.{PASS}.after.mir"))
    checked(len(paths) == 1, f"expected one MIR body for {method}")
    path = paths[0]
    header = path.read_text(encoding="utf-8").splitlines()[0]
    match = re.fullmatch(r"// MIR for `(transposition_table::<impl at [^`]+>::" + method
                         + r")` after " + PASS, header)
    checked(match, f"unexpected header: {header}")
    return blocks(path, match[1], f"fn {match[1]}{signature} {{")


def layout(repo):
    src = (repo / "crates/reversi-core/src/transposition_table.rs").read_text(encoding="utf-8")
    src = re.sub(r"//[^\n]*", "", src)
    def fields(name, repr_text):
        m = re.search(r"#\[repr\(" + re.escape(repr_text) + r"\)\]\s*(?:#\[derive\([^\]]+\)\]\s*)?"
                      + "pub struct " + name + r"\s*\{([^}]+)\}", src)
        checked(m, f"missing {repr_text} layout of {name}")
        return [x.strip() for x in m[1].split(",") if x.strip()]
    atomic = fields("TTEntry", "C, align(32)")
    locations = {"player": "(.payload .player)", "opponent": "(.payload .opponent)",
                 "data": "(.payload .metadata)", "seq": ".sequence"}
    checked(sorted(atomic) == sorted(k + ": AtomicU64" for k in locations), "atomic fields")
    checked(fields("TTEntryData", "C") == ["score: i16", "best_move: u8", "bound: u8", "depth: u8",
             "selectivity: u8", "generation: u8", "flags: u8"], "TTEntryData unpack layout")
    checked(re.search(r"#\[repr\(u8\)\]\s*pub enum Bound\s*\{\s*None = 0,", src), "Bound::None")
    return [locations[field.split(":")[0]] for field in atomic]


def render(dump, repo):
    fields = layout(repo)
    unpack = read_method(dump, "from_u64", "(_1: u64) -> TTEntryData")
    checked(unpack == {
        0: Block(["_2 = copy _1;"],
                 "_0 = std::intrinsics::transmute::<u64, TTEntryData>(move _2) -> [return: bb1, unwind: bb2];", False),
        1: Block([], "return;", False), 2: Block([], "resume;", True)}, "TTEntryData::from_u64")
    body = read_method(dump, "try_load_snapshot", "(_1: &TTEntry) -> Option<(u64, u64, u64, TTEntryData)>")
    visited = set()
    none = "Option::<(u64, u64, u64, TTEntryData)>::None"
    const = "const transposition_table::Bound::None::{constant#0}"

    def walk(bb, env, actions, ancestors, indent):
        checked(bb not in ancestors, f"cycle at bb{bb}")
        ancestors = ancestors | {bb}
        visited.add(bb)
        block = body[bb]
        for line in block.statements:
            m = re.fullmatch(r"(_\d+) = (.+);", line)
            checked(m, f"unknown assignment {line}")
            dest, expr = m.groups()
            if m := re.fullmatch(r"&\(\(\*_1\)\.(\d+): std::sync::atomic::Atomic<u64>\)", expr):
                value = fields[int(m[1])]
            elif m := re.fullmatch(r"std::sync::atomic::Ordering::(Relaxed|Acquire)", expr):
                value = "." + m[1].lower()
            elif m := re.fullmatch(r"(?:copy|move) (_\d+)", expr):
                value = env[m[1]]
            elif m := re.fullmatch(r"move (_\d+) as u8 \(IntToInt\)", expr):
                checked(env[m[1]] == "0#8", f"bb{bb}: cast operand differs from Bound::None")
                value = env[m[1]]
            elif m := re.fullmatch(r"BitAnd\(move (_\d+), const 1_u64\)", expr):
                value = f"({env[m[1]]} &&& 1)"
            elif m := re.fullmatch(r"(Eq|Ne)\(move (_\d+), (?:const 0_u64|move (_\d+))\)", expr):
                op = "=" if m[1] == "Eq" else "≠"
                value = f"{env[m[2]]} {op} {env[m[3]] if m[3] else '0'}"
            elif m := re.fullmatch(r"copy \((_\d+)\.2: u8\)", expr):
                value = f"(Packing.unpack bigEndian ({env[m[1]]})).bound"
            elif expr == f"AddWithOverflow({const}, const 0_u8)":
                value = ("0#8", False)
            elif m := re.fullmatch(r"move \((_\d+)\.0: u8\)", expr):
                value = env[m[1]][0]
            elif expr == none:
                value = "none"
            elif m := re.fullmatch(r"\(((?:const 0_u64|move _\d+)(?:, (?:const 0_u64|move _\d+)){3})\)", expr):
                value = ", ".join("0" if x == "const 0_u64" else env[x.split()[1]] for x in m[1].split(", "))
            elif m := re.fullmatch(r"Option::<\(u64, u64, u64, TTEntryData\)>::Some\(move (_\d+)\)", expr):
                value = f"some ⟨{env[m[1]]}⟩"
            else:
                raise ValueError(f"bb{bb}: unknown statement {line}")
            env[dest] = value
        t = block.end
        if m := split_call(t):
            dest, name, args, nxt, cleanup = m.groups()
            operands = [env[re.fullmatch(r"move (_\d+)", a)[1]] for a in args.split(", ")]
            if name == "Atomic::<u64>::load":
                loc, mode = operands
                env[dest] = f"word {len(actions)}"
                actions = actions + [f".load {loc} {mode}"]
            elif name == "fence":
                mode, = operands
                actions = actions + [f".fence {mode}"]
            elif name == "TTEntryData::from_u64":
                env[dest], = operands
            else:
                raise ValueError(f"bb{bb}: unknown call {name}")
            return walk(normal_target(body, nxt, cleanup), env, actions, ancestors, indent)
        if m := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];", t):
            return ([indent + "if " + env[m[1]] + " then"]
                    + walk(normal_target(body, m[3]), env.copy(), actions, ancestors, indent + "  ")
                    + [indent + "else"]
                    + walk(normal_target(body, m[2]), env.copy(), actions, ancestors, indent + "  "))
        if m := re.fullmatch(r"assert\(!move \((_\d+)\.1: bool\), " + re.escape(
                f'"attempt to compute `{{}} + {{}}`, which would overflow", {const}, const 0_u8)') +
                r" -> \[success: bb(\d+), unwind: bb(\d+)\];", t):
            checked(env[m[1]][1] is False, "overflowing discriminant expression")
            return walk(normal_target(body, m[2], m[3]), env, actions, ancestors, indent)
        if m := re.fullmatch(r"goto -> bb(\d+);", t):
            return walk(normal_target(body, m[1]), env, actions, ancestors, indent)
        checked(t == "return;", f"bb{bb}: unknown terminator {t}")
        return [indent + "⟨[" + ", ".join(actions) + "], " + env["_0"] + "⟩"]

    branches = walk(0, {}, [], set(), "  ")
    checked(all(bb in visited or block == Block([], "resume;", True) for bb, block in body.items()),
            "unmodeled MIR block")
    return "\n".join(["import NeuralReversi.TranspositionTable.ReaderMir.Syntax", "",
        "namespace NeuralReversi.TranspositionTable.ReaderMir", "", "open SeqLock (Word)", "",
        "def run (bigEndian : Bool) (word : Nat → Word) : Outcome :=", *branches, "",
        "end NeuralReversi.TranspositionTable.ReaderMir", ""])


def main():
    run_extractor_cli(
        __doc__,
        render,
        filters="try_load_snapshot|from_u64",
        check_message=lambda p: f"generated MIR differs from {p}",
    )


if __name__ == "__main__":
    main()

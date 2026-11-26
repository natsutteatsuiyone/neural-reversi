"""Execute every normal branch of one MIR probe iteration, retaining the slot identities."""

import re

from mir_common import Block, PASS, parse_blocks as blocks, checked, normal_target, split_call
from extract_endgame_order_mir import code
from extract_tt_reader_mir import read_method
from extract_tt_lookup_mir import dependencies, probe_receivers


KINDS = ("busy", "empty", "matching", "other")
SIGNATURE = "(_1: &TranspositionTable, _2: &Board, _3: u64) -> TTProbeResult<'_>"


def hint_bodies(dump):
    return [code(dump, name) for name in (
        "transposition_table-{impl#2}-best_move", "transposition_table-{impl#3}-best_move",
        "square-{impl#0}-from_u8", "square-{impl#0}-from_u8_unchecked")]


def check_hint(result, entry, square, cast):
    checked(result == {
        0: ["_2 = discriminant((*_1));", "switchInt(move _2) -> [0: bb2, 1: bb3, otherwise: bb1];"],
        1: ["unreachable;"], 2: ["falseEdge -> [real: bb4, imaginary: bb3];"],
        3: ["_0 = Option::<Square>::None;", "goto -> bb6;"],
        4: ["_3 = &(((*_1) as Hit).0: transposition_table::TTEntryData);", "_4 = &(*_3);",
            "_0 = TTEntryData::best_move(move _4) -> [return: bb5, unwind: bb7];"],
        5: ["goto -> bb6;"], 6: ["return;"]}, "best move of probe hit/miss")
    checked(entry == {0: ["_2 = copy ((*_1).1: u8);",
                          "_0 = Square::from_u8(move _2) -> [return: bb1, unwind: bb2];"],
                      1: ["return;"]}, "best move metadata field")
    checked(square == {
        0: ["_3 = copy _1;", "_2 = Lt(move _3, const 64_u8);", "switchInt(move _2) -> [0: bb3, otherwise: bb1];"],
        1: ["_5 = copy _1;", "_4 = Square::from_u8_unchecked(move _5) -> [return: bb2, unwind: bb5];"],
        2: ["_0 = Option::<Square>::Some(move _4);", "goto -> bb4;"],
        3: ["_0 = Option::<Square>::None;", "goto -> bb4;"], 4: ["return;"]}, "square byte range and payload")
    expected = {
        0: ["_3 = const true;", "switchInt(move _3) -> [0: bb6, otherwise: bb1];"],
        1: ["_6 = copy _1;", "_5 = Lt(move _6, const 64_u8);", "switchInt(move _5) -> [0: bb3, otherwise: bb2];"],
        2: ["_4 = const ();", "_2 = const ();", "goto -> bb7;"], 6: ["_2 = const ();", "goto -> bb7;"],
        7: ["_19 = copy _1;", "_0 = std::intrinsics::transmute::<u8, Square>(move _19) -> [return: bb8, unwind: bb9];"],
        8: ["return;"]}
    checked(all(cast[pc] == ops for pc, ops in expected.items()), "in-range square cast")


def helpers(dump, repo):
    src = (repo / "crates/reversi-core/src/transposition_table.rs").read_text(encoding="utf-8")
    checked(re.search(r"const AGE_WEIGHT: i32 = 8;", src) and re.search(r"const GENERATION_MASK: u8 = 0x7F;", src),
            "probe replacement constants")
    age = read_method(dump, "relative_age", "(_1: &TTEntryData, _2: u8) -> i32")
    score = read_method(dump, "replacement_score", "(_1: &TTEntryData, _2: u8) -> i32")
    signature = "(_1: &TranspositionTable) -> u8 {"
    paths = [(p, p.read_text(encoding="utf-8")) for p in dump.glob(f"reversi_core.transposition_table-*-generation.{PASS}.after.mir")]
    hits = [(p, text) for p, text in paths if signature in text]
    checked(len(hits) == 1, "table generation getter")
    path, body = hits[0]
    name = body.splitlines()[0].split("`")[1]
    generation = blocks(path, name, f"fn {name}{signature}")
    cleanup = Block([], "resume;", True)
    checked(generation == {
        0: Block(["_3 = &((*_1).2: std::sync::atomic::Atomic<u8>);", "_4 = std::sync::atomic::Ordering::Relaxed;"],
                 "_2 = Atomic::<u8>::load(move _3, move _4) -> [return: bb1, unwind: bb2];", False),
        1: Block(["_0 = BitAnd(move _2, const transposition_table::TTEntry::GENERATION_MASK);"], "return;", False),
        2: cleanup}, "probe generation order or mask")
    checked(age == {
        0: Block(["_5 = copy _2;", "_6 = copy ((*_1).5: u8);"],
                 "_4 = core::num::<impl u8>::wrapping_sub(move _5, move _6) -> [return: bb1, unwind: bb2];", False),
        1: Block(["_3 = BitAnd(move _4, const transposition_table::TTEntry::GENERATION_MASK);",
                  "_0 = move _3 as i32 (IntToInt);"], "return;", False), 2: cleanup}, "probe relative age")
    checked(score == {
        0: Block(["_4 = copy ((*_1).3: u8);", "_3 = move _4 as i32 (IntToInt);", "_7 = &(*_1);", "_8 = copy _2;"],
                 "_6 = TTEntryData::relative_age(move _7, move _8) -> [return: bb1, unwind: bb4];", False),
        1: Block(["_9 = MulWithOverflow(copy _6, const transposition_table::TTEntry::AGE_WEIGHT);"],
                 'assert(!move (_9.1: bool), "attempt to compute `{} * {}`, which would overflow", move _6, const transposition_table::TTEntry::AGE_WEIGHT) -> [success: bb2, unwind: bb4];', False),
        2: Block(["_5 = move (_9.0: i32);", "_10 = SubWithOverflow(copy _3, copy _5);"],
                 'assert(!move (_10.1: bool), "attempt to compute `{} - {}`, which would overflow", move _3, move _5) -> [success: bb3, unwind: bb4];', False),
        3: Block(["_0 = move (_10.0: i32);"], "return;", False), 4: cleanup}, "probe replacement score")


def loop(body):
    for i, statements, end in [
        (11, ["PlaceMention(_30);", "_32 = move _30;"], "goto -> bb12;"),
        (12, [], "falseUnwind -> [real: bb13, unwind: bb65];"),
        (13, ["_37 = &mut _32;", "_36 = &mut (*_37);"],
         "_35 = <std::ops::Range<u32> as Iterator>::next(move _36) -> [return: bb14, unwind: bb65];"),
        (14, ["PlaceMention(_35);", "_38 = discriminant(_35);"], "switchInt(move _38) -> [0: bb16, 1: bb17, otherwise: bb15];"),
        (15, ["FakeRead(ForMatchedPlace(None), _35);"], "unreachable;"),
        (16, [], "falseEdge -> [real: bb18, imaginary: bb17];"),
        (18, ["_120 = copy _28;", "_121 = &(*_13);"],
         "_119 = Option::<&TTEntry>::unwrap_or(move _120, move _121) -> [return: bb60, unwind: bb65];"),
        (60, ["_118 = &(*_119);", "_117 = TTSlot::<'_>(move _118);", "_0 = TTProbeResult::<'_>::Miss { slot: move _117 };"],
         "goto -> bb64;"),
        (64, [], "return;"), (65, [], "resume;"),
    ]:
        checked(body[i] == Block(statements, end, i == 65), f"probe loop or exhausted return bb{i}")


def project(expr, env):
    if m := re.fullmatch(r"(?:copy |move |&\(\*|&)(_\d+)\)?", expr):
        return env[m[1]]
    if m := re.fullmatch(r"const (true|false)", expr):
        return m[1] == "true"
    if expr == "Option::<TTEntryData>::None":
        return None
    if m := re.fullmatch(r"(?:Option::<(?:TTEntryData|&TTEntry)>::Some|TTSlot::<'_>)\(move (_\d+)\)", expr):
        return env[m[1]]
    if m := re.fullmatch(r"discriminant\((_\d+)\)", expr):
        return int(env[m[1]] is not None)
    if m := re.fullmatch(r"copy \(\(\((_\d+) as Some\)\.0: \(u64, u64, u64, transposition_table::TTEntryData\)\)\.([0-3]): (?:u64|transposition_table::TTEntryData)\)", expr):
        return env[m[1]][int(m[2])]
    if m := re.fullmatch(r"copy \(\((_\d+) as Some\)\.0: transposition_table::TTEntryData\)", expr):
        return env[m[1]]
    if m := re.fullmatch(r"Lt\(move (_\d+), move (_\d+)\)", expr):
        return env[m[1]] < env[m[2]]
    if m := re.fullmatch(r"TTProbeResult::<'_>::Hit \{ data: move (_\d+), slot: move (_\d+) \}", expr):
        return f".done (.hit {env[m[2]]} {env[m[1]]})"
    if m := re.fullmatch(r"TTProbeResult::<'_>::Miss \{ slot: move (_\d+) \}", expr):
        return f".done (.miss {env[m[1]]})"
    raise ValueError(f"unsupported probe expression: {expr}")


def evaluate(body, kinds, scores):
    names = ["front", "back"]
    env = {"_1": "table", "_13": 0, "_14": 1, "_28": "previous", "_7": "P", "_10": "O"}
    effects = []
    pc = 17
    visited = set()
    while pc != 12:
        checked(pc not in visited, "probe round cycled without reaching range::next")
        visited.add(pc)
        b = body[pc]
        for statement in b.statements:
            if re.fullmatch(r"PlaceMention\(_\d+\);", statement):
                continue
            m = re.fullmatch(r"(_\d+) = (.+);", statement)
            checked(m, f"unexpected probe statement {statement}")
            env[m[1]] = project(m[2], env)
        term = b.end
        if term == "return;":
            return tuple(effects), env["_0"]
        if m := split_call(term):
            dest, name, operands, normal, cleanup = m.groups()
            args = [project(arg, env) for arg in operands.split(", ")] if operands else []
            if name == "TTEntry::try_load_snapshot":
                slot, = args
                effects.append(f".read {slot}")
                key = ("P", "O") if kinds[slot] == "matching" else (f"P{slot}", f"O{slot}")
                value = None if kinds[slot] == "busy" else (*key, f"seq{slot}", names[slot])
            elif name == "TTEntryData::is_occupied":
                value = kinds[names.index(args[0])] != "empty"
            elif name == "same_board_bits":
                p, o, qp, qo = args
                slot = 0 if effects[-1] == ".read 0" else 1
                actual = ("P", "O") if kinds[slot] == "matching" else (f"P{slot}", f"O{slot}")
                checked((p, o) == actual and (qp, qo) == ("P", "O"), "probe board operands")
                value = (p, o) == (qp, qo)
            elif name == "Option::<TTEntryData>::is_some":
                value = args[0] is not None
            elif name == "TranspositionTable::generation":
                checked(args == ["table"], "probe generation receiver")
                effects.append(".generation")
                value = "generation"
            elif name == "TTEntryData::replacement_score":
                entry, generation = args
                checked(generation == "generation", "probe victim generation")
                value = scores[names.index(entry)]
            elif name == "spin_loop":
                effects.append(".spin")
                value = None
            else:
                raise ValueError(f"unsupported probe call {name}")
            env[dest] = value
            pc = normal_target(body, normal, cleanup)
        elif m := re.fullmatch(r"goto -> bb(\d+);|falseEdge -> \[real: bb(\d+), imaginary: bb\d+\];", term):
            pc = normal_target(body, m[1] or m[2])
        elif m := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[(.+)\];", term):
            edges = dict(re.findall(r"(\d+|otherwise): bb(\d+)", m[2]))
            pc = normal_target(body, edges.get(str(int(env[m[1]])), edges["otherwise"]))
        else:
            raise ValueError(f"unsupported probe terminator {term}")
    victim = env["_28"]
    checked(victim in ("previous", 0, 1), "invalid retry slot")
    return tuple(effects), ".retry none" if victim == "previous" else f".retry (some {victim})"


def arm(body, front, back):
    samples = [evaluate(body, (front,back), scores) for scores in [(1,0), (0,0), (0,1)]]
    effects, exit_ = samples[0]
    if not all(sample == samples[0] for sample in samples):
        checked(samples == [(effects, f".done (.miss {slot})") for slot in [1,0,0]], "probe victim comparison or tie")
        exit_ = ".done (.miss (Replacement.victim generation front back))"
    return f"⟨[{','.join(effects)}],{exit_}⟩"


def pattern(kind, name, result):
    if kind in ("matching","other"):
        return f"(.{kind} {name})" if re.search(rf"\b{name}\b", result) else f"(.{kind} _)"
    return "." + kind


def program(body):
    loop(body)
    clauses = []
    for first in KINDS:
        row = [arm(body, first, second) for second in KINDS]
        if len(set(row)) == 1:
            clauses.append(f"  | {pattern(first,'front',row[0])},_ => {row[0]}")
        else:
            for i, result in enumerate(row):
                clauses.append(f"  | {pattern(first,'front',result)},{pattern(KINDS[i],'back',result)} => {result}")
    return "\n".join([
        "import NeuralReversi.TranspositionTable.ReaderMir.ProbeSyntax", "import NeuralReversi.TranspositionTable.Packing", "",
        "namespace NeuralReversi.TranspositionTable.ReaderMir", "",
        "def bestMove : ProbeRetry.Result → Option (Fin 64)",
        "  | .hit _ data => Packing.decodeBestMove data.bestMove", "  | .miss _ => none", "",
        "def probePass (generation : BitVec 8) : ProbeRetry.Observation → ProbeRetry.Observation → Pass",
        *clauses, "", "end NeuralReversi.TranspositionTable.ReaderMir", ""])

def render(dump, repo):
    helpers(dump, repo)
    check_hint(*hint_bodies(dump))
    probe_receivers(dump, repo, dependencies(dump, repo))
    return program(read_method(dump, "probe", SIGNATURE))


def check_mutants(dump):
    body = read_method(dump, "probe", SIGNATURE)
    expected = program(body)
    mutations = [
        (18, "_121 = &(*_13)", "_121 = &(*_14)"),
        (39, "0: bb41", "0: bb40"),
        (40, "_91 = &(*_14)", "_91 = &(*_13)"),
        (45, "_99 = &_75", "_99 = &_94"),
        (47, "Lt(move _98, move _101)", "Lt(move _101, move _98)"),
        (51, "0: bb52", "0: bb53"),
        (55, "_115 = &(*_13)", "_115 = &(*_14)"),
        (58, "_116 = spin_loop() -> [return: bb59, unwind: bb65];", "goto -> bb59;"),
        (59, "goto -> bb12", "goto -> bb18"),
    ]
    for block, old, new in mutations:
        src = body[block]
        lines = src.statements + [src.end]
        checked(sum(line.count(old) for line in lines) == 1, f"missing mutation in bb{block}")
        lines = [line.replace(old, new) for line in lines]
        change = {**body, block: Block(lines[:-1], lines[-1], src.cleanup)}
        try:
            result = program(change)
        except ValueError:
            continue
        checked(result != expected, f"unchanged probe after mutation in bb{block}")
    getters = hint_bodies(dump)
    checks = [(0,0,"0: bb2, 1: bb3","0: bb3, 1: bb2"), (1,0,".1: u8",".4: u8"),
              (2,0,"Lt(move _3","Le(move _3"), (3,7,"copy _1","const 0_u8")]
    for method, pc, old, new in checks:
        original = getters[method][pc]
        changed = [op.replace(old, new) for op in original]
        checked(original != changed, "best move mutation missing")
        bodies = getters.copy()
        bodies[method] = {**getters[method], pc: changed}
        try:
            check_hint(*bodies)
        except ValueError:
            continue
        raise ValueError("accepted best move mutation: " + new)
    return len(mutations) + len(checks)

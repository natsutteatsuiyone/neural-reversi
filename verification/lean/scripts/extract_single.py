"""Extract root PVS on a nonempty, singleton move list whose cached wipeout is absent."""

import itertools
from pathlib import Path
import re
import subprocess
import tempfile

from extract_root_wipeout import constants
from mir_common import ASSERT, CALL, GOTO, blocks, checked


RESET = "RootMoves::reset_scores_from_pv_idx"
UPDATE = "SearchContext::<'_>::update"
CHILD = "search::<PV, SS>"
UNDO = "SearchContext::<'_>::undo"
STOP = "threading::thread::Thread::should_stop"
CHILD_EVENTS = [RESET, UPDATE, CHILD, UNDO, STOP]

SIGNATURE = (
    "fn search(_1: &mut SearchContext<'_>, _2: &Board, _3: u32, _4: Score, "
    "_5: Score, _6: &Arc<threading::thread::Thread>, _7: bool) -> Score {"
)


def run(cfg, case, traits, frame=None):
    claim, above_floor, above_alpha, below_beta, at_max, first_line = case
    events = []
    env = {
        "_1": ("ref", "ctx"), "_2": ("ref", "board"), "_3": "depth",
        "_4": "alpha", "_5": "beta", "_6": ("ref", "arc"), "_7": 0,
        "one_move": "move",
    }
    for name in ("ctx", "board", "arc", "roots", "tt", "stats", "selectivity", "flips", "sq", "thread"):
        env[name] = name
    if frame is not None:
        env.update(frame.inputs)

    def deref(value):
        checked(isinstance(value, tuple) and value[0] == "ref", f"not a reference {value}")
        return value[1]

    def place(text):
        if re.fullmatch(r"_\d+", text):
            return text
        if m := re.fullmatch(r"\(\*(_\d+)\)", text):
            return deref(env[m[1]])
        if m := re.fullmatch(r"\(\*\(\(\*(_\d+)\)\.(\d+): &([^)]+)\)\)", text):
            checked(deref(env[m[1]]) == "ctx", "field of a different context")
            return {
                ("1", "search::root_move::RootMoves"): "roots",
                ("5", "transposition_table::TranspositionTable"): "tt",
            }[(m[2], m[3])]
        if m := re.fullmatch(r"\(\(\*(_\d+)\)\.(\d+): ([^)]+)\)", text):
            checked(deref(env[m[1]]) == "ctx", "field of a different context")
            return {
                ("12", "search::counters::SearchStats"): "stats",
                ("7", "probcut::Selectivity"): "selectivity",
            }[(m[2], m[3])]
        if m := re.fullmatch(r"\((_\d+)\.(\d+): ([^)]+)\)", text):
            checked(env[m[1]] == "move", f"field of an unknown move {env[m[1]]}")
            return {("0", "bitboard::Bitboard"): "flips", ("2", "square::Square"): "sq"}[(m[2], m[3])]
        raise ValueError(f"unknown place {text}")

    def operand(text):
        if text in traits:
            return traits[text]
        if m := re.fullmatch(r"const (\d+)_(?:usize|u32|i32)", text):
            return int(m[1])
        if m := re.fullmatch(r"copy \(\((_\d+) as Some\)\.0: square::Square\)", text):
            value = env[m[1]]
            checked(isinstance(value, tuple) and value[0] == "some", "wrong cached square")
            return value[1]
        if m := re.fullmatch(r"(?:move|copy) \((_\d+)\.([01]): (usize|u32|bool)\)", text):
            value = env[m[1]]
            checked(isinstance(value, tuple) and value[0] == "checked", "expected checked arithmetic")
            return value[int(m[2]) + 1]
        if m := re.fullmatch(r"(?:move|copy) (.+)", text):
            return env[place(m[1])]
        if m := re.fullmatch(r"&(?:mut )?(.+)", text):
            return ("ref", place(m[1]))
        raise ValueError(f"unknown operand {text}")

    def expression(text):
        if text == "const ()":
            return "unit"
        if text == "Option::<Square>::None":
            return "none"
        if m := re.fullmatch(r"(\{closure@[^}]+\}) \{ ctx: (move _\d+), board: (move _\d+) \}", text):
            checked(env[deref(operand(m[2]))] == "ctx" and env[deref(operand(m[3]))] == "board",
                    "unexpected callback capture")
            return ("closure", m[1])
        if m := re.fullmatch(r"discriminant\((_\d+)\)", text):
            return {"claim": claim, "none": 0, ("some", "sq"): 1}[env[m[1]]]
        if m := re.fullmatch(r"Option::<Square>::Some\((.+)\)", text):
            return ("some", operand(m[1]))
        if m := re.fullmatch(r"Not\((.+)\)", text):
            value = operand(m[1])
            checked(value in (0, 1), f"unsupported Boolean negation {value}")
            return int(not value)
        if m := re.fullmatch(r"(Sub|Add)WithOverflow\((.+), (.+)\)", text):
            value = expression(f"{m[1]}({m[2]}, {m[3]})")
            checked(value in ("childDepth", "multiple") or type(value) is int and 0 <= value <= 2,
                    "arithmetic outside singleton move indices and depth")
            return ("checked", value, 0)
        if m := re.fullmatch(r"(Sub|Add|Eq|Ge|Le|Lt|Gt)\((.+), (.+)\)", text):
            op, left, right = m[1], operand(m[2]), operand(m[3])
            if type(left) is int and type(right) is int:
                return {
                    "Add": left + right, "Sub": left - right, "Eq": int(left == right),
                    "Ge": int(left >= right), "Le": int(left <= right),
                    "Lt": int(left < right), "Gt": int(left > right),
                }[op]
            return {
                ("Eq", "depth", 0): 0, ("Eq", "positiveIndex", 0): 0,
                ("Sub", "depth", 1): "childDepth",
                ("Eq", "multiple", 0): 0, ("Gt", "multiple", 1): 1, ("Sub", "multiple", 0): "multiple",
            }[(op, left, right)]
        return operand(text)

    externals = {
        "SearchContext::<'_>::prepare_pv": (["ctx"], "unit"),
        "Board::hash": (["board"], "hash"),
        "TranspositionTable::prefetch": (["tt", "hash"], "unit"),
        RESET: (["roots"], "unit"),
        "Board::legal_moves": (["board"], "legal"),
        "RootMoves::moves_bb_from_pv_idx": (["roots"], "suffix"),
        "<LegalMoves as BitAnd<Bitboard>>::bitand": (["legal", "suffix"], "selected"),
        "LegalMoves::is_empty": (["selected"], 0),
        "MoveList::with_moves": (["board", "selected"], "list"),
        "MoveList::count": (["list"], 1),
        "MoveList::wipeout_move": (["list"], "none"),
        "TranspositionTable::probe": (["tt", "board", "hash"], "probe"),
        "SearchStats::increment_tt_probe": (["stats"], "unit"),
        "TTProbeResult::<'_>::best_move": (["probe"], "hint"),
        "<MoveList as Index<usize>>::index": (["list", 0], ("ref", "one_move")),
        "Board::make_move_with_flipped": (["board", "flips", "sq"], "played"),
        "<Arc<threading::thread::Thread> as Deref>::deref": (["arc"], ("ref", "thread")),
        "claim_child::<SS>": (["thread", "played", "depth"], "claim"),
        UPDATE: (["ctx", "sq", "flips"], "unit"),
        CHILD: (["ctx", "played", "childDepth", "negBeta", "negAlpha", "arc", 0], "child"),
        UNDO: (["ctx"], "unit"),
        STOP: (["thread"], 0),
        "TTProbeResult::<'_>::slot": (["probe"], "slot"),
        "RootMoves::pv_idx": (["roots"], 0 if first_line else "positiveIndex"),
    }

    if frame is not None:
        externals.update(frame.externals)

    def call(fn, args):
        values = [env[deref(a)] if isinstance(a, tuple) and a[0] == "ref" else a for a in args]
        if fn in externals:
            expected, result = externals[fn]
            checked(values == expected, f"{fn} expected {expected}, got {values}")
            if fn in CHILD_EVENTS or frame is not None and fn in frame.observe:
                events.append(fn)
            return result
        if m := re.fullmatch(r"core::bool::<impl bool>::then::<Score, (\{closure@[^}]+\})>", fn):
            checked(values == [0, ("closure", m[1])], "evaluation closure is live")
            return "none"
        if m := re.fullmatch(r"Option::<Score>::map::<Score, (\{closure@[^}]+\})>", fn):
            checked(values == ["none", ("closure", m[1])], "correction closure is live")
            return "none"
        if fn == "<Score as Neg>::neg":
            checked(len(values) == 1, "score negation arity")
            return {
                -32767: 32767, 32767: -32767, "alpha": "negAlpha",
                "beta": "negBeta", "child": "negChild",
            }[values[0]]
        if m := re.fullmatch(r"<Score as PartialOrd>::(gt|lt|ge)", fn):
            return {
                ("gt", "negChild", -32767): above_floor, ("gt", "negChild", "alpha"): above_alpha,
                ("lt", "negChild", "beta"): below_beta, ("ge", "negChild", 16384): at_max,
            }[(m[1], *values)]
        if fn == "SearchContext::<'_>::update_root_move":
            checked(len(values) == 4 and values[0] == "ctx", "invalid root write")
            events.append(("write", *values[1:]))
            return "unit"
        if fn == "transposition_table::Bound::classify::<NT>":
            checked(len(values) == 3 and values[1:] == ["alpha", "beta"], "wrong TT classification")
            return ("bound", values[0])
        if fn == "TranspositionTable::store":
            checked(len(values) == 10, "invalid TT arity")
            table, slot, board, score, bound, depth, move, selectivity, endgame, pv = values
            checked([table, slot, board] == ["tt", "slot", "board"] and bound == ("bound", score)
                    and depth == "depth" and move in ("none", ("some", "sq"))
                    and [selectivity, endgame, pv] == ["selectivity", 1, 1], f"unexpected TT store {values}")
            events.append(("store", *values) if frame is not None else ("store", score))
            return "unit"
        raise ValueError(f"unknown external {fn} with arguments {values}")

    pc = 0 if frame is None else frame.start
    # Each branch must reach a normal return. The limit only rejects unsupported loops.
    for _ in range(500):
        if frame is not None and pc == frame.end:
            return events, env
        *statements, end = cfg[pc]
        for instruction in statements:
            if m := re.fullmatch(r"Storage(?:Live|Dead)\((_\d+)\);", instruction):
                env.pop(m[1], None)
                continue
            if re.fullmatch(r"PlaceMention\(_\d+\);|FakeRead\(ForLet\(None\), _\d+\);", instruction):
                continue
            if re.fullmatch(r"AscribeUserType\(_\d+, o, UserTypeProjection \{ base: UserType\(\d+\), projs: \[\] \}\);", instruction):
                continue
            m = re.fullmatch(r"(_\d+) = (.+);", instruction)
            checked(m, f"bb{pc}: {instruction}")
            try:
                env[m[1]] = expression(m[2])
            except (KeyError, ValueError) as error:
                raise ValueError(f"bb{pc}: {instruction} ({error})") from error
        if m := re.fullmatch(CALL, end):
            try:
                env[m[1]] = call(m[2], [operand(a) for a in m[3].split(", ") if a])
            except (KeyError, ValueError) as error:
                raise ValueError(f"bb{pc}: {end} ({error})") from error
            pc = int(m[4])
        elif m := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[((?:\d+: bb\d+, )*otherwise: bb\d+)\];", end):
            branches = {label: int(dest) for label, dest in re.findall(r"(\w+): bb(\d+)", m[2])}
            value = env[m[1]]
            checked(type(value) is int, f"unresolved branch condition {value}")
            pc = branches.get(str(value), branches["otherwise"])
        elif m := re.fullmatch(GOTO, end):
            pc = int(m[1] or m[2])
        elif m := re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];", end):
            checked(env[m[1]] == "claim", f"unexpected destructor {end}")
            pc = int(m[2])
        elif m := re.fullmatch(ASSERT, end):
            checked(operand(m[1]) == 0, "arithmetic overflow")
            pc = int(m[2])
        elif end == "return;":
            return events, env["_0"]
        else:
            raise ValueError(f"bb{pc}: unsupported terminator {end}")
    raise ValueError("singleton execution exceeded block limit")


def check_score_operators(repo):
    score = repo / "crates/reversi-core/src/score.rs"
    rust = [f'#[path = r#"{score}"#] mod score;', "use score::Score;"]
    operations = {}
    for name, symbol, compare in [("gt", ">", "sgt"), ("ge", ">=", "sge"), ("lt", "<", "slt")]:
        rust.append(f'#[unsafe(no_mangle)] pub extern "C" fn score_{name}(left:i32, right:i32)->bool '
                    f'{{ Score::from_raw(left) {symbol} Score::from_raw(right) }}')
        operations[name] = f"icmp {compare} i32 %left, %right", "i1"
    rust.append('#[unsafe(no_mangle)] pub extern "C" fn score_neg(raw:i32)->i32 { (-Score::from_raw(raw)).value() }')
    operations["neg"] = "sub i32 0, %raw", "i32"
    rust.append('#[unsafe(no_mangle)] pub extern "C" fn score_successor(raw:i32)->i32 { (Score::from_raw(raw) + 1).value() }')
    operations["successor"] = "add i32 %raw, 1", "i32"
    with tempfile.TemporaryDirectory() as scratch:
        src, ir = Path(scratch) / "ops.rs", Path(scratch) / "ops.ll"
        src.write_text("\n".join(rust), encoding="utf-8")
        subprocess.run(["rustc", "--edition=2024", "--crate-type=lib", "-Adead_code", "--emit=llvm-ir",
                        "-Copt-level=2", str(src), "-o", str(ir)], cwd=repo, check=True)
        llvm = ir.read_text(encoding="utf-8")
    for name, (operation, ty) in operations.items():
        body = re.findall(rf"^define [^\n]*@score_{name}\([^\n]*\{{\n(.*?)^\}}", llvm, re.M | re.S)
        checked(len(body) == 1 and body[0].splitlines() == [
            "start:", "  %_0 = " + operation, f"  ret {ty} %_0",
        ], f"unexpected compiled Score::{name} operator")


def render(mir, repo):
    source = mir.read_text(encoding="utf-8")
    checked(source.startswith("// MIR for `search::pvs::search` after SimplifyCfg-initial\n"), "wrong MIR")
    checked(SIGNATURE in source.splitlines(), "unexpected function signature")
    cfg = blocks(source)
    vals = constants(repo)
    strategy = (repo / "crates/reversi-core/src/search/strategy.rs").read_text(encoding="utf-8")
    endgame_flag = re.findall(r"^impl SearchStrategy for EndGameStrategy \{\s*const IS_ENDGAME: bool = (true|false);",
                              strategy, re.M)
    checked(endgame_flag == ["true"], "unsupported EndGameStrategy::IS_ENDGAME")
    vals["const <SS as search::strategy::SearchStrategy>::IS_ENDGAME"] = 1
    checked(vals["const score::Score::INF"] == 32767 and vals["const score::Score::MAX"] == 16384,
            "unsupported score representation")
    check_score_operators(repo)
    outputs = [set(), set()]
    for case in itertools.product(range(3), *([range(2)] * 5)):
        events, result = run(cfg, case, vals)
        if case[-1]:
            checked(events.pop() == ("store", result), "first-line TT publication differs from returned bound")
        checked(len(events) == 6 and events[:5] == CHILD_EVENTS, f"unexpected child events {events}")
        kind, square, score, pv = events[5]
        checked((kind, square) == ("write", "sq") and pv in (0, 1), "invalid root update")
        outputs[case[1]].add((score, pv, result))

    def lean(value):
        if value == "negChild":
            return "(-child)"
        if type(value) is int:
            return str(value)
        raise ValueError(f"unknown output {value}")

    def summary(rows):
        checked(len(rows) == 1, f"output depends on additional decisions {rows}")
        score, pv, result = next(iter(rows))
        return f"([(square, {lean(score)}, {str(bool(pv)).lower()})], {lean(result)})"

    return f"""import NeuralReversi.Bitboard.SquareModel

namespace NeuralReversi.PvsParallel

def singleRootExit (square : Bitboard.Square) (child : Int) :
    Prod (List (Prod Bitboard.Square (Prod Int Bool))) Int :=
  if -{vals['const score::Score::INF']} < -child then {summary(outputs[1])}
  else {summary(outputs[0])}

end NeuralReversi.PvsParallel
"""


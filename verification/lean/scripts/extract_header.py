"""Symbolic root-loop segment from its repeated entry up to a final child response."""

import itertools
from pathlib import Path
import re
import subprocess
import tempfile
import threading

from extract_single import SIGNATURE, check_score_operators
from mir_common import ASSERT, CALL, GOTO, blocks, checked, const_value
from extract_root_wipeout import constants

ENDGAME = "const <SS as search::strategy::SearchStrategy>::IS_ENDGAME"
CONDITIONS = ("idx < count", "idx = 0", "minSplit ≤ depth", "2 ≤ count - idx",
              "canSplit", "busy", "idx + 1 < fresh", "alpha < -scout")
PVS = "search::<PV, SS>"
SCOUT = "search::<NonPV, SS>"
_SPECIALIZATION_LOCK = threading.Lock()
_SPECIALIZATION_CHECKED = set()


def seed_specialization(repo):
    with _SPECIALIZATION_LOCK:
        _SPECIALIZATION_CHECKED.add(Path(repo).resolve())

def specialization(repo):
    key = Path(repo).resolve()
    with _SPECIALIZATION_LOCK:
        if key in _SPECIALIZATION_CHECKED:
            return
    src = key / "crates/reversi-core/src/search"
    strategy = (src / "strategy.rs").read_text(encoding="utf-8")
    checked(re.findall(r"^impl SearchStrategy for EndGameStrategy \{\s*const IS_ENDGAME: bool = (true|false);",
                       strategy, re.M) == ["true"], "unsupported endgame flag")
    rust = f'''#![feature(core_intrinsics)]
#[path = r#"{src / "node_type.rs"}"#] mod node_type;
mod search {{
    mod threading {{ #[path = r#"{src / "threading/in_flight.rs"}"#] pub(super) mod in_flight; }}
    pub fn tags() -> [isize; 3] {{
        use threading::in_flight::Claim;
        let slot = std::sync::atomic::AtomicU64::new(0);
        [Claim::Owned(&slot), Claim::Busy, Claim::Untracked]
            .map(|c| std::intrinsics::discriminant_value(&c))
    }}
}}
fn main() {{
    use node_type::{{Root, PV, NodeType, NodeTypeId}};
    assert_eq!(Root::ID, NodeTypeId::Root);
    assert_eq!(PV::ID, NodeTypeId::Pv);
    assert_eq!(search::tags(), [0, 1, 2]);
}}
'''
    with tempfile.TemporaryDirectory() as scratch:
        path, exe = Path(scratch) / "specialization.rs", Path(scratch) / "specialization.exe"
        path.write_text(rust, encoding="utf-8")
        subprocess.run(["rustc", "--edition=2024", "-Adead_code", "-Ainternal_features", str(path), "-o", str(exe)],
                       cwd=key, check=True)
        subprocess.run([str(exe)], check=True)
    with _SPECIALIZATION_LOCK:
        _SPECIALIZATION_CHECKED.add(key)


def binding(source, name, last=False):
    found = re.findall(r"^\s*debug " + name + r" => (_\d+);$", source, re.M)
    checked(len(found) == 1 or (last and found), f"ambiguous/missing binding {name}")
    return found[-1]


def ordinary(instructions):
    return [i for i in instructions if not re.fullmatch(r"Storage(?:Live|Dead)\(_\d+\);", i)]


def lmr_returns_zero(source):
    checked(source.startswith("// MIR for `search::pvs::compute_lmr_reduction` after SimplifyCfg-initial\n"),
            "wrong reduction MIR")
    cfg = blocks(source)
    pc = 0
    env = {}
    visited = set()
    while pc not in visited:
        visited.add(pc)
        *assignments, end = ordinary(cfg[pc])
        for instruction in assignments:
            m = re.fullmatch(r"(_\d+) = (.+);", instruction)
            checked(m and m[2] in (ENDGAME, "const 0_u32"), f"unexpected reduction operation {instruction}")
            env[m[1]] = int(m[2] == ENDGAME)
        if m := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];", end):
            pc = int(m[3] if env[m[1]] else m[2])
        elif m := re.fullmatch(r"goto -> bb(\d+);", end):
            pc = int(m[1])
        elif end == "return;":
            checked(env["_0"] == 0, "endgame reduction was nonzero")
            return
        else:
            raise ValueError(f"unsupported reduction terminator {end}")
    raise ValueError("loop in endgame reduction")


def initial(source, cfg):
    names = ["ctx", "board", "depth", "alpha", "beta", "thread", "cut_node", "all_node",
             "org_alpha", "move_list", "tt_pv", "n_moves", "best_move", "best_score", "fresh_end", "idx"]
    names = {name: binding(source, name) for name in names}
    outer_count = re.findall(r"^\s*debug move_count => (_\d+);$", source, re.M)
    checked(len(outer_count) == 2, "expected a loop-local shadow of move_count")
    matches = [num for num, code in cfg.items()
               if f"{names['idx']} = copy {outer_count[0]};" in code
               and f"{names['fresh_end']} = copy {names['n_moves']};" in code]
    checked(len(matches) == 1, "could not identify loop initialization")
    init = cfg[matches[0]]
    head = re.fullmatch(r"goto -> bb(\d+);", init[-1])
    checked(head, "missing loop entry")
    required = [f"{names['fresh_end']} = copy {names['n_moves']};", f"FakeRead(ForLet(None), {names['fresh_end']});",
                f"{names['idx']} = copy {outer_count[0]};", f"FakeRead(ForLet(None), {names['idx']});", init[-1]]
    checked(ordinary(init) == required, "unexpected loop initializer with additional effects")
    names["move"] = binding(source, "mv", last=True)
    names["score"] = binding(source, "score", last=True)
    return names, int(head[1])


_RE_STORAGE = re.compile(r"Storage(?:Live|Dead)\((_\d+)\);")
_RE_NOP = re.compile(
    r"PlaceMention\(_\d+\);|FakeRead\(ForLet\(None\), _\d+\);|"
    r"AscribeUserType\(_\d+, o, UserTypeProjection \{ base: UserType\(\d+\), projs: \[\] \}\);"
)
_RE_ASSIGN = re.compile(r"(_\d+) = (.+);")
_RE_CALL = re.compile(CALL)
_RE_SWITCH = re.compile(r"switchInt\(move (_\d+)\) -> \[((?:\d+: bb\d+, )*otherwise: bb\d+)\];")
_RE_SWITCH_BRANCH = re.compile(r"(\w+): bb(\d+)")
_RE_GOTO = re.compile(GOTO)
_RE_DROP = re.compile(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];")
_RE_ASSERT = re.compile(ASSERT)
_RE_LOCAL = re.compile(r"_\d+")
_RE_DEREF = re.compile(r"\(\*(_\d+)\)")
_RE_SELECTIVITY = re.compile(r"\(\(\*(_\d+)\)\.7: probcut::Selectivity\)")
_RE_CONST_INT = re.compile(r"const (\d+)_(?:usize|u32|i32)")
_RE_PROJ = re.compile(r"(?:move|copy) \((_\d+)\.(\d): ([^)]+)\)")
_RE_MOVE_COPY = re.compile(r"(?:move|copy) (.+)")
_RE_REF = re.compile(r"&(?:mut )?(.+)")
_RE_DISC = re.compile(r"discriminant\((_\d+)\)")
_RE_NOT = re.compile(r"Not\((.+)\)")
_RE_BINOP = re.compile(r"(Sub|Add|Lt|Le|Ge|Eq|Gt)(WithOverflow)?\((.+), (.+)\)")
_RE_SPLIT_REQ = re.compile(r"SplitRequest::<'_> \{ (.*) \}")

_STMT_CACHE = {}
_TERM_CACHE = {}
_EXPR_CACHE = {}
_OPERAND_CACHE = {}
_PLACE_CACHE = {}


def _parse_place(text):
    parsed = _PLACE_CACHE.get(text)
    if parsed is not None:
        return parsed
    if _RE_LOCAL.fullmatch(text):
        parsed = ("local", text)
    elif m := _RE_DEREF.fullmatch(text):
        parsed = ("deref", m[1])
    elif m := _RE_SELECTIVITY.fullmatch(text):
        parsed = ("selectivity", m[1])
    else:
        parsed = ("err", text)
    _PLACE_CACHE[text] = parsed
    return parsed


def _parse_operand(text):
    parsed = _OPERAND_CACHE.get(text)
    if parsed is not None:
        return parsed
    if m := _RE_CONST_INT.fullmatch(text):
        parsed = ("int", int(m[1]))
    elif m := _RE_PROJ.fullmatch(text):
        parsed = ("proj", m[1], m[2], m[3])
    elif m := _RE_MOVE_COPY.fullmatch(text):
        parsed = ("copy", _parse_place(m[1]))
    elif m := _RE_REF.fullmatch(text):
        parsed = ("ref", _parse_place(m[1]))
    else:
        parsed = ("err", text)
    _OPERAND_CACHE[text] = parsed
    return parsed


def _parse_expr(rhs):
    parsed = _EXPR_CACHE.get(rhs)
    if parsed is not None:
        return parsed
    if rhs == "const ()":
        parsed = ("unit",)
    elif m := _RE_DISC.fullmatch(rhs):
        parsed = ("disc", m[1])
    elif m := _RE_NOT.fullmatch(rhs):
        parsed = ("not", m[1])
    elif m := _RE_BINOP.fullmatch(rhs):
        parsed = ("binop", m[1], bool(m[2]), m[3], m[4])
    elif m := _RE_SPLIT_REQ.fullmatch(rhs):
        parsed = ("split_req", tuple(tuple(item.split(": ", 1)) for item in m[1].split(", ")))
    else:
        parsed = ("operand", rhs)
    _EXPR_CACHE[rhs] = parsed
    return parsed


def _parse_stmt(ins):
    parsed = _STMT_CACHE.get(ins)
    if parsed is not None:
        return parsed
    if m := _RE_STORAGE.fullmatch(ins):
        parsed = ("storage", m[1])
    elif _RE_NOP.fullmatch(ins):
        parsed = ("nop",)
    elif m := _RE_ASSIGN.fullmatch(ins):
        parsed = ("assign", m[1], _parse_expr(m[2]))
    else:
        parsed = ("err", ins)
    _STMT_CACHE[ins] = parsed
    return parsed


def _parse_term(end):
    parsed = _TERM_CACHE.get(end)
    if parsed is not None:
        return parsed
    if m := _RE_CALL.fullmatch(end):
        args = tuple(a for a in m[3].split(", ") if a)
        parsed = ("call", m[1], m[2], args, int(m[4]))
    elif m := _RE_SWITCH.fullmatch(end):
        branches = {k: int(v) for k, v in _RE_SWITCH_BRANCH.findall(m[2])}
        parsed = ("switch", m[1], branches)
    elif m := _RE_GOTO.fullmatch(end):
        parsed = ("goto", int(m[1] or m[2]))
    elif m := _RE_DROP.fullmatch(end):
        parsed = ("drop", m[1], int(m[2]))
    elif m := _RE_ASSERT.fullmatch(end):
        parsed = ("assert", m[1], int(m[2]))
    else:
        parsed = ("err", end)
    _TERM_CACHE[end] = parsed
    return parsed


def transition(cfg, names, header, flags, traits, claim_tag, node="rootID", cut=0,
               tt_pv=1, speculative=False):
    inside, first, deep, enough, free, busy, has_fresh, scout_improves = flags
    env = {local: name for name, local in names.items() if name not in ("move", "score")}
    for name in ("ctx", "board", "thread"):
        env[names[name]] = ("ref", name)
    env.update({name: name for name in ("ctx", "board", "thread", "worker", "moveRecord", "selectivity")})
    env[names["tt_pv"]] = tt_pv
    env[names["all_node"]] = int(node == "nonPvID" and not cut)
    env[names["cut_node"]] = cut
    persistent = [(names[n], env[names[n]]) for n in names if n not in ("move", "score", "idx", "fresh_end")]
    traits = {**traits, ENDGAME: 1, "const search::endgame::MIN_SPLIT_DEPTH": "minSplit",
              "const search::endgame::SPECULATIVE_SPLIT_MAX_DEPTH": "maxSpeculative",
              "const <NT as search::node_type::NodeType>::ID": node}
    events = []
    pc = header
    externals = {
        "<Arc<threading::thread::Thread> as Deref>::deref": (["thread"], ("ref", "worker")),
        "threading::thread::Thread::can_split": (["worker"], free),
        "<MoveList as Index<usize>>::index": (["move_list", "idx"], ("ref", "moveRecord")),
        "Board::make_move_with_flipped": (["board", "flips", "sq"], "played"),
        "claim_child::<SS>": (["worker", "played", "depth"], "claim"),
        "MoveList::postpone_move": (["move_list", "idx"], "unit"),
        "SearchContext::<'_>::update": (["ctx", "sq", "flips"], "unit"),
        "compute_lmr_reduction::<NT, SS>": (["selectivity", "depth", "nextIndex", "n_moves",
                                             cut, tt_pv, "ordering", "alpha"], 0),
        "<Score as Add<i32>>::add": (["alpha", 1], "alphaPlusOne"),
        SCOUT: (["ctx", "played", "childDepth", "negAlphaPlusOne", "negAlpha", "thread", int(not cut)], "scout"),
        PVS: (["ctx", "played", "childDepth", "negBeta", "negAlpha", "thread", 0], "pv"),
        "SearchContext::<'_>::undo": (["ctx"], "undo"),
        "threading::thread::Thread::split": (["thread", "ctx", "request"], "split"),
        "transposition_table::Bound::classify::<NT>": (["best_score", "org_alpha", "beta"], "done"),
    }
    move_fields = {("0", "bitboard::Bitboard"): "flips", ("1", "i32"): "ordering",
                   ("2", "square::Square"): "sq"}
    neg_map = {32767: -32767, "alpha": "negAlpha", "beta": "negBeta",
               "alphaPlusOne": "negAlphaPlusOne", "scout": "negScout", "pv": "negPv"}
    expected_split_req = [
        ("board", ("ref", "board")), ("alpha", "alpha"), ("beta", "beta"),
        ("best_score", "best_score"), ("best_move", "best_move"), ("depth", "depth"),
        ("move_list", "move_list"), ("move_count", "idx"), ("node_type", node),
        ("cut_node", cut),
    ]

    def pointee(val):
        checked(isinstance(val, tuple) and val[0] == "ref", f"not a reference {val}")
        return val[1]

    def eval_place(parsed):
        kind = parsed[0]
        if kind == "local":
            return parsed[1]
        if kind == "deref":
            return pointee(env[parsed[1]])
        if kind == "selectivity":
            checked(pointee(env[parsed[1]]) == "ctx", "selectivity of another context")
            return "selectivity"
        raise ValueError(f"unsupported header place {parsed[1]}")

    def operand(text):
        if text in traits:
            return traits[text]
        parsed = _parse_operand(text)
        kind = parsed[0]
        if kind == "int":
            return parsed[1]
        if kind == "proj":
            value = env[parsed[1]]
            if value == "moveRecord":
                return move_fields[(parsed[2], parsed[3])]
            checked(isinstance(value, tuple) and value[0] == "checked", f"projection {text}: {value}")
            return value[int(parsed[2]) + 1]
        if kind == "copy":
            return env[eval_place(parsed[1])]
        if kind == "ref":
            return ("ref", eval_place(parsed[1]))
        raise ValueError(f"unsupported header operand {text}")

    arithmetic = {
        ("Lt", "idx", "n_moves"): inside, ("Ge", "idx", 1): not first,
        ("Ge", "depth", "minSplit"): deep, ("Sub", "n_moves", "idx"): "remaining",
        ("Le", "depth", "maxSpeculative"): speculative,
        ("Ge", "remaining", 2): enough, ("Eq", "idx", 0): first,
        ("Add", "idx", 1): "nextIndex", ("Lt", "nextIndex", "fresh_end"): has_fresh,
        ("Sub", "fresh_end", 1): "oneLessFresh", ("Gt", "nextIndex", 1): not first,
        ("Sub", "depth", 1): "childDepth", ("Sub", "childDepth", 0): "childDepth",
        ("Gt", 0, 0): 0, ("Eq", "nextIndex", 1): first,
    }

    def eval_expr(parsed):
        kind = parsed[0]
        if kind == "operand":
            return operand(parsed[1])
        if kind == "unit":
            return "unit"
        if kind == "disc":
            checked(env[parsed[1]] == "claim", "unknown discriminant")
            return claim_tag
        if kind == "not":
            condition = operand(parsed[1])
            checked(condition in (0, 1), "unknown negation")
            return int(not condition)
        if kind == "binop":
            value = arithmetic[(parsed[1], operand(parsed[3]), operand(parsed[4]))]
            return ("checked", value, 0) if parsed[2] else value
        actual = [(name, operand(value)) for name, value in parsed[1]]
        checked(actual == expected_split_req, f"invalid split request {actual}")
        return "request"

    def verify_carried():
        checked(all(env.get(k) == v for k, v in persistent), "loop-carried state changed before dispatch")

    idx_reg = names["idx"]
    fresh_reg = names["fresh_end"]
    move_reg = names["move"]
    score_reg = names["score"]

    for ticks in range(500):
        if pc == header and ticks:
            verify_carried()
            checked(events == ["MoveList::postpone_move"] and env[idx_reg] == "idx"
                    and env[fresh_reg] == "oneLessFresh", "invalid postpone back-edge")
            return ".postpone idx (fresh - 1)"
        block = cfg[pc]
        for ins in block[:-1]:
            stmt = _parse_stmt(ins)
            skind = stmt[0]
            if skind == "assign":
                try:
                    env[stmt[1]] = eval_expr(stmt[2])
                except (ValueError, KeyError) as err:
                    raise ValueError(f"bb{pc}: {ins}: {err}") from err
            elif skind == "storage":
                env.pop(stmt[1], None)
            elif skind == "nop":
                continue
            else:
                checked(False, f"bb{pc} unhandled {ins}")
        term = _parse_term(block[-1])
        tkind = term[0]
        if tkind == "call":
            _, dst, fn, raw_args, nxt = term
            args = [operand(a) for a in raw_args]
            values = [env[pointee(a)] if isinstance(a, tuple) and a[0] == "ref" else a for a in args]
            if fn == "<Score as Neg>::neg":
                value, = values
                result = neg_map[value]
            elif fn == "<Score as PartialOrd>::gt":
                checked(values == ["negScout", "alpha"], f"unexpected research comparison {values}")
                result = scout_improves
            else:
                expected, result = externals[fn]
                checked(values == expected, f"{fn} expected {expected}, got {values}")
                if fn in (PVS, SCOUT, "SearchContext::<'_>::update", "MoveList::postpone_move"):
                    events.append(fn)
                if result in ("done", "split"):
                    verify_carried()
                    checked(not events and env[idx_reg] == "idx", "nonempty prefix before handoff")
                    checked(env[fresh_reg] == "fresh_end", "unexpected fresh bound at exit")
                    return ".finish" if result == "done" else ".split"
                if result == "undo":
                    verify_carried()
                    checked(env[idx_reg] == "nextIndex" and env[fresh_reg] == "fresh_end",
                            "unexpected child cursor")
                    checked(env[move_reg] == "moveRecord" and events[0] == "SearchContext::<'_>::update",
                            "unmatched context update before undo")
                    search = {(PVS,): ("false true", "negPv", "-pv"), (SCOUT,): ("true false", "negScout", "-scout"),
                              (SCOUT, PVS): ("true true", "negPv", "-pv")}[tuple(events[1:])]
                    checked(env[score_reg] == search[1], "wrong final child reply")
                    return f".child (idx + 1) {search[0]} ({search[2]})"
            env[dst] = result
            pc = nxt
        elif tkind == "switch":
            val = env[term[1]]
            checked(isinstance(val, int), f"unresolved switch {val}")
            branches = term[2]
            pc = branches.get(str(int(val)), branches["otherwise"])
        elif tkind == "goto":
            pc = term[1]
        elif tkind == "drop":
            checked(env[term[1]] == "claim", "unsupported destructor")
            pc = term[2]
        elif tkind == "assert":
            checked(operand(term[1]) == 0, "arithmetic overflow")
            pc = term[2]
        else:
            raise ValueError(f"bb{pc} unsupported {block[-1]}")
    raise ValueError("header slice did not reach a transition")


def render(pvs, reduction, repo):
    source = pvs.read_text(encoding="utf-8")
    checked(source.startswith("// MIR for `search::pvs::search` after SimplifyCfg-initial\n")
            and SIGNATURE in source.splitlines(), "unexpected PVS signature")
    lmr_returns_zero(reduction.read_text(encoding="utf-8"))
    cfg = blocks(source)
    names, header = initial(source, cfg)
    traits = constants(repo)
    check_score_operators(repo)
    specialization(repo)
    threshold = const_value(pvs.parent, "search-endgame-MIN_SPLIT_DEPTH")
    checked(threshold > 5, f"MIN_SPLIT_DEPTH is {threshold}, but smallHeader "
            "(PvsParallel/PvSerialMir.lean) needs shallow-search depths (5 and below) never to split")
    results = {}
    for flags in itertools.product((False, True), repeat=len(CONDITIONS)):
        for tag in ([1] if flags[5] else [0, 2]):
            action = transition(cfg, names, header, flags, traits, tag)
            checked(results.setdefault(flags, action) == action, "behavior depends on non-busy claim subtype")

    branches = {}
    def build(path=()):
        if len(path) == len(CONDITIONS):
            return results[path]
        then, otherwise = build(path + (True,)), build(path + (False,))
        if then == otherwise:
            return then
        rhs = f"if {CONDITIONS[len(path)]} then {then} else {otherwise}"
        return branches.setdefault(rhs, f"branch{len(branches) + 1}")

    root = build()
    steps = "".join(f"  let {name} : HeaderAction := {rhs}\n" for rhs, name in branches.items())
    return f"""module

@[expose] public section

namespace NeuralReversi.PvsParallel

def rootMinimumSplitDepth : Nat := {threshold}

inductive HeaderAction where
  | finish
  | split
  | postpone (idx fresh : Nat)
  | child (next : Nat) (scoutUsed pvUsed : Bool) (score : Int)
deriving DecidableEq, Repr

def headerAction (idx count fresh depth minSplit : Nat) (canSplit busy : Bool) (alpha scout pv : Int) :
    HeaderAction :=
""" + steps + f"  {root}\n\nend NeuralReversi.PvsParallel\n"


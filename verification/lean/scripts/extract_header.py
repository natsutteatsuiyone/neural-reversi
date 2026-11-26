"""Symbolic root-loop segment from its repeated entry up to a final child response."""

import itertools
from pathlib import Path
import re
import subprocess
import tempfile

from extract_single import SIGNATURE, check_score_operators
from mir_common import ASSERT, CALL, GOTO, blocks, checked, const_value
from extract_root_wipeout import constants

ENDGAME = "const <SS as search::strategy::SearchStrategy>::IS_ENDGAME"
CONDITIONS = ("idx < count", "idx = 0", "minSplit ≤ depth", "2 ≤ count - idx",
              "canSplit", "busy", "idx + 1 < fresh", "alpha < -scout")
PVS = "search::<PV, SS>"
SCOUT = "search::<NonPV, SS>"

def specialization(repo):
    src = repo / "crates/reversi-core/src/search"
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
                       cwd=repo, check=True)
        subprocess.run([str(exe)], check=True)


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
    persistent = {names[n]: env[names[n]] for n in names if n not in ("move", "score", "idx", "fresh_end")}
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

    def pointee(val):
        checked(isinstance(val, tuple) and val[0] == "ref", f"not a reference {val}")
        return val[1]

    def place(text):
        if re.fullmatch(r"_\d+", text):
            return text
        if m := re.fullmatch(r"\(\*(_\d+)\)", text):
            return pointee(env[m[1]])
        if m := re.fullmatch(r"\(\(\*(_\d+)\)\.7: probcut::Selectivity\)", text):
            checked(pointee(env[m[1]]) == "ctx", "selectivity of another context")
            return "selectivity"
        raise ValueError(f"unsupported header place {text}")

    def operand(text):
        if text in traits:
            return traits[text]
        if m := re.fullmatch(r"const (\d+)_(?:usize|u32|i32)", text):
            return int(m[1])
        if m := re.fullmatch(r"(?:move|copy) \((_\d+)\.(\d): ([^)]+)\)", text):
            value = env[m[1]]
            if value == "moveRecord":
                return {("0", "bitboard::Bitboard"): "flips", ("1", "i32"): "ordering",
                        ("2", "square::Square"): "sq"}[(m[2], m[3])]
            checked(isinstance(value, tuple) and value[0] == "checked", f"projection {text}: {value}")
            return value[int(m[2]) + 1]
        if m := re.fullmatch(r"(?:move|copy) (.+)", text):
            return env[place(m[1])]
        if m := re.fullmatch(r"&(?:mut )?(.+)", text):
            return ("ref", place(m[1]))
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

    def expr(rhs):
        if rhs == "const ()":
            return "unit"
        if m := re.fullmatch(r"discriminant\((_\d+)\)", rhs):
            checked(env[m[1]] == "claim", "unknown discriminant")
            return claim_tag
        if m := re.fullmatch(r"Not\((.+)\)", rhs):
            condition = operand(m[1])
            checked(condition in (0, 1), "unknown negation")
            return int(not condition)
        if m := re.fullmatch(r"(Sub|Add|Lt|Le|Ge|Eq|Gt)(WithOverflow)?\((.+), (.+)\)", rhs):
            value = arithmetic[(m[1], operand(m[3]), operand(m[4]))]
            return ("checked", value, 0) if m[2] else value
        if m := re.fullmatch(r"SplitRequest::<'_> \{ (.*) \}", rhs):
            fields = [item.split(": ", 1) for item in m[1].split(", ")]
            actual = [(name, operand(value)) for name, value in fields]
            expected = [("board", ("ref", "board")), ("alpha", "alpha"), ("beta", "beta"),
                        ("best_score", "best_score"), ("best_move", "best_move"), ("depth", "depth"),
                        ("move_list", "move_list"), ("move_count", "idx"), ("node_type", node),
                        ("cut_node", cut)]
            checked(actual == expected, f"invalid split request {actual}")
            return "request"
        return operand(rhs)

    def verify_carried():
        checked(all(env.get(k) == v for k, v in persistent.items()), "loop-carried state changed before dispatch")

    for ticks in range(500):
        if pc == header and ticks:
            verify_carried()
            checked(events == ["MoveList::postpone_move"] and env[names["idx"]] == "idx"
                    and env[names["fresh_end"]] == "oneLessFresh", "invalid postpone back-edge")
            return ".postpone idx (fresh - 1)"
        *statements, end = cfg[pc]
        for ins in statements:
            if m := re.fullmatch(r"Storage(?:Live|Dead)\((_\d+)\);", ins):
                env.pop(m[1], None)
                continue
            if re.fullmatch(r"PlaceMention\(_\d+\);|FakeRead\(ForLet\(None\), _\d+\);", ins):
                continue
            if re.fullmatch(r"AscribeUserType\(_\d+, o, UserTypeProjection \{ base: UserType\(\d+\), projs: \[\] \}\);", ins):
                continue
            m = re.fullmatch(r"(_\d+) = (.+);", ins)
            checked(m, f"bb{pc} unhandled {ins}")
            try:
                env[m[1]] = expr(m[2])
            except (ValueError, KeyError) as err:
                raise ValueError(f"bb{pc}: {ins}: {err}") from err
        if m := re.fullmatch(CALL, end):
            args = [operand(a) for a in m[3].split(", ") if a]
            values = [env[pointee(a)] if isinstance(a, tuple) and a[0] == "ref" else a for a in args]
            fn = m[2]
            if fn == "<Score as Neg>::neg":
                value, = values
                result = {32767: -32767, "alpha": "negAlpha", "beta": "negBeta",
                          "alphaPlusOne": "negAlphaPlusOne", "scout": "negScout", "pv": "negPv"}[value]
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
                    checked(not events and env[names["idx"]] == "idx", "nonempty prefix before handoff")
                    checked(env[names["fresh_end"]] == "fresh_end", "unexpected fresh bound at exit")
                    return ".finish" if result == "done" else ".split"
                if result == "undo":
                    verify_carried()
                    checked(env[names["idx"]] == "nextIndex" and env[names["fresh_end"]] == "fresh_end",
                            "unexpected child cursor")
                    checked(env[names["move"]] == "moveRecord" and events[0] == "SearchContext::<'_>::update",
                            "unmatched context update before undo")
                    search = {(PVS,): ("false true", "negPv", "-pv"), (SCOUT,): ("true false", "negScout", "-scout"),
                              (SCOUT, PVS): ("true true", "negPv", "-pv")}[tuple(events[1:])]
                    checked(env[names["score"]] == search[1], "wrong final child reply")
                    return f".child (idx + 1) {search[0]} ({search[2]})"
            env[m[1]] = result
            pc = int(m[4])
        elif m := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[((?:\d+: bb\d+, )*otherwise: bb\d+)\];", end):
            branches = dict(re.findall(r"(\w+): bb(\d+)", m[2]))
            val = env[m[1]]
            checked(isinstance(val, int), f"unresolved switch {val}")
            pc = int(branches.get(str(int(val)), branches["otherwise"]))
        elif m := re.fullmatch(GOTO, end):
            pc = int(m[1] or m[2])
        elif m := re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];", end):
            checked(env[m[1]] == "claim", "unsupported destructor")
            pc = int(m[2])
        elif m := re.fullmatch(ASSERT, end):
            checked(operand(m[1]) == 0, "arithmetic overflow")
            pc = int(m[2])
        else:
            raise ValueError(f"bb{pc} unsupported {end}")
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
    return f"""namespace NeuralReversi.PvsParallel

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


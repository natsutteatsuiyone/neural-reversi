"""Translate the completed-iteration path from the pinned rustc MIR to Lean."""

from dataclasses import dataclass
import re

from mir_common import run_extractor_cli, PASS, checked


METHODS = {
    "snapshot": ("root_move-{impl#1}-snapshot", "(_1: &RootMoves) -> Vec<RootMove>"),
    "key": ("result-{impl#0}-capture-{closure#0}", "(_1: &mut CLOSURE, _2: &RootMove) -> Reverse<Score>"),
    "capture": ("result-{impl#0}-capture", "(_1: &SearchContext<'_>, _2: u32, _3: bool) -> Iteration"),
    "first": ("result-{impl#0}-best_move", "(_1: &Iteration) -> &RootMove"),
    "record": ("result-{impl#1}-iteration_completed",
               "(_1: &mut CompletedState, _2: &SearchContext<'_>, _3: u32, _4: bool) -> &RootMove"),
    "row": ("result-{impl#2}-from_root_move_snapshot-{closure#0}", "(_1: &mut CLOSURE, _2: &RootMove) -> PvMove"),
    "result": ("result-{impl#2}-from_root_move_snapshot",
               "(_1: &[RootMove], _2: &RootMove, _3: u32, _4: Selectivity, _5: bool, _6: SearchCounters) -> SearchResult"),
    "finish": ("result-{impl#1}-into_result", "(_1: CompletedState, _2: &SearchContext<'_>, _3: &Board) -> SearchResult"),
    "score": ("result-{impl#2}-score", "(_1: &SearchResult) -> Option<Score>"),
}


def read(dump, method):
    name, signature = METHODS[method]
    path = dump / f"reversi_core.search-{name}.{PASS}.after.mir"
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
    method_name = name.split("}-", 1)[1].replace("-{closure#0}", "::{closure#0}")
    header = re.fullmatch(r"// MIR for `([^\x60]+::" + re.escape(method_name) + r")` after " + PASS, lines[0])
    checked(header, f"unexpected MIR method: {name}")
    closures = set(re.findall(r"\{closure@[^}]+\}", "\n".join(lines)))
    # Distinguish closure bodies by caller and source span, including inside generic arguments.
    def norm(line):
        for closure in closures:
            line = line.replace(closure, "CLOSURE")
        return line
    printed = header[1].removeprefix("search::") if method == "snapshot" else header[1]
    checked(any(norm(line) == f"fn {printed}{signature} {{" for line in lines), f"{method} signature")
    blocks = {}
    current = None
    for line in lines:
        if m := re.fullmatch(r"bb(\d+)(?: \(cleanup\))?: \{", line):
            current = int(m[1])
            checked(current not in blocks, "repeated MIR block")
            blocks[current] = []
        elif line == "}":
            current = None
        elif current is not None:
            if re.fullmatch(r"Storage(?:Live|Dead)\(_\d+\);|FakeRead\(ForLet\(None\), _\d+\);", line):
                continue
            blocks[current].append(norm(line))
    return blocks, closures


@dataclass(frozen=True)
class Value:
    kind: str
    text: str
    place: str = ""


# MIR field numbers for the production types. A source-layout check below binds
# these numbers to their Rust names (two adjacent fields have the same Score type).
FIELDS = {
    "context": {1: ("rows", "rows"), 7: ("sel", "selectivity"), 5: ("tt", "tt")},
    "roots": {0: ("rows", "rows")},
    "row": {0: ("square", "square"), 1: ("score", "score"), 2: ("score", "average"), 4: ("pv", "pv")},
    "iter": {0: ("rows", "rows"), 1: ("depth", "depth"), 2: ("sel", "selectivity"), 3: ("bool", "endgame")},
    "state": {1: ("last", "last")},
    "counters": {0: ("nodes", "nodes")},
    "best": {1: ("score", "best.score")},
}


def construct(name, fields):
    get = lambda k: fields[k].text
    if name == "Iteration":
        checked(set(fields) == {"root_moves", "depth", "selectivity", "is_endgame"}, "Iteration fields")
        return Value("iter", f"⟨{get('root_moves')}, {get('depth')}, {get('selectivity')}, {get('is_endgame')}⟩")
    if name == "PvMove":
        checked(set(fields) == {"sq", "score", "pv_line"}, "PvMove fields")
        return Value("publicRow", f"⟨{get('sq')}, {get('score')}, {get('pv_line')}⟩")
    checked(name == "SearchResult::BestMove" and set(fields) == {
        "sq", "score", "n_nodes", "pv_line", "depth", "selectivity", "is_endgame", "pv_moves", "counters"},
        "unexpected SearchResult construction")
    return Value("output", f".bestMove ⟨⟨{get('sq')}, {get('score')}, {get('pv_line')}⟩, {get('pv_moves')}, "
                 + f"⟨{get('depth')}, {get('selectivity')}⟩, {get('is_endgame')}, {get('n_nodes')}⟩")


def translate(blocks, method, inputs, variant=None):
    env = {f"_{i}": v for i, v in enumerate(inputs, 1)}
    visited = set()
    emitted = []

    def arg(local):
        checked(re.fullmatch(r"_\d+", local), f"bad register {local}")
        return env[local]

    def at(base, index):
        kind, field = FIELDS[base.kind][int(index)]
        if base.kind in ("state", "context", "counters", "roots"):
            term = field
        else:
            term = f"{base.text}.{field}"
        return Value(kind, term, "@last" if kind == "last" else "")

    def expr(s):
        if m := re.fullmatch(r"(?:copy |move |&mut |&)?(_\d+)", s):
            v = arg(m[1])
            if s.startswith("&mut "):
                return Value(v.kind, v.text, v.place or m[1])
            return v
        if m := re.fullmatch(r"(?:&(?:mut )?|copy |move )?\(\*(_\d+)\)", s):
            return arg(m[1])
        if m := re.fullmatch(r"(?:copy |move |&mut |&)?\(\(\*(_\d+)\)\.(\d+): .+\)", s):
            return at(arg(m[1]), m[2])
        if m := re.fullmatch(r"(?:copy |move |&)?\((_\d+)\.(\d+): .+\)", s):
            return at(arg(m[1]), m[2])
        if m := re.fullmatch(r"&\(\*\(\(\*(_\d+)\)\.(\d+): .+\)\)", s):
            return at(arg(m[1]), m[2])
        if m := re.fullmatch(r"move \(\(\((_\d+)\.(\d+): [^()]+\) as Some\)\.0: search::result::Iteration\)", s):
            checked(at(arg(m[1]), m[2]).kind == "last" and variant == "some", "invalid Some projection")
            return Value("iter", "saved")
        if m := re.fullmatch(r"&\(\(\(\*(_\d+)\) as BestMove\)\.(\d+): score::Score\)", s):
            checked(arg(m[1]).kind == "output" and variant == "best", "invalid BestMove projection")
            return at(Value("best", "report"), m[2])
        if m := re.fullmatch(r"Reverse::<Score>\(move (_\d+)\)", s):
            v = arg(m[1])
            checked(v.kind == "score", "unexpected reverse key")
            return Value("reverse", v.text)
        if m := re.fullmatch(r"(Iteration|PvMove|SearchResult::BestMove) \{ (.*) \}", s):
            fields = dict(pair.split(": ",1) for pair in m[2].split(", "))
            return construct(m[1], {k: expr(v) for k,v in fields.items()})
        if m := re.fullmatch(r"Option::<Score>::Some\(move (_\d+)\)", s):
            return Value("optScore", "some " + arg(m[1]).text)
        if s == "Option::<Score>::None":
            return Value("optScore", "none")
        if s == "CLOSURE" and method in ("capture", "result"):
            return Value("closure", "key" if method == "capture" else "project")
        if s.startswith('const "') and method == "first":
            return Value("message", "")
        if s == "discriminant((*_1))" and method == "score":
            return Value("tag", "0" if variant == "best" else "1")
        if s == "discriminant((_1.1: std::option::Option<search::result::Iteration>))" and method == "finish":
            return Value("tag", "1" if variant == "some" else "0")
        raise ValueError(f"{method}: unknown expression {s}")

    def call(name, values):
        args = [v.text for v in values]
        if name == "RootMoves::snapshot":
            return Value("rows", f"(snapshot {args[0]})")
        # Match the same two rustc spellings already accepted for the root sort;
        # the receiver, key closure and subsequent effects are still checked.
        if name in ("std::slice::<impl [RootMove]>::sort_by_key::<Reverse<Score>, CLOSURE>",
                    "slice::<impl [RootMove]>::sort_by_key::<Reverse<Score>, CLOSURE>"):
            ptr, key = values
            checked(ptr.place in env and key.kind == "closure", "sort receiver or key")
            env[ptr.place] = Value("rows", f"(sort.byReverseKey {key.text} {ptr.text})")
            return Value("unit","()")
        if name == "Iteration::capture":
            checked(values[0].kind == "context", "capture receiver")
            return Value("iter", f"(capture sort rows {args[1]} selectivity {args[2]})")
        if name == "Option::<Iteration>::insert":
            ptr, new = values
            checked(ptr.place == "@last", "not writing last")
            env["@last"] = Value("last", f"(some {new.text})")
            return new
        if name == "core::slice::<impl [RootMove]>::first":
            return Value("optRow", args[0] + ".head?")
        if name == "Option::<&RootMove>::expect":
            checked(values[0].kind == "optRow" and values[1].kind == "message", "expect")
            emitted.append(f"let chosen ← {args[0]}")
            return Value("row","chosen")
        if name == "Iteration::best_move":
            emitted.append(f"let chosen ← first {args[0]}")
            return Value("row","chosen")
        if name == "SearchContext::<'_>::counters":
            checked(values[0].kind == "context", "counters receiver")
            return Value("counters","nodes")
        if name == "SearchResult::from_root_move_snapshot":
            return Value("output", "(build " + " ".join(args) + ")")
        if name == "<std::slice::Iter<'_, RootMove> as Iterator>::map::<PvMove, CLOSURE>":
            return Value("pvRows", f"({args[0]}.map {args[1]})")
        if name in ("std::sync::Mutex::<Vec<RootMove>>::lock",
                    "Result::<std::sync::MutexGuard<'_, Vec<RootMove>>, PoisonError<std::sync::MutexGuard<'_, Vec<RootMove>>>>::unwrap",
                    "<std::sync::MutexGuard<'_, Vec<RootMove>> as Deref>::deref", "<Vec<RootMove> as Clone>::clone",
                    "<Vec<RootMove> as Deref>::deref", "<Vec<RootMove> as DerefMut>::deref_mut",
                    "core::slice::<impl [RootMove]>::iter", "<Vec<Square> as Clone>::clone",
                    "<Map<std::slice::Iter<'_, RootMove>, CLOSURE> as Iterator>::collect::<Vec<PvMove>>"):
            v, = values
            return v
        raise ValueError(f"{method}: unknown call {name}")

    bb = 0
    while True:
        checked(bb not in visited and bb in blocks, f"{method}: bad edge to bb{bb}")
        visited.add(bb)
        statements, term = blocks[bb][:-1], blocks[bb][-1]
        for line in statements:
            if re.fullmatch(r"PlaceMention\(.+\);|AscribeUserType\(_\d+, o, UserTypeProjection \{ base: UserType\(\d+\), projs: \[\] \}\);", line):
                continue
            m = re.fullmatch(r"(_\d+) = (.+);", line)
            checked(m, f"{method}: unknown statement: {line}")
            env[m[1]] = expr(m[2])
        if term == "return;":
            return emitted, env["_0"], env.get("@last")
        if m := re.fullmatch(r"(_\d+) = (.+)\((.*)\) -> \[return: bb(\d+), unwind: bb\d+\];", term):
            if m[2] == "tt_move" and method == "finish":
                checked(variant == "none", "fallback reached on a recorded iteration")
                # Everything starting at the first fallback action is intentionally opaque.
                return [], Value("output","(fallback ())"), None
            env[m[1]] = call(m[2], [expr(a) for a in m[3].split(", ")])
            bb = int(m[4])
        elif m := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), 1: bb(\d+), otherwise: bb\d+\];", term):
            tag = arg(m[1])
            checked(tag.kind == "tag", "expected enum discriminant")
            bb = int(m[2 + int(tag.text)])
        elif m := re.fullmatch(r"falseEdge -> \[real: bb(\d+), imaginary: bb\d+\];|goto -> bb(\d+);|drop\(_\d+\) -> \[return: bb(\d+), unwind: bb\d+\];", term):
            bb = int(next(x for x in m.groups() if x))
        else:
            raise ValueError(f"{method}: unknown terminator {term}")


def verify_layout(repo):
    src = repo / "crates/reversi-core/src"
    def derives(path, declaration, required):
        text = (src / path).read_text(encoding="utf-8")
        m = re.search(r"#\[derive\(([^)]+)\)\](?:\s*#\[[^\]]+\])*\s*" + declaration, text)
        checked(m and required <= {s.strip() for s in m[1].split(",")}, f"{declaration} derives")

    def fields(path, name):
        text = re.sub(r"//[^\n]*", "", (src/path).read_text(encoding="utf-8"))
        body = re.search(r"struct " + name + r"(?:<'a>)? \{([^}]+)\}", text)[1]
        return re.findall(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(\w+)\s*:", body, re.M)
    checked(fields("search/root_move.rs","RootMove") == ["sq","score","average_score","mean_squared_score","pv"],
            "RootMove field map changed")
    checked(fields("search/root_move.rs","RootMoves")[0] == "moves", "RootMoves field map changed")
    checked(fields("search/result.rs","Iteration") == ["root_moves","depth","selectivity","is_endgame"]
            and fields("search/result.rs","CompletedState") == ["lines","last"], "iteration/state fields")
    context = fields("search/context.rs","SearchContext")
    checked((context[1], context[5], context[7]) == ("root_moves","tt","selectivity"), "context fields")
    result = re.sub(r"//[^\n]*", "", (src/"search/result.rs").read_text(encoding="utf-8"))
    checked(re.search(r"pub enum SearchResult\s*\{\s*BestMove\s*\{\s*sq: Square,\s*score: Score,"
                      r"[^{}]*\},\s*NoLegalMove,\s*\}", result),
            "SearchResult variant/field map")
    derives("score.rs", r"pub struct Score\(i32\);", {"Clone", "Ord", "PartialOrd"})
    derives("search/root_move.rs", r"pub struct RootMove \{", {"Clone"})
    derives("square.rs", r"pub enum Square \{", {"Clone"})


def render(dump, repo):
    verify_layout(repo)
    methods = {m: read(dump, m) for m in METHODS}
    for caller, callback in (("capture","key"), ("result","row")):
        checked(len(methods[caller][1]) == 1 and methods[caller][1] == methods[callback][1],
                f"{caller}: wrong callback")
    def run(method, args, variant=None):
        return translate(methods[method][0], method, [Value(*v) for v in args], variant)
    lines = ["module", "public import NeuralReversi.PublicationMir.Syntax", "", "@[expose] public section", "", "namespace NeuralReversi.PublicationMir", "",
             "open RootMoveStorage (Row)", ""]
    def definition(method, params, typ, args, monad=False, recorded=False):
        emitted, output, state = run(method,args)
        term = f"({state.text}, {output.text})" if recorded else output.text
        name = {"result":"build","row":"project"}.get(method,method)
        start = f"def {name} {params} : {typ} :="
        lines.extend([start+" do", *["  "+x for x in emitted], "  return "+term,""]
                     if monad else [start, "  "+term,""])
    context=("context","ctx")
    rows=("rows","rows")
    row=("row","r")
    d=("depth","depth")
    sel=("sel","selectivity")
    e=("bool","endgame")
    definition("snapshot","(rows : List Row)","List Row",[("roots","roots")])
    definition("key","(r : Row)","Int",[("closure",""),row])
    definition("capture","(sort : SliceSort) (rows : List Row) (depth selectivity : Nat) (endgame : Bool)",
               "Iteration",[context,d,e])
    definition("first","(saved : Iteration)","Option Row",[("iter","saved")],monad=True)
    definition("record","(sort : SliceSort) (rows : List Row) (depth selectivity : Nat) (endgame : Bool)",
               "Option (Option Iteration × Row)",[("state","state"),context,d,e],monad=True,recorded=True)
    definition("row","(r : Row)","SearchPublication.Row",[("closure",""),row])
    definition("result","(rows : List Row) (r : Row) (depth selectivity : Nat) (endgame : Bool) (nodes : Nat)",
               "SearchResult",[rows,row,d,sel,e,("counters","nodes")])
    lines.extend(["def intoResult (last : Option Iteration) (nodes : Nat) (fallback : Unit → SearchResult) :",
                  "    Option SearchResult :=", "  match last with"])
    for variant, pattern in (("none","none"), ("some","some saved")):
        emitted, out, _ = run("finish",[("state","state"),context,("board","board")], variant)
        lines.extend([f"  | {pattern} => do", *["    "+x for x in emitted], "    return "+out.text])
    lines.extend(["", "def score : SearchResult → Option Int"])
    for variant, pattern in (("best",".bestMove report"), ("none",".noLegalMove")):
        _, out, _ = run("score",[("output","result")],variant)
        lines.append("  | "+pattern+" => "+out.text)
    return "\n".join(lines + ["", "end NeuralReversi.PublicationMir", ""])


def main():
    run_extractor_cli(
        __doc__,
        render,
        filters="search::result|search::root_move",
        check_message="generated completed-publication program differs",
    )


if __name__=="__main__":
    main()

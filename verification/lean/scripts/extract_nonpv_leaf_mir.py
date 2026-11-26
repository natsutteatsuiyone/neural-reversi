"""Verify and compose normal MIR returns of NonPV endgame PVS and its short leaves."""

import re

from extract_fixed_leaf_mir import constant, expected, path_result, read_body
from mir_common import (
    run_extractor_cli,
    PASS,
    Block,
    parse_blocks as blocks,
    checked,
    normal_target,
    split_call,
)

FILTER = "pvs::search|shallow_search|DEPTH_TO_SHALLOW|DEPTH_TO_NWS|IS_ENDGAME|PV_NODE|evaluate|final_score"


def impl_constant(repo, dump, module, source_decl, name, typ):
    source = repo / f"crates/reversi-core/src/search/{module}.rs"
    lines = [i for i, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1)
             if line == source_decl]
    checked(len(lines) == 1, f"ambiguous impl in {source}")
    span = (rf"crates[\\/]reversi-core[\\/]src[\\/]search[\\/]{module}\.rs:{lines[0]}:1:"
            rf" {lines[0]}:{len(source_decl.removesuffix(' {')) + 1}")
    hdr = rf"{module}::<impl at {span}>::{name}"
    pattern = (rf"// MIR for `search::{hdr}` after built\s+const {hdr}: {typ} = \{{\s+"
               rf"let mut _0: {typ};\s+bb0: \{{\s+_0 = const (.+);\s+return;\s*\}}\s*\}}\s*")
    hits = [match[1] for p in dump.glob(f"reversi_core.search-{module}-*-{name}.built.after.mir")
            if (match := re.fullmatch(pattern, p.read_text(encoding="utf-8")))]
    checked(len(hits) == 1, f"missing or ambiguous compiler constant {source_decl} {name}")
    return hits[0]


def parameters(repo, dump, pv):
    decl = "impl NodeType for PV {" if pv else "impl NodeType for NonPV {"
    node = impl_constant(repo, dump, "node_type", decl, "PV_NODE", "bool")
    end = "impl SearchStrategy for EndGameStrategy {"
    strategy = impl_constant(repo, dump, "strategy", end, "IS_ENDGAME", "bool")
    limit = impl_constant(repo, dump, "strategy", end, "DEPTH_TO_SHALLOW", "u32")
    checked(node == ("true" if pv else "false") and strategy == "true", "unexpected specialization")
    checked(limit == "search::endgame::DEPTH_TO_NWS", "unrecognized shallow threshold definition")
    return {"const <NT as search::node_type::NodeType>::PV_NODE": pv,
            "const <SS as search::strategy::SearchStrategy>::IS_ENDGAME": True,
            "const <SS as search::strategy::SearchStrategy>::DEPTH_TO_SHALLOW": constant(dump, "DEPTH_TO_NWS")}


def pvs_body(dump):
    path = dump / f"reversi_core.search-pvs-search.{PASS}.after.mir"
    text = path.read_text(encoding="utf-8")
    checked(text.startswith(f"// MIR for `search::pvs::search` after {PASS}\n"), "wrong PVS MIR")
    signature = ("fn search(_1: &mut SearchContext<'_>, _2: &Board, _3: u32, _4: Score, _5: Score, "
                 "_6: &Arc<threading::thread::Thread>, _7: bool) -> Score {")
    checked(text.splitlines().count(signature) == 1, "unexpected PVS signature")
    units = set(re.findall(r"^\s+let (?:mut )?(_\d+): \(\);$", text, re.M))
    result = {}
    for m in re.finditer(r"^    bb(\d+)( \(cleanup\))?: \{\n((?:        [^\n]+\n)+)    \}$", text, re.M):
        bb = int(m[1])
        checked(bb not in result, f"duplicate bb{bb}")
        ops = []
        for line in m[3].splitlines():
            ins = line.strip()
            if re.fullmatch(r"Storage(?:Live|Dead)\(_\d+\);|FakeRead\(ForLet\(None\), _\d+\);", ins):
                continue
            if u := re.fullmatch(r"(_\d+) = const \(\);", ins):
                checked(u[1] in units, f"unit write to non-unit: {ins}")
                continue
            ops.append(ins)
        checked(ops, f"empty block bb{bb}")
        result[bb] = Block(ops[:-1], ops[-1], bool(m[2]))
    checked(len(result) == len(re.findall(r"^    bb\d+( \(cleanup\))?: \{", text, re.M)), "incomplete PVS body")
    return result


def follow_pvs(body, params, n, cut, pv=False):
    env = dict(_1="ctx", _2="board", _3=n, _4="alpha", _5="beta", _6="thread", _7=cut)
    calls = []

    def operand(expr):
        if expr in params:
            return params[expr]
        if expr == "const false":
            return False
        if expr == "const 0_u32":
            return 0
        if m := re.fullmatch(r"(?:copy|move) (_\d+)|&(?:mut )?\(\*(_\d+)\)", expr):
            key = m[1] or m[2]
            checked(key in env, f"undefined local {key}")
            return env[key]
        if expr == "copy ((*_1).10: u32)":
            calls.append(("count", "empties"))
            return n
        if m := re.fullmatch(r"Not\((.+)\)", expr):
            arg = operand(m[1])
            checked(type(arg) is bool, "non-boolean Negation")
            return not arg
        if m := re.fullmatch(r"(Le|Eq)\(([^,]+), (.+)\)", expr):
            left, right = operand(m[2]), operand(m[3])
            checked(type(left) is int and type(right) is int, "non-numeric depth comparison")
            return left <= right if m[1] == "Le" else left == right
        raise ValueError(f"unknown PVS operand: {expr}")

    pc, visited = 0, set()
    while True:
        checked(pc not in visited, f"loop at bb{pc}")
        visited.add(pc)
        block = body[pc]
        for ins in block.statements:
            m = re.fullmatch(r"(_\d+) = (.+);", ins)
            checked(m, f"unsupported PVS statement at bb{pc}: {ins}")
            env[m[1]] = operand(m[2])
        t = block.end
        if t == "return;":
            want = ("evaluate", "ctx", "board") if pv else ("shallow_search", "ctx", "board", n, "alpha", "beta", "thread")
            first = ("clear_pv", "ctx") if pv else ("count", "empties")
            checked(calls == [first, want] and env.get("_0") == want,
                    f"PVS({n}, {cut}) lost the shallow return or operands: {calls} -> {env.get('_0')}")
            return env["_0"]
        if m := re.fullmatch(r"goto -> bb(\d+);", t):
            pc = normal_target(body, m[1])
        elif m := re.fullmatch(r"switchInt\((.+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];", t):
            v = operand(m[1])
            checked(type(v) is bool, "non-boolean PVS branch")
            pc = normal_target(body, m[3] if v else m[2])
        elif m := split_call(t):
            names = {"<SS as SearchStrategy>::shallow_search": "shallow_search",
                     "SearchContext::<'_>::clear_pv": "clear_pv", "<SS as SearchStrategy>::evaluate": "evaluate"}
            checked(m[2] in names, f"unexpected PVS call {m[2]} at bb{pc}")
            call = (names[m[2]], *(operand(arg) for arg in m[3].split(", ")))
            calls.append(call)
            env[m[1]] = call
            pc = normal_target(body, m[4], m[5])
        else:
            raise ValueError(f"unexpected PVS terminator bb{pc}: {t}")


def forwarding_body(repo, dump, module, decl, method, args):
    source = repo / ("crates/reversi-core/src/" + module.replace("::", "/") + ".rs")
    lines = [i for i, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1) if line == decl]
    checked(len(lines) == 1, f"ambiguous {decl}")
    span = source.relative_to(repo).as_posix() + f":{lines[0]}:1: {lines[0]}:{len(decl.removesuffix(' {'))+1}"
    name = f"{module}::<impl at {span}>::{method}"
    candidates = []
    for path in dump.glob(f"reversi_core.{module.replace('::', '-')}-*-{method}.{PASS}.after.mir"):
        header = path.read_text(encoding="utf-8").splitlines()[0]
        if header.replace("\\", "/") == f"// MIR for `{name}` after {PASS}":
            original = header.split("`")[1]
            fn = original.removeprefix("search::")
            candidates.append(blocks(path, original, "fn " + fn + args))
    checked(len(candidates) == 1, f"missing or ambiguous compiler body {method}")
    return candidates[0]


def final_evaluation(repo, dump):
    evaluate = forwarding_body(repo, dump, "search::strategy", "impl SearchStrategy for EndGameStrategy {",
                               "evaluate", "(_1: &SearchContext<'_>, _2: &Board) -> Score {")
    final = forwarding_body(repo, dump, "board", "impl Board {", "final_score", "(_1: &Board) -> Score {")
    checked(evaluate == {0: Block(["_3 = &(*_2);"], "_0 = Board::final_score(move _3) -> [return: bb1, unwind: bb2];", False),
                         1: Block([], "return;", False), 2: Block([], "resume;", True)}, "unknown evaluate return")
    checked(final == {0: Block(["_3 = &(*_1);"], "_2 = Board::final_disc_diff(move _3) -> [return: bb1, unwind: bb3];", False),
                      1: Block([], "_0 = Score::from_disc_diff(move _2) -> [return: bb2, unwind: bb3];", False),
                      2: Block([], "return;", False), 3: Block([], "resume;", True)}, "unknown final_score return")


def verify(dump, repo):
    body = pvs_body(dump)
    params = parameters(repo, dump, False)
    pv_params = parameters(repo, dump, True)
    for cut in (False, True):
        follow_pvs(body, pv_params, 0, cut, True)
    final_evaluation(repo, dump)
    fixed = read_body(dump, repo)
    limit = constant(dump, "DEPTH_TO_NWS")
    shallow = constant(dump, "DEPTH_TO_SHALLOW_SEARCH")
    arms = []
    for n in range(7):
        for cut in (False, True):
            follow_pvs(body, params, n, cut)
        actual = path_result(fixed, n, limit, shallow)
        checked(actual == expected(n), f"PVS leaf result or arguments differ at {n} empty squares")
        leaf = actual[0][1][0]
        arms.append(".shallow" if leaf in ("shallow5", "shallow6") else ".fixed ." + leaf)
    return "\n".join(["import NeuralReversi.EndgameSearch.Direct.Mir.Program", "import NeuralReversi.ScoreRefresh.BoardBridge", "",
                      "namespace NeuralReversi.PvsParallel.NonPvMir", "",
                      "def pvZeroScore (checked : Bool) (board : Board.Board) : Except String ScoreRefresh.Score :=",
                      "  ScoreRefresh.BoardBridge.finalScore checked board", "",
                      "inductive Arm where", "  | fixed (arm : EndgameSearch.Direct.Mir.Arm)", "  | shallow", "",
                      "def dispatch : Nat → Option Arm",
                      *(f"  | {n} => some ({arm})" for n, arm in enumerate(arms)),
                      "  | _ => none", "", "end NeuralReversi.PvsParallel.NonPvMir", ""])


def main():
    run_extractor_cli(
        __doc__,
        verify,
        filters=FILTER,
        check_message="generated NonPV MIR program differs",
        check_ok_message="MIR NonPV and endgame leaf returns verified for full depths zero through six",
        exclude_alloc_bytes=False,
        extra_env={"CARGO_INCREMENTAL": "0"},
    )


if __name__ == "__main__":
    main()


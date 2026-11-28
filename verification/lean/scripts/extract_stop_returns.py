"""Project normally returning suffixes of pvs::search from the boolean poll or child result."""

from pathlib import Path
import re

from cache_owner_mir import Body, PASS, THREAD, WORKER
from check_move_iterator_mir import call, compare, graph, jump
from extract_cache_owner import routing
from mir_common import CALL, blocks, checked


def administrative(s):
    return re.fullmatch(r"Storage(?:Live|Dead)\(_\d+\);|FakeRead\(ForLet\(None\), _\d+\);", s)


def claim_drop(dump):
    body = Body(dump, "search-threading-in_flight-{impl#1}-drop")
    atomic = "std::sync::atomic::Atomic<u64>"
    compare("Claim::drop", graph(body.source), [
        ["_2 = discriminant((*_1));", "switchInt(move _2) -> [0: bb1, otherwise: bb4];"],
        ["falseEdge -> [real: bb2, imaginary: bb4];"],
        [f"_3 = &mut (((*_1) as Owned).0: &{atomic});", "_5 = &(*(*_3));",
         "_6 = std::sync::atomic::Ordering::Relaxed;",
         call("_4 = Atomic::<u64>::store(move _5, const search::threading::in_flight::EMPTY, move _6)", 3, 6)],
        ["_0 = const ();", jump(5)], ["_0 = const ();", jump(5)], ["return;"], ["unwind"]])


def abort_poll(dump):
    body = Body(dump, WORKER + "is_search_aborted")
    compare("is_search_aborted", graph(body.source), [
        ["_5 = &((*_1).5: std::sync::Arc<search::threading::abort::AbortState>);",
         call("_4 = <Arc<AbortState> as Deref>::deref(move _5)", 1, 7)],
        ["_3 = &(*_4);", call("_2 = AbortState::is_aborted(move _3)", 2, 7)],
        ["switchInt(move _2) -> [0: bb4, otherwise: bb3];"],
        ["_0 = const true;", jump(6)],
        ["_6 = &((*_1).7: std::sync::atomic::Atomic<bool>);", "_7 = std::sync::atomic::Ordering::Relaxed;",
         call("_0 = Atomic::<bool>::load(move _6, move _7)", 5, 7)],
        [jump(6)], ["return;"], ["unwind"]])


def finish(body, pc, value):
    """Retain every normal block and effect, including non-TT claim release."""
    path = []
    assigned = value == "_0"
    while pc not in [n for n, _ in path]:
        todo = [s for s in body.cfg[pc] if not administrative(s)]
        effects = []
        for stmt in todo[:-1]:
            checked(not assigned and stmt == f"_0 = {value};", f"bb{pc} return assignment: {stmt}")
            effects.append("zero" if value.startswith("const ") else "forward")
            assigned = True
        end = todo[-1]
        if end == "return;":
            checked(assigned, "no return value")
            return path + [(pc, effects + ["ret"])]
        if m := re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];", end):
            checked(re.search(r"^\s*let " + m[1] + r": search::threading::in_flight::Claim<'_>;$",
                              body.source, re.M), f"bb{pc} unrecognized destructor: {end}")
            effects.append("releaseClaim")
            nxt = int(m[2])
        else:
            m = re.fullmatch(r"goto -> bb(\d+);", end)
            checked(m, f"bb{pc} unexpected return edge: {end}")
            nxt = int(m[1])
        path.append((pc, effects))
        pc = nxt
    raise ValueError("cyclic return suffix")


def normal_target(body, pc):
    m = re.fullmatch(CALL, body.cfg[pc][-1])
    checked(m, f"bb{pc}: normal call edge")
    return int(m[4])


def project(dump):
    p = Body(dump, "search-pvs-search")
    seeds = {p.arg(x): x for x in ("ctx", "thread", "board")}
    routes = []
    for pc, fn, args, reg in p.calls:
        if fn not in (THREAD + "::should_stop", THREAD + "::should_stop_after_split"):
            continue
        checked(len(args) == 1 and p.resolve(args[0], seeds) == "thread", f"bb{pc}: poll receiver")
        branch = normal_target(p, pc)
        code = [s for s in p.cfg[branch] if not administrative(s)]
        m = re.fullmatch(r"switchInt\(move " + reg + r"\) -> \[0: bb(\d+), otherwise: bb(\d+)\];", code[-1])
        checked(len(code) == 1 and m, f"bb{branch}: poll result operand")
        mode = "cached" if fn.endswith("::should_stop") else "postJoin"
        routes.append((mode, int(m[1]), [(pc, ["poll"]), (branch, ["taken"])] +
                       finish(p, int(m[2]), "const score::Score::ZERO")))
    checked([m for m, _, _ in routes] == ["cached", "postJoin", "cached"], "PVS stop sites")

    pc, _, _, child = p.named("search::<NT, SS>")
    neg = normal_target(p, pc)
    dest, fn, operand, undo = re.fullmatch(CALL, p.cfg[neg][-1]).groups()
    checked(fn == "<Score as Neg>::neg" and operand == "move " + child
            and all(administrative(s) for s in p.cfg[neg][:-1]), "pass child word")
    undo = int(undo)
    undo_pc, _, args, _ = p.named("SearchContext::<'_>::undo_pass")
    checked(undo_pc == undo and p.resolve(args[0], seeds) == "ctx", "pass restore receiver")
    checked(all(administrative(s) or s == args[0].removeprefix("move ") + " = &mut (*" + p.arg("ctx") + ");"
                for s in p.cfg[undo][:-1]), "extra write before undo_pass")
    routes.append(("passChild", 0, [(pc, ["child"]), (neg, ["negate"]), (undo, ["undoPass"])] +
                   finish(p, normal_target(p, undo), "copy " + dest)))

    solve, _, args, dst = p.named("Board::solve")
    checked(dst == "_0" and p.resolve(args[0], seeds) == "board", "terminal board/result")
    count = args[1].removeprefix("move ")
    checked(p.defs[count] == [f"copy ((*{p.arg('ctx')}).10: u32)"], "terminal empties source")
    routes.append(("terminal", 0, [(solve, ["solve"])] + finish(p, normal_target(p, solve), "_0")))
    claim_drop(dump)
    abort_poll(dump)
    zero = (dump / ("reversi_core.score-{impl#5}-ZERO" + PASS)).read_text(encoding="utf-8")
    checked(blocks(zero) == {0: ["_0 = Score(const 0_i32);", "return;"]}, "Score::ZERO")
    sites = routing(dump, "search-pvs-search")
    pass_sites = [i for i, (n, target, transfer) in enumerate(sites)
                  if n == pc and target == "pvs" and transfer == "forward"]
    checked(len(pass_sites) == 1, "missing forced-pass recursion in the native event projection")
    return routes, pass_sites[0]


def render(dump):
    def site(row):
        n, ops = row
        return f"({n}, [{', '.join('.'+x for x in ops)}])"

    def route(item):
        mode, fallback, path = item
        return f"  ⟨.{mode}, {fallback}, [{', '.join(map(site, path))}]⟩"

    dump = Path(dump)
    routes, pass_site = project(dump)
    p = Body(dump, "search-pvs-search")
    leaf, _, _, dest = p.named("<SS as SearchStrategy>::shallow_search")
    checked(dest == "_0", "PVS boundary return word")
    forwarded = [leaf] + [n for n, _ in finish(p, normal_target(p, leaf), "_0")]
    deep = []
    for stem in ("search-endgame-null_window_search", "search-endgame-search_move_nws_ec"):
        body = Body(dump, stem)
        sites = []
        for pc, fn, args, reg in body.calls:
            if fn != "<C as SplitPointCutoff>::occurred":
                continue
            checked(body.resolve(args[0], {body.arg("cutoff"): "poller"}) == "poller", "recursive cutoff receiver")
            branch = normal_target(body, pc)
            match = re.fullmatch(r"switchInt\(move " + reg + r"\) -> \[0: bb\d+, otherwise: bb(\d+)\];",
                                 body.cfg[branch][-1])
            checked(match, "branch on recursive cutoff outcome")
            finish(body, int(match[1]), "const search::endgame::SPLIT_CUTOFF_SCORE")
            sites.append(pc)
        deep.append(sites)
    checked([len(x) for x in deep] == [2, 1], "unexpected recursion poll sites")
    boundary = Body(dump, "search-strategy-{impl#1}-shallow_search")
    pc, _, args, reg = boundary.named(THREAD + "::should_stop")
    checked(boundary.resolve(args[0], {boundary.arg("thread"): "thread"}) == "thread", "boundary stop receiver")
    tail = graph(boundary.source)
    counter = "copy ((*" + boundary.arg("ctx") + ").2: u64)"
    saved = [n for n, s in boundary.ops if s == "_35 = " + counter + ";"]
    read = [n for n, s in boundary.ops if s == "_141 = " + counter + ";"]
    checked(len(saved) == len(read) == 1 and boundary.defs["_35"] == [counter]
            and boundary.defs["_141"] == [counter] and boundary.defs["_142"] == ["copy _35"]
            and boundary.defs["_143"] == ["SubWithOverflow(copy _141, copy _142)"]
            and boundary.defs["_140"] == ["move (_143.0: u64)"]
            and boundary.defs["_139"] == ["Ge(move _140, const search::strategy::BOUNDARY_STORE_MIN_SUBTREE_NODES)"]
            and tail[69][-1] == "switchInt(move _139) -> [0: bb77, otherwise: bb70];",
            "boundary saved/current context node count and store threshold")
    checked(tail[normal_target(boundary, pc)] == [f"switchInt(move {reg}) -> [0: bb74, otherwise: bb73];"]
            and [tail[n] for n in (73,78,79,80,81)] ==
            [[jump(78)], [jump(79)], ["_136 = const ();", jump(80)], ["_0 = copy _134;",jump(81)], ["return;"]]
            and boundary.defs["_134"] == [("call","Score::from_disc_diff",["move _135"])],
            "boundary cutoff cannot publish and returns the converted score register")
    return ("module\npublic import NeuralReversi.PvsParallel.NativeStopReturns.Syntax\n\n"
            "@[expose] public section\n\n"
            "namespace NeuralReversi.PvsParallel.NativeStopReturns\n\n"
            f"def boundaryPcs : List Nat := {forwarded}\n"
            f"def deepPollPcs : List Nat := {deep[0]}\n"
            f"def movePollPcs : List Nat := {deep[1]}\n"
            f"def boundaryPollPc : Nat := {pc}\n\n"
            f"def nodesBeforePc : Nat := {saved[0]}\n"
            f"def nodesAfterPc : Nat := {read[0]}\n\n"
            f"def passSite : Nat := {pass_site}\n\n"
            "def routes : List Route := [\n" + ",\n".join(map(route, routes)) + "]\n\n"
            "end NeuralReversi.PvsParallel.NativeStopReturns\n")

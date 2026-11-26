"""Check the selective endgame PVS return, verifier effects and ordering dispatch in pinned MIR."""

import argparse
from itertools import product
from pathlib import Path
import re
import tempfile

from cache_owner_mir import PASS, Body
from check_serial_child_calls_mir import REPO, PV
from check_serial_level_mir import BORROW, REG
from mir_common import checked, CALL
from extract_general_nonpv_mir import auxiliary, plus_threshold
from extract_header import ENDGAME
from extract_pvs_context_mir import Graph

PVS, WRAP, DEC = "search-pvs-search", "search-endgame-try_probcut", "search-endgame-probcut_decision"
ORDER = "move_list-ordering-shallow_search_score"
MG = "search-midgame-"
CTX = "search-context-{impl#0}-"
SORT = "move_list-ordering-{impl#0}-evaluate_moves_by_search"
C = "SearchContext::<'_>::"
ctx, board, depth, beta, cut = ("arg" + str(i) for i in range(1, 6))
field = lambda n: ("field", ctx, n)
NO = "Option::<Score>::None"
DEPTH = "const <SS as search::strategy::SearchStrategy>::MIN_PROBCUT_DEPTH"


def trace(g, guards, start=0, resume=None):
    events, result, seen = [], None, set()
    pc = start
    while pc != resume:
        checked(pc not in seen, "unexpected cycle in finite decision")
        seen.add(pc)
        *ops, end = g.cfg[pc]
        for op in ops:
            lhs, _, rhs = op.removesuffix(";").partition(" = ")
            if lhs == "((*_1).9: eval::EvalMode)":
                events.append(("mode", g.val(rhs)))
            else:
                checked(REG.fullmatch(lhs), f"unexpected decision effect: {op}")
                if lhs == "_0":
                    result = g.val(rhs)
        if end == "return;":
            return events, result
        if end.endswith(" -> bb17;") and "unreachable_default_msg()" in end:
            return events, "unreachable"
        if m := re.fullmatch(CALL, end):
            args = m[3].split(", ") if m[3] else []
            call = (m[2], *(g.val(a) for a in args))
            events.append(call)
            if m[1] == "_0":
                result = call
        elif m := re.fullmatch(r"switchInt\((.+)\) -> \[(.+)\];", end):
            choice = g.val(m[1])
            checked(choice in guards, f"unexpected decision predicate {choice}")
            options = {k:int(t) for k,t in re.findall(r"(\w+): bb(\d+)", m[2])}
            pc = options.get(str(int(guards[choice])), options["otherwise"])
            continue
        else:
            checked(end.startswith(("goto -> ", "falseEdge -> ")), f"unexpected decision terminator {end}")
        pc, = g.normal(pc)
    return events, "resume"


def decisions(dump):
    g = Graph(dump, PVS, 7, {270:"static"})
    callers = [pc for pc,fn,_ in g.calls() if fn == "<SS as SearchStrategy>::try_probcut"]
    reachable = Body(dump, PVS).reachable(0)
    entrances = {target for pc in reachable if pc not in range(169,180)
                 for target in g.normal(pc) if target in range(169,180)}
    checked(callers == [172] and entrances == {169}, "ProbCut bypasses its node/depth gate")
    gate = ("Ge", depth, DEPTH)
    delegated = ("<SS as SearchStrategy>::try_probcut", ctx, board, depth, "arg5", "arg7", "static", "arg6")
    for pv, deep, positive in product((False, True), repeat=3):
        guards = {PV:pv, gate:deep, ("discriminant", delegated):positive}
        calls, out = trace(g, guards, 169, 180)
        expected = [("SearchStats::increment_probcut_attempt", field(12)), delegated] if deep and not pv else []
        won = bool(expected) and positive
        if won:
            expected.append(("SearchStats::increment_probcut_cut", field(12)))
        checked(calls == expected and out == (("unSome", delegated) if won else "resume"),
                "PVS did not return a successful ProbCut immediately or resume an unsuccessful one")
    g = Graph(dump, WRAP, 5)
    enabled = ("Selectivity::is_enabled", field(7))
    decision = ("probcut_decision", ctx, board, depth, beta, cut)
    for active in (False, True):
        expected = [enabled, ("mode", "EvalMode::Main"), decision, ("mode", "EvalMode::Small")] if active else [enabled]
        checked(trace(g, {enabled:active}) == (expected, decision if active else NO), "changed enabled/disabled dispatch")
    g = Graph(dump, DEC, 5)
    ev = ("evaluate", ctx, board)
    window = ("endgame_probcut_window", depth, field(7), cut, beta, ev)
    pc, screen = (("field", window, i) for i in (0, 1))
    cap = ("<Score as PartialOrd>::ge", pc, "const score::Score::MAX")
    miss = ("<Score as PartialOrd>::lt", ev, screen)
    lo = ("<Score as Sub<i32>>::sub", pc, "const 1_i32")
    value = ("evaluate_depth2", ctx, board, lo, pc)
    positive = ("<Score as PartialOrd>::ge", value, pc)
    result = ("core::bool::<impl bool>::then_some::<Score>", positive, beta)
    for maxed, screened in product((False, True), repeat=2):
        verify = not maxed and not screened
        expected = [ev, window, cap] + ([] if maxed else [miss]) + ([lo, value, positive, result] if verify else [])
        checked(trace(g, {cap:maxed, miss:screened}) == (expected, result if verify else NO),
                "wrong ProbCut screen, verification window or return payload")
    g = Graph(dump, ORDER, 3)
    inf = "const score::Score::INF"
    neg = "<Score as Neg>::neg"
    # The catch-all switch edge covers every other signed sort_depth.
    for n in range(4):
        if n < 3:
            inner = ("evaluate", ctx, board) if n == 0 else (f"evaluate_depth{n}", ctx, board, (neg, inf), inf)
            answer = (neg, inner)
            want = ([] if n == 0 else [(neg, inf)]) + [inner, answer]
        else:
            want, answer = [], "unreachable"
        checked(trace(g, {depth:n}) == (want, answer), "move ordering called an unexpected search")


# Follow every argument carrying the complete SearchContext, including on
# pass, terminal, wipeout and beta exits. Its projected fields have these
# concrete types; none can carry a reference to the TT or a worker.
RECIPIENTS = {n:(MG+n, 0) for n in ("evaluate", "evaluate_depth1", "evaluate_depth2")}
RECIPIENTS.update({C+n:(CTX+n, 0) for n in ("ply", "evaluate_main_with_key", "pattern_feature", "correct_eval",
                   "update", "undo", "update_pass", "undo_pass", "increment_nodes", "switch_players")})
RECIPIENTS["ordering::<impl MoveList>::evaluate_moves_fast"] = ("move_list-ordering-{impl#0}-evaluate_moves_fast", 1)
RECIPIENTS["evaluate_fast_value"] = ("move_list-ordering-evaluate_fast_value", 0)
RECIPIENTS["ordering::<impl MoveList>::evaluate_moves::<NT, SS>"] = ("move_list-ordering-{impl#0}-evaluate_moves", 1)
RECIPIENTS["ordering::<impl MoveList>::evaluate_moves_by_search::<NT, SS>"] = (SORT, 1)
RECIPIENTS["shallow_search_score"] = (ORDER, 0)
RECIPIENTS[C+"prefetch_eval_cache"] = (CTX+"prefetch_eval_cache", 0)
FIELDS = {0:"eval::pattern_feature::PatternFeatures", 2:"u64",
          4:"std::option::Option<std::boxed::Box<search::correction_history::CorrectionHistory>>",
          6:"&eval::Eval", 8:"search::side_to_move::SideToMove", 9:"eval::EvalMode", 10:"empty_list::EmptyList"}


def frontier(dump):
    stack, seen = ["evaluate_depth2", "evaluate", "ordering::<impl MoveList>::evaluate_moves::<NT, SS>"], set()
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stem, arg = RECIPIENTS[name]
        g = Graph(dump, stem, arg+1)
        ops = [op for block in g.cfg.values() for op in block]
        loans = [m.groups() for op in ops if (m := BORROW.fullmatch(op))]
        aliases = {"_" + str(arg+1)}
        while extra := {dst for dst, src in loans if src in aliases} - aliases:
            aliases.update(extra)
        for op in ops:
            checked(not any(t in op for t in ("TranspositionTable", "TTEntry", "EndGameCache", "endgame_caches")),
                    f"{name}: verifier reached shared search storage")
            if aliases.isdisjoint(REG.findall(op)) or BORROW.fullmatch(op):
                continue
            if m := re.fullmatch(CALL, op):
                args = m[3].split(", ")
                checked(m[2] in RECIPIENTS and [i for i,a in enumerate(args) if not aliases.isdisjoint(REG.findall(a))]
                        == [RECIPIENTS[m[2]][1]], f"{name}: untracked full-context recipient {op}")
                stack.append(m[2])
            else:
                other = op
                for reg in aliases.intersection(REG.findall(op)):
                    for i, ty in FIELDS.items():
                        other = other.replace(f"((*{reg}).{i}: {ty})", "field")
                checked(aliases.isdisjoint(REG.findall(other)), f"{name}: unverified use of search context {op}")
    checked(seen == RECIPIENTS.keys(), "missing transitive evaluator")


def sort_depths(dump):
    g = Graph(dump, SORT, 7)
    field = "_8"  # debug sort_depth in the pinned MIR
    # Probe both sides of every depth bound, so that tuning the bounds needs no edit here.
    bounds = {int(m[1]) for block in g.cfg.values() for op in block if "copy _4" in op
              for m in re.finditer(r"const (\d+)_u32", op)}
    cases = {n for t in bounds for n in (t-1,t,t+1) if n >= 0} | {2**32-1}
    levels = {}
    for end, n in product((False, True), cases):
        pc, env, visited = 0, {"_4":n}, set()
        while pc != 19:
            checked(pc not in visited, "cycle before move-ordering sort_depth is chosen")
            visited.add(pc)
            *ops, term = g.cfg[pc]
            for op in ops:
                lhs, rhs = op.removesuffix(";").split(" = ")
                if rhs == ENDGAME:
                    env[lhs] = end
                elif m := re.fullmatch(r"const (\d)_i32", rhs):
                    env[lhs] = int(m[1])
                else:
                    test = re.fullmatch(r"Le\((.+), (.+)\)", rhs)
                    checked(test, f"new sort-depth expression {rhs}")
                    def word(arg):
                        if arg == "copy _4":
                            return n
                        lit = re.fullmatch(r"const (\d+)_u32", arg)
                        checked(lit, f"unverified sort-depth partition {arg}")
                        return int(lit[1])
                    env[lhs] = word(test[1]) <= word(test[2])
            if sw := re.fullmatch(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];", term):
                pc = int(sw[3] if env[sw[1]] else sw[2])
            else:
                pc, = g.normal(pc)
        levels[end, n] = env[field]
    for end in (False, True):
        schedule = [(n, level) for (mode, n), level in levels.items() if mode == end]
        first = max((n for n, level in schedule if level == 0), default=-1)
        last = max((n for n, level in schedule if level <= 1), default=-1)
        # The bounds are tuning parameters; decisions() shows that the scoring helper accepts only 0, 1 and 2.
        checked(first < last and all(level == int(n > first) + int(n > last) for n, level in schedule),
                "sort_depth is not a 0, 1, 2 step schedule")
    stores = [op for block in g.cfg.values() for op in block if op.startswith(field+" = ")]
    checked(len(stores) == 6, "sort_depth reassigned after dispatch")
    site, = [(pc,args) for pc, fn,args in g.calls() if fn == "shallow_search_score"]
    pc,args = site
    local = args[2].removeprefix("move ")
    checked(g.defs[local] == {"copy " + field}, "scoring helper did not receive selected sort_depth")


EVAL = "eval-{impl#0}-"
EC = "eval-eval_cache-{impl#0}-"
NEURAL = (EVAL+"evaluate_main", EVAL+"evaluate_main-{closure#0}", EVAL+"prefetch",
          EVAL+"evaluate_small", EC+"get_or_insert_with", EC+"probe_for_store")
TT_SOURCES = {
    PVS:(7, [(field(7), ENDGAME)]*2),
    "search-pvs-enhanced_transposition_cutoff":(7, [(field(7), ENDGAME)]),
    MG+"evaluate_shallow_with_tt":(5, [("Selectivity::None", "const false")]),
    "search-strategy-{impl#1}-shallow_search":(6, [("Selectivity::None", "const true")]),
}


def store_tags(dump):
    for stem, (argc, expected) in TT_SOURCES.items():
        g = Graph(dump, stem, argc)
        actual = [(g.val(args[7]),g.val(args[8])) for _,name,args in g.calls() if name == "TranspositionTable::store"]
        checked(actual == expected, f"{stem}: TT score was tagged with a different search grade")


def neural(dump):
    main, closure, prefetch, small, insert, probe = [Graph(dump, name, n) for name,n in zip(NEURAL, (5,1,2,3,3,2))]
    # A cache miss can publish a neural evaluation word; its address must be
    # projected from the Eval-owned bucket returned by probe_for_store.
    _, lookup, inputs = next(main.calls())
    checked(lookup.startswith("EvalCache::get_or_insert_with::<") and [main.val(a) for a in inputs[:2]] ==
            [field(2), "arg5"] and len(list(main.calls())) == 1, "main network used a different cache")
    checked([fn for _,fn,_ in closure.calls()] == ["Network::evaluate"] and
            [fn for _,fn,_ in small.calls()] == ["NetworkSmall::evaluate"], "unexpected network evaluation effect")
    prefetch.sole("EvalCache::prefetch", 0, field(2))
    checked(len(list(prefetch.calls())) == 1, "unknown neural cache prefetch")
    route = ("EvalCache::probe_for_store", ctx, board)
    insert.sole("Atomic::<u64>::store", 0, ("field",route,1))
    checked([fn for _,fn,_ in insert.calls()] == ["EvalCache::probe_for_store",
            "<impl FnOnce() -> Score as FnOnce<()>>::call_once", "pack", "Atomic::<u64>::store"],
            "unknown neural-cache publication")
    bucket = ("core::slice::<impl [Bucket]>::get_unchecked::<usize>",field(0),
              ("field",("EvalCache::location",ctx,board),0))
    slot = probe.val("_0")[2]
    checked(slot[:2] == ("core::slice::<impl [Atomic<u64>]>::get_unchecked::<usize>", ("field",bucket,0)),
            "evaluation payload escaped its own bucket")


def check(dump, repo=REPO):
    checked(plus_threshold(dump, repo, "MIN_PROBCUT_DEPTH") == 13, "wrong endgame ProbCut threshold")
    auxiliary(dump, repo)  # actual EndGameStrategy delegation and Selectivity::is_enabled
    decisions(dump)
    frontier(dump)
    sort_depths(dump)
    neural(dump)
    store_tags(dump)
    groups = [
        (DEC, "_34 = copy _4;", "_34 = copy _15;", decisions),
        (DEC, "<Score as PartialOrd>::lt", "<Score as PartialOrd>::le", decisions),
        (WRAP, "_17 = EvalMode::Small;", "_17 = EvalMode::Main;", decisions),
        (PVS, "Ge(move _278,", "Gt(move _278,", decisions),
        (ORDER, "_11 = evaluate_depth2(", "_11 = evaluate_shallow_with_tt(", decisions),
        (MG+"evaluate_depth1", "    bb0: {", "    bb0: {\n        ((*_1).7: probcut::Selectivity) = Selectivity::None;", frontier),
        (MG+"evaluate_depth2", " = evaluate_depth1(", " = evaluate_shallow_with_tt(", frontier),
        (SORT, "switchInt(move _13) -> [0: bb4, otherwise: bb3];", "switchInt(move _13) -> [0: bb3, otherwise: bb4];",
         sort_depths),
        (EC+"get_or_insert_with", "_17 = &(*_5);", "_17 = &(*_1);", neural),
        (MG+"evaluate_shallow_with_tt", ", const false, move ", ", const true, move ", store_tags),
    ]
    with tempfile.TemporaryDirectory(prefix="reversi-probcut-mutants-") as tmp:
        folder = Path(tmp)
        stems = {x[0] for x in RECIPIENTS.values()} | {PVS, DEC, WRAP, ORDER, *NEURAL, *TT_SOURCES}
        for stem in stems:
            path = "reversi_core." + stem + PASS
            (folder/path).write_text((dump/path).read_text())
        for stem, old, new, validate in groups:
            path = folder / ("reversi_core." + stem + PASS)
            original = path.read_text()
            checked(original.count(old) == 1, f"missing mutation anchor {stem}: {old}")
            path.write_text(original.replace(old, new))
            try:
                validate(folder)
            except ValueError:
                pass
            else:
                raise ValueError(f"accepted {stem} mutation {new}")
            path.write_text(original)
    return len(groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    print(f"verified selective ProbCut, ordering and verifier effects; rejected {check(args.mir, args.repo)} MIR mutations")

"""Check internal PV from the move-generation exit to the owner, and TT store to return."""

import re

from mir_common import checked, normal_target, split_call, blocks
from extract_header import binding, initial
from extract_nonpv_leaf_mir import parameters, pvs_body
from extract_pv_entry_mir import positive
from extract_root_wipeout import FILENAME, constants


def walk(body, start, env, traits, count, done, validate):
    calls, visited = [], set()

    def operand(s):
        if s in traits:
            return traits[s]
        if m := re.fullmatch(r"(?:move|copy) (_\d+)|&(?:mut )?\(\*(_\d+)\)|&(?:mut )?(_\d+)", s):
            return env[m[1] or m[2] or m[3]]
        known = {"&(*((*_1).5: &transposition_table::TranspositionTable))": "tt",
                 "&mut ((*_1).12: search::counters::SearchStats)": "stats",
                 "copy ((*_1).7: probcut::Selectivity)": "selectivity",
                 "Option::<Square>::None": "noMove"}
        if s in known:
            return known[s]
        if m := re.fullmatch(r"const (\d+)_usize", s):
            return int(m[1])
        if m := re.fullmatch(r"(\{closure@[^}]+\}) \{ ctx: move (_\d+), board: move (_\d+) \}", s):
            checked((env[m[2]], env[m[3]]) == ("ctx", "board"), "wrong excluded evaluation captures")
            return m[1]
        if m := re.fullmatch(r"discriminant\((_\d+)\)", s):
            checked(env[m[1]] == "noEval", "unexpected option")
            return 0
        if m := re.fullmatch(r"SubWithOverflow\(copy (_\d+), copy (_\d+)\)", s):
            checked(env[m[1]] == count and env[m[2]] == 0, "wrong remaining move count")
            return ("difference", count, False)
        if m := re.fullmatch(r"move \((_\d+)\.0: usize\)", s):
            checked(env[m[1]] == ("difference", count, False), "wrong subtraction")
            return count
        if m := re.fullmatch(r"(Gt|Eq)\(move (_\d+), const (\d+)_usize\)", s):
            val, limit = env[m[2]], int(m[3])
            checked(val == count and (m[1], limit) in (("Gt",1), ("Eq",2)), "wrong ordering condition")
            return val > limit if m[1] == "Gt" else val == limit
        raise ValueError(f"unsupported owner path operand {s}")

    pc = start
    while pc not in visited:
        if pc == done:
            return validate(env, calls)
        visited.add(pc)
        block = body[pc]
        for ins in block.statements:
            if m := re.fullmatch(r"PlaceMention\((_\d+)\);|AscribeUserType\((_\d+), o, UserTypeProjection \{ base: UserType\(\d+\), projs: \[\] \}\);", ins):
                checked((m[1] or m[2]) in env, f"absent place {ins}")
                continue
            m = re.fullmatch(r"(_\d+) = (.+);", ins)
            checked(m, f"unexpected owner instruction {ins}")
            env[m[1]] = operand(m[2])
        term = block.end
        if term == "return;":
            checked(done is None, "unexpected early return")
            return validate(env, calls)
        if c := split_call(term):
            args = tuple(operand(s) for s in c[3].split(", "))
            fn = c[2]
            if m := re.fullmatch(r"core::bool::<impl bool>::then::<Score, (\{closure@[^}]+\})>", fn):
                checked(args == (False,m[1]), "unexpected static evaluation")
                result, call = "noEval", None
            elif m := re.fullmatch(r"Option::<Score>::map::<Score, (\{closure@[^}]+\})>", fn):
                checked(args == ("noEval",m[1]), "unexpected corrected evaluation")
                result, call = "noEval", None
            else:
                names = {"TranspositionTable::probe": "probe", "SearchStats::increment_tt_probe": "incProbe",
                         "TTProbeResult::<'_>::best_move": "hint", "MoveList::count": "count", "<Score as Neg>::neg": "neg",
                         "ordering::<impl MoveList>::evaluate_moves::<NT, SS>": "evaluate", "ordering::<impl MoveList>::sort": "sort",
                         "transposition_table::Bound::classify::<NT>": "classify", "TTProbeResult::<'_>::slot": "slot",
                         "TranspositionTable::store": "store"}
                checked(fn in names, f"unexpected ordinary PV call bb{pc}: {fn}")
                call = (names[fn], *args)
                result = count if fn == "MoveList::count" else call
            if call is not None:
                calls.append(call)
            env[c[1]] = result
            pc = normal_target(body,c[4],c[5])
        elif g := re.fullmatch(r"goto -> bb(\d+);", term):
            pc = normal_target(body,g[1])
        elif sw := re.fullmatch(r"switchInt\((.+)\) -> \[(\d+): bb(\d+), otherwise: bb(\d+)\];", term):
            v = operand(sw[1])
            checked(isinstance(v,int), "unresolved ordinary PV branch")
            pc = normal_target(body, sw[3] if v == int(sw[2]) else sw[4])
        elif am := re.fullmatch(r'assert\(!move \((_\d+)\.1: bool\), "attempt to compute `\{\} - \{\}`, which would overflow", move (_\d+), move (_\d+)\) -> \[success: bb(\d+), unwind: bb(\d+)\];',term):
            checked((env[am[1]],env[am[2]],env[am[3]]) == (("difference",count,False),count,0), "wrong arithmetic assertion")
            pc = normal_target(body, am[4], am[5])
        else:
            raise ValueError(f"unexpected ordinary PV terminator bb{pc}: {term}")
    raise ValueError("cycle outside owner loop")


def verify(dump, repo):
    cfg = pvs_body(dump)
    src = (dump / FILENAME).read_text(encoding="utf-8")
    regs, header = initial(src, blocks(src))
    best = regs["best_score"]
    probe_reg = binding(src, "tt_probe_result")
    raw_reg = binding(src, "raw_eval")
    traits = dict(constants(repo), **parameters(repo, dump, True))
    traits["const <NT as search::node_type::NodeType>::ROOT_NODE"] = False
    gen = ("new","board")
    probe = ("probe","tt","board",("hash","board"))
    expected = [probe, ("incProbe","stats"), ("hint",probe), ("count",gen), ("neg",32767)]
    for cut in (False,True):
        for n in range(1,65):
            def initialized(env, calls):
                order = [] if n == 1 else [("evaluate",gen,"ctx","board","depth",("hint",probe),"alpha",cut), ("sort",gen)]
                checked(calls == [*expected,*order], "wrong ordinary PV order, score, or context")
                wanted = dict(ctx="ctx", board="board", depth="depth", alpha="alpha", beta="beta", thread="thread",
                              cut_node=cut, all_node=False, org_alpha="alpha", move_list=gen, tt_pv=True,
                              n_moves=n, best_move="noMove", best_score=("neg",32767), fresh_end=n, idx=0)
                checked(all(env[regs[k]] == value for k,value in wanted.items()), "wrong ordinary PV carried value")
                checked(env[probe_reg] == probe and env[raw_reg] == "noEval", "wrong owner TT probe or static eval")
                return True
            result = positive(cfg, traits, cut, True, False, False,
                              lambda pc,env: walk(cfg,pc,env,traits,n,header,initialized))
            checked(result is True, "PV failed to reach owner")
    finish, = [pc for pc,b in cfg.items() if "::Bound::classify::<NT>(" in b.end]
    state = {reg:name for name,reg in regs.items()}
    state.update({probe_reg:probe, raw_reg:"noEval", regs["tt_pv"]:True})
    def returned(env, calls):
        bound = ("classify","best_score","org_alpha","beta")
        slot = ("slot",probe)
        write = ("store","tt",slot,"board","best_score",bound,"depth","best_move","selectivity",True,True)
        checked(calls == [bound,slot,write] and env["_0"] == "best_score", "wrong ordinary PV store or return")
        return True
    checked(walk(cfg,finish,state,traits,0,None,returned), "PV did not return the owner score")

"""Follow exact NonPV presearch through its TT store, return, or ordered owner entry."""

from itertools import product
import re

from mir_common import Block, checked, normal_target, split_call, blocks
from extract_header import initial
from extract_root_wipeout import FILENAME


def verify(cfg, dump, traits, reach):
    src = (dump / FILENAME).read_text(encoding="utf-8")
    regs, header = initial(src, blocks(src))
    gen = ("new", "board")
    probe = ("probe", "tt", "board", ("hash", "board"))
    hint = ("hint", probe)
    indexed = ("index", gen, 0)
    f = ("flips", indexed)
    h = ("some", hint)
    names = {"SearchStats::increment_probcut_attempt": "stat", "<SS as SearchStrategy>::try_probcut": "prob",
             "MoveList::count": "count", "<Score as Neg>::neg": "neg", "Option::<Square>::is_some": "some?",
             "Score::value": "raw", "Board::player": "player", "Board::opponent": "opponent", "flip::flip": "flip",
             "<MoveList as Index<usize>>::index": "index", "Board::make_move_with_flipped": "make",
             "<Arc<threading::thread::Thread> as Deref>::deref": "deref", "claim_child::<SS>": "claim",
             "SearchContext::<'_>::update": "update", "<Score as Add<i32>>::add": "add",
             "search::<NonPV, SS>": "child", "SearchContext::<'_>::undo": "undo",
             "threading::thread::Thread::should_stop": "stop?", "<Score as PartialOrd>::gt": "gt",
             "<Score as PartialOrd>::ge": "ge", "TTProbeResult::<'_>::slot": "slot", "TranspositionTable::store": "store",
             "ordering::<impl MoveList>::evaluate_moves::<NT, SS>": "evaluate", "ordering::<impl MoveList>::sort": "sort",
             "<Option<Square> as PartialEq>::ne": "different", "MoveList::swap_moves": "swap"}
    def test(start, incoming, cut, count, has_hint, front, above, stopped):
        env, effects, visited = incoming.copy(), [], set()
        sq = h if has_hint else ("sq",indexed)
        flip = ("flip",h,("player","board"),("opponent","board")) if has_hint else f
        board = ("make","board",flip,sq)
        guard = ("claim",("deref","thread"),board,"depth")
        a1 = ("add","alpha",1)
        ch = ("child","ctx",board,("depthMinus1",),("neg",a1),("neg","alpha"),"thread",not cut)
        score = ("neg",ch)
        expected = [("stat","stats"), ("prob","ctx","board","depth","beta",cut,"noEval","thread"),
                    ("count",gen),("neg",32767)]
        if count != 1: expected.append(("some?",hint))
        use = has_hint or count == 1
        if use:
            expected.extend([("raw","beta"),("raw","alpha")])
            expected.extend([("player","board"),("opponent","board"),flip] if has_hint else [indexed])
            expected.extend([board,("deref","thread"),guard,("update","ctx",sq,flip),a1,("neg",a1),("neg","alpha"),ch,
                             score,("undo","ctx",sq),("deref","thread"),("stop?",("deref","thread"))])
            if not stopped:
                expected.append(("gt",score,"alpha"))
                if above:
                    expected.extend([("ge",score,"beta"),("slot",probe),
                        ("store","tt",("slot",probe),"board",score,"lower","depth",("some",sq),"exact",True,"ttPv")])
            expected.append(("drop",guard))
        returns = use and (above or stopped)
        if not returns:
            if count - int(use) > 1:
                expected.extend([("evaluate",gen,"ctx","board","depth",hint,"alpha",cut),("sort",gen)])
            elif count == 2:
                expected.extend([indexed,("different",("some",("sq",indexed)),hint)])
                if not front: expected.append(("swap",gen,0,1))

        def arg(s):
            if s in traits: return traits[s]
            known = {"const true":True,"copy ((*_1).7: probcut::Selectivity)":"exact",
                     "&mut ((*_1).12: search::counters::SearchStats)":"stats",
                     "&(*((*_1).5: &transposition_table::TranspositionTable))":"tt",
                     "Option::<Square>::None":"noMove","transposition_table::Bound::Lower":"lower"}
            if s in known: return known[s]
            if m := re.fullmatch(r"(?:move|copy) (_\d+)|&(?:mut )?\(\*(_\d+)\)|&(?:mut )?(_\d+)|copy \(\*(_\d+)\)",s):
                return env[next(x for x in m.groups() if x)]
            if m := re.fullmatch(r"const (\d+)_(?:usize|i32|u32)",s): return int(m[1])
            if m := re.fullmatch(r"(\{closure@[^}]+\}) \{ ctx: move (_\d+), board: move (_\d+) \}",s):
                checked((env[m[2]],env[m[3]])==("ctx","board"),"wrong static evaluation captures");return m[1]
            if m := re.fullmatch(r"discriminant\((_\d+)\)",s):
                v=env[m[1]]
                checked(v in ("noEval","noProb",hint),"unexpected discriminant")
                return int(v==hint and has_hint)
            if m := re.fullmatch(r"copy \(\((_\d+) as Some\)\.0: square::Square\)",s):
                checked(env[m[1]]==hint and has_hint,"wrong decoded TT move");return h
            if m := re.fullmatch(r"Option::<Square>::Some\(move (_\d+)\)",s): return ("some",env[m[1]])
            if m := re.fullmatch(r"\(move (_\d+), move (_\d+)\)",s): return (env[m[1]],env[m[2]])
            if m := re.fullmatch(r"(?:copy|move) \((?:\(\*(_\d+)\)|(_\d+))\.(\d+): ([^)]+)\)",s):
                v, i, ty = env[m[1] or m[2]],int(m[3]),m[4]
                if v==indexed:
                    checked((i,ty) in ((0,"bitboard::Bitboard"),(2,"square::Square")),"wrong move projection")
                    return f if i==0 else ("sq",indexed)
                checked(isinstance(v,tuple) and len(v)==2,"wrong tuple projection")
                return v[i]
            if m := re.fullmatch(r"Not\((.+)\)",s): return not arg(m[1])
            if m := re.fullmatch(r"Ge\(move (_\d+), const <SS as search::strategy::SearchStrategy>::MIN_PROBCUT_DEPTH\)",s):
                checked(env[m[1]]=="depth","wrong ProbCut depth");return True
            if m := re.fullmatch(r"(Eq|Gt|AddWithOverflow|SubWithOverflow)\(([^,]+), (.+)\)",s):
                a,b = arg(m[2]),arg(m[3])
                if m[1]=="Eq":
                    if (m[2],m[3]) == ("move _356","move _357"):
                        checked((a,b)==(("raw","beta"),("rawAlphaPlus1",)),"wrong debug window")
                        return True
                    checked((m[2],m[3]) in (("move _338","const 1_usize"),("move _475","const 2_usize"))
                            and a==count,"unexpected count partition")
                    return a==b
                if m[1]=="Gt":
                    checked((m[2],m[3])==("move _460","const 1_usize") and a==count-int(use),"wrong residual partition")
                    return a>b
                if m[1]=="AddWithOverflow":
                    checked((a,b)==(("raw","alpha"),1),"wrong null window assertion")
                    return (("rawAlphaPlus1",),False)
                if isinstance(a,int) and isinstance(b,int):
                    checked(a==count and b==int(use) and 0<=b<=a,"wrong residual or index underflow");return (a-b,False)
                checked((a,b)==("depth",1),"wrong recursion depth");return (("depthMinus1",),False)
            raise ValueError(f"unsupported presearch operand {s}")

        def dest(n, unwind=None):
            if unwind is not None and cfg[int(unwind)].end.startswith("drop("):
                checked(cfg[int(unwind)] == Block([], "drop(_388) -> [return: bb450, unwind terminate(cleanup)];",True)
                        and env["_388"]==guard,"wrong cleanup claim")
                unwind=450
            return normal_target(cfg,n,unwind)
        pc=start
        while pc not in visited:
            if pc==header:
                checked(not returns and effects==expected,"wrong presearch owner effects")
                want=dict(ctx="ctx",board="board",depth="depth",alpha="alpha",beta="beta",thread="thread",cut_node=cut,
                          all_node=not cut,org_alpha="alpha",move_list=gen,tt_pv="ttPv",n_moves=count,best_move="noMove",
                          best_score=score if use else ("neg",32767),fresh_end=count,idx=int(use))
                checked(all(env[regs[k]]==v for k,v in want.items()),"wrong presearch owner state")
                return True
            visited.add(pc)
            b=cfg[pc]
            for ins in b.statements:
                if m:=re.fullmatch(r"PlaceMention\((_\d+)\);|AscribeUserType\((_\d+), o, UserTypeProjection \{ base: UserType\(\d+\), projs: \[\] \}\);",ins):
                    checked((m[1] or m[2]) in env,"absent observed place");continue
                m=re.fullmatch(r"(_\d+) = (.+);",ins);checked(m,f"unexpected presearch statement {ins}")
                env[m[1]]=arg(m[2])
            t=b.end
            if t=="return;":
                checked(returns and env["_0"]==(0 if stopped else score) and effects==expected,"wrong presearch return or store")
                return True
            if c:=split_call(t):
                args=tuple(arg(a) for a in c[3].split(", "))
                if m:=re.fullmatch(r"core::bool::<impl bool>::then::<Score, (\{closure@[^}]+\})>",c[2]):
                    checked(args==(False,m[1]),"raw evaluation in endgame");v="noEval"
                elif m:=re.fullmatch(r"Option::<Score>::map::<Score, (\{closure@[^}]+\})>",c[2]):
                    checked(args==("noEval",m[1]),"correction history in endgame");v="noEval"
                else:
                    checked(c[2] in names,f"unexpected presearch function {c[2]}")
                    call=(names[c[2]],*args);effects.append(call)
                    v={"prob":"noProb","count":count,"some?":has_hint,"gt":above,"ge":above,"stop?":stopped,
                       "different":not front}.get(call[0],call)
                env[c[1]]=v;pc=dest(c[4],c[5])
            elif m:=re.fullmatch(r"goto -> bb(\d+);|falseEdge -> \[real: bb(\d+), imaginary: bb\d+\];",t):
                pc=dest(m[1] or m[2])
            elif m:=re.fullmatch(r"switchInt\((.+)\) -> \[(\d+): bb(\d+), otherwise: bb(\d+)\];",t):
                v=arg(m[1]);checked(isinstance(v,int),"unresolved presearch branch")
                pc=dest(m[3] if v==int(m[2]) else m[4])
            elif m:=re.fullmatch(r'assert\(!move \((_\d+)\.1: bool\), "attempt to compute `\{\} ([-+]) \{\}`, which would overflow", (.+), (.+)\) -> \[success: bb(\d+), unwind: bb(\d+)\];',t):
                operation="AddWithOverflow" if m[2]=="+" else "SubWithOverflow"
                checked(env[m[1]]==arg(f"{operation}({m[3]}, {m[4]})") and env[m[1]][1] is False,"wrong arithmetic assertion")
                pc=dest(m[5],m[6])
            elif m:=re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb(\d+)\];",t):
                effects.append(("drop",env[m[1]]));pc=dest(m[2],m[3])
            else: raise ValueError(f"unexpected presearch terminator bb{pc}: {t}")
        raise ValueError("unexpected presearch cycle")

    # With idx in {0,1}, comparisons partition move counts into 1, 2, and >=3.
    for cut, deep, hit, has_hint, front, above, stopped in product((False,True),repeat=7):
        if (has_hint and not hit) or (front and not has_hint): continue
        for n in (1,2,3):
            callback=lambda pc,env: test(pc,env,cut,n,has_hint,front,above,stopped)
            checked(reach(cut,deep,hit,callback),"NonPV did not reach the presearch")

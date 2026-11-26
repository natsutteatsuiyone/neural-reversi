"""Extract the normal score producers and per-move writes of endgame move ordering."""
import re

from mir_common import checked, split_call, PASS, blocks, const_value
from extract_header import ordinary

PRE = "move_list-ordering-"
IMPL = PRE + "{impl#0}-"
MARGIN = "move_list::ordering::<impl move_list::MoveList>::evaluate_moves_by_search::"
COL = "move_list::ordering::"

def read(dump, stem, pass_name=PASS):
    path = dump / f"reversi_core.{stem}.{pass_name}.after.mir"
    return path.read_text(encoding="utf-8").replace("\\", "/")

def code(dump, stem, pass_name=PASS):
    return {pc:[op for op in ordinary(ops) if not op.startswith(("FakeRead(", "AscribeUserType(", "PlaceMention("))]
            for pc,ops in blocks(read(dump, stem, pass_name)).items()}

def call_name(name):
    return name.split("::<{closure@")[0] if "::<{closure@" in name else name

def symbolic(cfg, start, stop, locals_, decisions, expected_writes=None):
    """Evaluate reachable statements with an arbitrary move and arbitrary child values."""
    env=dict(locals_);pc=start;writes=[];seen=set()
    def arg(s):
        if s.startswith("const "):
            lit=s[6:]
            n=re.fullmatch(r"(-?\d+)_(?:i32|u32|usize|u64)",lit)
            return int(n[1]) if n else lit
        if m:=re.fullmatch(r"(?:move |copy )?(_\d+)",s):return env[m[1]]
        if m:=re.fullmatch(r"(?:&(?:mut )?|\b(?:copy|move) )\(\*(_\d+)\)|&(?:mut )?(_\d+)",s):
            return env[m[1] or m[2]]

        if m:=re.fullmatch(r"(?:move|copy) \(\((_\d+) as Some\)\.0: (.+)\)",s):
            checked(env[m[1]]=="item" and m[2]=="&mut move_list::Move","unexpected item")
            return "move"
        if m:=re.fullmatch(r"(?:copy |move |&)\(\(\*(_\d+)\)\.(\d+): [^)]+\)",s):
            obj=env[m[1]];field=int(m[2])
            return env["value"] if field==1 and obj=="move" else ("field",obj,field)
        if m:=re.fullmatch(r"(?:copy|move) \((_\d+)\.(\d+): (.+)\)",s):
            obj=env[m[1]];field=int(m[2])
            return obj[1] if isinstance(obj,tuple) and obj[0]=="overflow" and field==0 else ("field",obj,field)
        if m:=re.fullmatch(r"move (_\d+) as (?:i32|&\[i32\]) \((?:IntToInt|PointerCoercion\(Unsize, Implicit\))\)",s):
            return env[m[1]]
        if m:=re.fullmatch(r"(?:Option::<Square>::Some|discriminant|Neg)\((.+)\)",s):
            return (s.split("(")[0],arg(m[1]))
        if m:=re.fullmatch(r"(\w+)\((.+), (.+)\)",s):
            op=m[1];left=arg(m[2]);right=arg(m[3])
            return ("overflow",(op[:-12],left,right)) if op.endswith("WithOverflow") else (op,left,right)
        raise ValueError(f"unsupported ordering operand {s}")
    while pc not in seen:
        if pc in stop and seen:break
        seen.add(pc);*ops,t=cfg[pc]
        for op in ops:
            if m:=re.fullmatch(r"\(\(\*(_\d+)\)\.1: i32\) = (.+);",op):
                checked(env[m[1]]=="move","key was not written to the current record")
                env["value"]=arg(m[2]);writes.append(env["value"]);continue
            m=re.fullmatch(r"(_\d+) = (.+);",op);checked(m,f"unsupported ordering instruction {op}")
            env[m[1]]=arg(m[2])
        if t=="return;":pc=-1;break
        if c:=split_call(t):
            fn=c[2];aa=tuple(arg(s) for s in c[3].split(", ")) if c[3] else ()
            v=(fn,*aa)
            if fn in ("<std::slice::IterMut<'_, move_list::Move> as Iterator>::next",):
                checked(aa==("iter",),"wrong move iterator");v="next"
            env[c[1]]=v;pc=int(c[4]);continue
        if m:=re.fullmatch(r"switchInt\((.+)\) -> \[(.+)\];",t):
            v=arg(m[1]);branches=dict(re.findall(r"(\d+|otherwise): bb(\d+)",m[2]))
            selected=decisions[v]
            pc=int(branches.get(str(int(selected)),branches["otherwise"]));continue
        if m:=re.fullmatch(r'assert\(.+\) -> \[success: bb(\d+), unwind: bb\d+\];',t):
            pc=int(m[1]);continue
        if m:=re.fullmatch(r'goto -> bb(\d+);|false(?:Edge|Unwind) -> \[real: bb(\d+), .+;',t):
            pc=int(m[1] or m[2]);continue
        raise ValueError(f"unsupported ordering terminator bb{pc}: {t}")
    checked(pc in stop, f"incorrect move-order iteration or exit {pc}")
    if expected_writes is not None:
        checked(writes==expected_writes,f"wrong writes: {writes} vs {expected_writes}")
    return env

# Lean names of the ordering weights and the dump stems of their Rust constants.
WEIGHTS = {
    "squareWeight": PRE + "SQUARE_VALUE_WEIGHT",
    "stabilityWeight": PRE + "CORNER_STABILITY_WEIGHT",
    "mobilityWeight": PRE + "MOBILITY_WEIGHT",
    "searchStabilityWeight": PRE + "CORNER_STABILITY_WEIGHT_SEARCH",
    "searchMobilityWeight": IMPL + "evaluate_moves_by_search-MOBILITY_SCALE_SEARCH",
    "searchPotentialWeight": IMPL + "evaluate_moves_by_search-POTENTIAL_MOBILITY_SCALE_SEARCH",
}

def constants(d):
    """Return the square table and the evaluated ordering weights keyed by their Lean names."""
    def body(name):return code(d,PRE+name,"built")
    square=body("SQUARE_VALUE")
    m=re.fullmatch(r"_0 = \[(.+)\];",square[0][0]);checked(m and square=={0:[m[0],"return;"]},"square weights")
    sq=[int(re.fullmatch(r"const (\d+)_i32",x)[1]) for x in m[1].split(", ")]
    checked(len(sq)==64,"bad square weight count")
    def rhs(name):
        return [x.split(" = ",1)[1][:-1] for block in body(name).values() for x in block if re.match(r"_\d+ = ",x)]
    tt=rhs("TT_MOVE_VALUE");checked(tt==["const 30_i32 as u32 (IntToInt)","Lt(move _1, const 32_u32)","Shl(const 1_i32, const 30_i32)"],"TT sentinel")
    return sq,{name:const_value(d,stem) for name,stem in WEIGHTS.items()}

def validate_heuristics(d):
    f=code(d,PRE+"fast_move_value")
    hit=symbolic(f,0,{-1},{"_1":"sq","_2":"next","_3":"moves"}, {})
    st=("Bitboard::corner_stability",("Board::opponent","next"))
    mob=("LegalMoves::corner_weighted_count","moves")
    val=("core::slice::<impl [i32]>::get_unchecked::<usize>",COL+"SQUARE_VALUE",("Square::index","sq"))
    expected=("Add",("Add",("<&i32 as Mul<i32>>::mul",val,COL+"SQUARE_VALUE_WEIGHT"),
               ("Mul",st,COL+"CORNER_STABILITY_WEIGHT")),("Mul",("Sub",36,mob),COL+"MOBILITY_WEIGHT"))
    checked(hit["_0"]==expected,"fast formula differs")
    move="move";sq=("field",move,2);flip=("field",move,0)
    n=("Board::make_move_with_flipped","board",flip,sq)
    hit=symbolic(code(d,PRE+"evaluate_fast_value"),0,{-1},{"_1":"ctx","_2":"board","_3":move},{})
    checked(hit["_0"]==("fast_move_value",sq,n,("Board::legal_moves",n)),"fast board or move key")
    fast=code(d,IMPL+"evaluate_moves_fast")
    fast_values=assignments(fast)
    checked(sole(fast_values,"_23")=="copy ((_4 as Some).0: square::Square)","TT hint payload")
    for tag,destination in ((0,1),(1,13)):
        symbolic(fast,0,{destination},{"_4":"input-hint"},{("discriminant","input-hint"):tag})
    instance=("evaluate_fast_value","ctx","board",move)
    hint="hint";ks=COL+"TT_MOVE_VALUE"
    for h in (True,False):
        symbolic(fast,21,{16},{"_30":"item","_23":hint,"_2":"ctx","_3":"board"},
            {("<Square as PartialEq>::eq",sq,hint):h},[ks if h else instance])
    symbolic(fast,9,{4},{"_13":"item","_2":"ctx","_3":"board"},{},[instance])
    search=code(d,IMPL+"evaluate_moves_by_search")
    for cfg in (fast,search):
        changed=assignments(cfg)
        checked(not any(f"_{arg}" in changed for arg in range(1,5 if cfg is fast else 8)),"move ordering inputs overwritten")
    hintEq=("<Option<Square> as PartialEq>::eq",("Option::<Square>::Some",sq),hint)
    for h in (True,False):
        opts={hintEq:h}
        skip=("Neg",("LegalMoves::corner_weighted_count",("Board::legal_moves",n)))
        symbolic(search,40,{35},{"_41":"item","_2":"ctx","_3":"board","_5":hint},opts,[ks if h else skip])
    for cached in (0,1):
        for h in (True,False):
            opts={hintEq:h,"cached":cached,"<SS as search::strategy::SearchStrategy>::IS_ENDGAME":1}
            pre=("SearchContext::<'_>::prefetch_eval_cache","ctx",n)
            ev=(("<Score as Neg>::neg",("SearchContext::<'_>::evaluate_main_with_key","ctx",n,pre))
                 if cached else ("shallow_search_score","ctx",n,"depth"))
            initial=("Score::value",ev)
            stable=("Bitboard::corner_stability",("Board::opponent",n))
            with_st=("Add",initial,("Mul",stable,COL+"CORNER_STABILITY_WEIGHT_SEARCH"))
            both=("Board::legal_moves_and_potential",n)
            with_mob=("Sub",with_st,("Mul",("Bitboard::corner_weighted_count",("field",both,0)),MARGIN+"MOBILITY_SCALE_SEARCH"))
            with_pm=("Sub",with_mob,("Mul",("Bitboard::corner_weighted_count",("field",both,1)),MARGIN+"POTENTIAL_MOBILITY_SCALE_SEARCH"))
            symbolic(search,65,{60},{"_76":"item","_2":"ctx","_3":"board","_5":hint,"_8":"depth","_63":"cached"},
                     opts,[ks] if h else [initial,with_st,with_mob,with_pm])

def assignments(cfg):
    result={}
    for ops in cfg.values():
        for op in ops:
            if c:=split_call(op):
                result.setdefault(c[1],[]).append((c[2],c[3]));continue
            m=re.fullmatch(r"(_\d+) = (.+);",op)
            if m:result.setdefault(m[1],[]).append(m[2])
    return result

def producer_graph(dump, name, allowed_calls):
    """Trace every definition of each returned Score, including loop-carried best locals."""
    ds=assignments(code(dump,name))
    def go(operand,path):
        if m:=re.fullmatch(r"(?:copy|move) (_\d+)",operand):return local(m[1],path)
        if operand in ("const score::Score::INF","const score::Score::MAX"):return {operand}
        raise ValueError(f"unknown origin {name}: {operand}")
    def local(reg, path):
        if reg in path:return {"backedge"}
        sources=set()
        for rhs in ds[reg]:
            if isinstance(rhs,str):sources |= go(rhs,path|{reg});continue
            fn,args=rhs
            if fn in ("<Score as Neg>::neg","<Score as Ord>::max"):
                sources.add(fn)
                for operand in args.split(", "):sources |= go(operand,path|{reg})
            else:
                checked(fn in allowed_calls,f"unknown ordering score source {name}: {fn}")
                sources.add(fn)
        checked(sources,"uninitialized score local");return sources
    return local("_0",set())

def aliases(ds,operand):
    """Unique backwards copy path. Unrecognized instructions remain visible to the caller."""
    h=re.fullmatch(r"(?:copy |move |&)?(_\d+)",operand)
    if not h or h[1] not in ds:return operand
    origins=ds[h[1]];checked(len(origins)==1,"alias was overwritten: "+h[1]);rhs=origins[0]
    return aliases(ds,rhs) if isinstance(rhs,str) else rhs

def sole(ds, reg="_0"):
    outputs=ds[reg]
    checked(len(outputs)==1,"unexpected join on return/publication");return outputs[0]

def simple_body(d,stem,expected):
    ds=assignments(code(d,stem)); checked(call_name(sole(ds)[0])==expected,f"unexpected callee in {stem}")
    return ds

def eval_sources(d):
    neg="<Score as Neg>::neg";mx="<Score as Ord>::max"
    for name,sources in (
        ("search-midgame-evaluate_depth1",{"evaluate","evaluate_depth1","Board::solve"}),
        ("search-midgame-evaluate_depth2",{"evaluate_depth1","evaluate_depth2","Board::solve"})):
        got=producer_graph(d,name,sources)
        checked(got==sources|{neg,mx,"backedge","const score::Score::INF","const score::Score::MAX"},
                f"unexpected {name} evaluation grammar: {got}")
        ds=assignments(code(d,name))
        for block in ds.values():
            for v in block:
                if isinstance(v,tuple) and v[0]=="Board::solve":
                    counts=aliases(ds,v[1].split(", ")[1])
                    checked(counts[0]=="EmptyList::count","terminal used an unaligned count")
    simple={"evaluate","evaluate_depth1","evaluate_depth2"}
    checked(producer_graph(d,PRE+"shallow_search_score",simple)==simple|{neg},"shallow scoring source")
    direct={"Board::final_score","SearchContext::<'_>::evaluate_main_with_key","Eval::evaluate_small"}
    checked(producer_graph(d,"search-midgame-evaluate",direct)==direct,"unexpected midgame leaf")
    ctx="search-context-{impl#0}-"
    ds=simple_body(d,ctx+"evaluate_main_with_key","SearchContext::<'_>::correct_eval")
    checked(aliases(ds,sole(ds)[1].split(", ")[2])[0]=="Eval::evaluate_main","main evaluation skipped origin")
    ds=assignments(code(d,ctx+"correct_eval"))
    outputs=ds["_0"];checked(len(outputs)==2 and "copy _3" in outputs and
        [x[0] for x in outputs if isinstance(x,tuple)]==["CorrectionHistory::correct"],"unchecked correction value")
    for stem in ("search-correction_history-{impl#1}-correct","eval-network-{impl#1}-evaluate-{closure#0}","eval-network_small-{impl#2}-evaluate"):
        ds=simple_body(d,stem,"<Score as Ord>::clamp")
        _,lo,hi=sole(ds)[1].split(", ")
        checked(aliases(ds,lo)==("<Score as Add<i32>>::add","const score::Score::MIN, const 1_i32") and
                aliases(ds,hi)==("<Score as Sub<i32>>::sub","const score::Score::MAX, const 1_i32"),"wrong score clamp limits")
    simple_body(d,"eval-{impl#0}-evaluate_small","NetworkSmall::evaluate")
    ds=simple_body(d,"eval-{impl#0}-evaluate_main","EvalCache::get_or_insert_with")
    closure=aliases(ds,sole(ds)[1].split(", ")[2]);checked(closure.startswith("{closure@") and "eval.rs:" in closure,"wrong cache miss closure")
    simple_body(d,"eval-{impl#0}-evaluate_main-{closure#0}","Network::evaluate")
    ds=assignments(code(d,"eval-network-{impl#1}-evaluate"))
    target,arg=sole(ds)
    closure=aliases(ds,arg)
    checked(target.startswith("with_network_buffers::<Score, {closure@") and closure.startswith("{closure@") and
            closure.split("}",1)[0] in target,"wrong main network closure")
    ds=assignments(code(d,"eval-network-with_network_buffers"))
    target,args=sole(ds)
    receiver,callback=args.split(", ")
    # Windows stores NetworkBuffers directly. Other targets realign byte storage in the callback.
    call=re.fullmatch(r"LocalKey::<UnsafeCell<(?:NetworkBuffers|\[u8; \d+\])>>::with::<(\{closure@[^}]+\}), R>",target)
    checked(call and aliases(ds,receiver)=="const eval::network::NETWORK_BUFFERS" and
            aliases(ds,callback)==call[1]+" { f: move _1 }","unknown thread-local evaluation exit")
    ds=simple_body(d,"eval-network-with_network_buffers-{closure#0}","<impl FnOnce(&mut NetworkBuffers) -> R as FnOnce<(&mut NetworkBuffers,)>>::call_once")
    checked("(_1.0: impl FnOnce" in aliases(ds,sole(ds)[1].split(", ")[0]),"wrong eval closure passed to thread-local storage")
    ds=assignments(code(d,"eval-eval_cache-{impl#0}-get_or_insert_with"))
    outputs=[aliases(ds,x) for x in ds["_0"]]
    hit=[x for x in outputs if isinstance(x,str)]
    misses=[x for x in outputs if isinstance(x,tuple)]
    checked(len(hit)==1 and len(misses)==1 and misses[0][0]=="<impl FnOnce() -> Score as FnOnce<()>>::call_once" and
            aliases(ds,misses[0][1].split(", ")[0])=="move _3","invalid eval-cache miss")
    opt=re.fullmatch(r"copy \(\((_\d+) as Some\)\.0: score::Score\)",hit[0]);checked(opt,"wrong eval-cache hit")
    field=re.fullmatch(r"copy \((_\d+)\.0: std::option::Option<score::Score>\)",sole(ds,opt[1]))
    checked(field and sole(ds,field[1])[0]=="EvalCache::probe_for_store","eval cache score provenance")
    ds=assignments(code(d,"eval-eval_cache-{impl#0}-probe_for_store"))
    aggregate=re.fullmatch(r"\((move _\d+), .+\)",sole(ds));checked(aggregate,"wrong probe pair")
    field=re.fullmatch(r"copy \((_\d+)\.0: std::option::Option<score::Score>\)",aliases(ds,aggregate[1]))
    checked(field and sole(ds,field[1])[0]=="select_way","invalid eval-cache lookup origin")
    ds=assignments(code(d,"eval-eval_cache-policy-select_way"))
    outputs=ds["_0"];checked(len(outputs)==2,"wrong eval-cache alternatives")
    options=[aliases(ds,re.fullmatch(r"\((move _\d+), .+\)",x)[1]) for x in outputs]
    checked("Option::<Score>::None" in options,"expected cache miss")
    some=next(x for x in options if x!="Option::<Score>::None")
    score=aliases(ds,re.fullmatch(r"Option::<Score>::Some\((.+)\)",some)[1])
    checked(score[0]=="Score::from_raw","wrong eval-cache score wrapper")
    extend=re.fullmatch(r"(move _\d+) as i32 \(IntToInt\)",aliases(ds,score[1]))
    checked(extend and re.fullmatch(r"move _\d+ as i16 \(IntToInt\)",aliases(ds,extend[1])),"eval cache hit is not sign-extended from i16")

def normal_edges(term):
    if c:=split_call(term):
        return [int(c[4])]
    if term.startswith("switchInt("):
        return [int(n) for n in re.findall(r": bb(\d+)",term)]
    if m:=re.search(r"(?:success: |return: |real: |goto -> )bb(\d+)",term):
        return [int(m[1])]
    if term=="return;":return [-1]
    if term=="unreachable;" or " -> bb" in term:return []
    raise ValueError(f"unrecognized control edge {term}")

def fronts(cfg,start,stops):
    seen=set();todo=[start];out=set()
    while todo:
        pc=todo.pop()
        if pc in stops or pc==-1:out.add(pc);continue
        if pc not in seen:
            seen.add(pc);todo.extend(normal_edges(cfg[pc][-1]))
    return out

def verify_loops(d):
    from check_keys import verify_evaluators,verify_reordering
    verify_evaluators(d);verify_reordering(d)
    wrapper=code(d,IMPL+"evaluate_moves")
    ds=assignments(wrapper)
    funcs={"ordering::<impl MoveList>::evaluate_moves_fast":"_1 _2 _3 _5",
           "ordering::<impl MoveList>::evaluate_moves_by_search::<NT, SS>":"_1 _2 _3 _4 _5 _6 _7"}
    targets=set()
    for pc,ops in wrapper.items():
        c=split_call(ops[-1])
        if c and c[2] in funcs:
            want=funcs[c[2]].split()
            args=c[3].split(", ")
            for a,parameter in zip(args,want,strict=True):
                incoming=aliases(ds,a)
                checked(incoming in ("copy "+parameter,"&(*"+parameter+")","&mut (*"+parameter+")"),"wrong ordering arguments")
            targets.add(pc)
            checked(fronts(wrapper,int(c[4]),set())=={-1},"unexpected dispatcher continuation")
    checked(len(targets)==2 and fronts(wrapper,0,targets)==targets,"ordering can leave without scoring")
    for name,loops in (("evaluate_moves_fast",[(1,4,9,10,10),(13,16,21,22,28)]),
                       ("evaluate_moves_by_search",[(32,35,40,41,107),(57,60,65,66,107)])):
        cfg=code(d,IMPL+name);checked(fronts(cfg,0,{loop[0] for loop in loops})=={loop[0] for loop in loops},"score loop omitted")
        for start,head,member,empty,done in loops:
            checked(fronts(cfg,start,{head})=={head},"iterator constructor bypassed")
            it={1:"_10",13:"_28",32:"_38",57:"_74"}[start]
            ds=assignments(cfg)
            source=aliases(ds,sole(ds,it))
            checked(source[0]=="<std::slice::IterMut<'_, move_list::Move> as IntoIterator>::into_iter","unexpected scoring iterator")
            inner=aliases(ds,source[1])
            checked(inner[0]=="MoveList::iter_mut" and aliases(ds,inner[1])=="&mut (*_1)","not scoring the initialized move list")
            for present in (False,True):
                target=member if present else empty
                symbolic(cfg,head,{target},{it:"iter"},{("discriminant","next"):present})
            checked(fronts(cfg,empty,set())=={-1},"unexpected exhaustion path")

def scalar_scores(d):
    value=code(d,"score-{impl#5}-value")
    checked(value=={0:["_0 = copy (_1.0: i32);","return;"]},"Score::value changed")
    neg=assignments(code(d,"score-{impl#8}-neg"))
    checked(sole(neg)=="Score(move _2)" and sole(neg,"_2")=="Neg(move _3)" and sole(neg,"_3")=="copy (_1.0: i32)",
            "negative Score projection changed")
    fr=assignments(code(d,"score-{impl#5}-from_raw"));q=re.fullmatch(r"Score\((.+)\)",sole(fr))
    checked(q and aliases(fr,q[1])=="copy _1","Score::from_raw changes the number")
    for name,fn,args,want in (("final_score","Board::final_disc_diff",{"_1":"b"},("b",)),
                               ("solve","Board::solve_disc_diff",{"_1":"b","_2":"empty"},("b","empty"))):
        out=symbolic(code(d,"board-{impl#1}-"+name),0,{-1},args,{})
        checked(out["_0"]==("Score::from_disc_diff",(fn,*want)),"terminal scoring projection changed")
    final=simple_body(d,"board-{impl#1}-final_disc_diff","Board::disc_score")
    checked(aliases(final,sole(final)[1])=="&(*_1)","wrong final disc-score board")
    cast=assignments(code(d,"score-{impl#5}-from_disc_diff"))
    checked(sole(cast)=="Score(move _16)" and sole(cast,"_16")=="Shl(move _17, const score::Score::SCALE_BITS)" and
            sole(cast,"_17")==("DiscScore::value","move _18") and sole(cast,"_18")=="copy _1","wrong terminal score scaling")
    base=("Board::disc_score","b")
    total=("DiscScore::value",base)
    both=("Add",total,"empty")
    checked(symbolic(code(d,"board-{impl#1}-disc_score"),0,{-1},{"_1":"b"},{})["_0"]==
            ("DiscScore::new",("Sub",("Mul",("Board::player_count","b"),2),("DiscScore::value","score::DISC_SCORE_MAX"))),
            "raw final difference changed")
    solved=code(d,"board-{impl#1}-solve_disc_diff")
    checked(fronts(solved,0,{10})=={10},"unexpected early terminal score")
    for cmp,expected in ((255,total),(0,both),(1,("Add",both,"empty"))):
        op=("discriminant",("<i32 as Ord>::cmp",both,0))
        checked(symbolic(solved,10,{-1},{"_1":"b","_2":"empty"},{op:cmp})["_0"]==("DiscScore::new",expected),
                "awarded terminal difference changed")

def verify_scores(d):
    def init(stem):return assignments(code(d,stem,"built"))
    for name,n in (("MIN",-64),("MAX",64)):
        ds=init("score-{impl#5}-"+name)
        checked(sole(ds)=="Score(move _1)" and sole(ds,"_1")=="Shl(move _2, const score::Score::SCALE_BITS)" and
                sole(ds,"_2")==("DiscScore::value",f"const score::DISC_SCORE_{name}"),"score bounds changed")
        checked(sole(init("score-DISC_SCORE_"+name))==("DiscScore::new",f"const {n}_i32"),"disc bounds changed")
    checked(sole(init("score-{impl#5}-SCALE_BITS"))=="const 8_i32" and
            sole(init("score-{impl#5}-SCALE"))=="Shl(const 1_i32, const score::Score::SCALE_BITS)","score binary point")
    inf=init("score-{impl#5}-INF");checked(sole(inf)=="Score(move _1)" and sole(inf,"_1")==
        "const core::num::<impl i16>::MAX as i32 (IntToInt)","infinity range")


FILTER = "evaluate|correct_eval|correction_history|select_way|get_or_insert_with|probe_for_store|with_network_buffers|score::|board::"

def endgame_sort_depths(dump):
    name = IMPL + "evaluate_moves"
    cfg = code(dump, name)
    constant = "move_list::ordering::<impl move_list::MoveList>::evaluate_moves::ENDGAME_MIN_SORT_DEPTH"
    expected = {
        0: ["_9 = const <SS as search::strategy::SearchStrategy>::IS_ENDGAME;",
            "switchInt(move _9) -> [0: bb4, otherwise: bb1];"],
        1: ["_10 = copy _4;", f"_12 = const {constant};", "_15 = &((*_2).10: empty_list::EmptyList);",
            "_14 = EmptyList::count(move _15) -> [return: bb2, unwind: bb11];"],
        2: ["_13 = move _14 as usize (IntToInt);", "_16 = Lt(copy _13, const 64_usize);",
            'assert(move _16, "index out of bounds: the length is {} but the index is {}", const 64_usize, copy _13) '
            '-> [success: bb3, unwind: bb11];'],
        3: ["_11 = copy _12[_13];", "_8 = Lt(move _10, move _11);", "goto -> bb5;"],
        5: ["_19 = copy _8;", "switchInt(move _19) -> [0: bb8, otherwise: bb6];"],
    }
    checked(all(cfg[pc] == ops for pc, ops in expected.items()), "endgame ordering depth dispatch")
    for pc, method in [(6, "evaluate_moves_fast"), (8, "evaluate_moves_by_search::<NT, SS>")]:
        checked(split_call(cfg[pc][-1])[2] == f"ordering::<impl MoveList>::{method}", "wrong ordering arm")
    raw = code(dump, name + "-ENDGAME_MIN_SORT_DEPTH", "built")
    row = re.fullmatch(r"_0 = \[(.*)\];", raw[0][0])
    checked(row and raw == {0: [row[0], "return;"]}, "sort-depth table initialization")
    fields = row[1].split(", ")
    checked(len(fields) == 64 and all(re.fullmatch(r"const \d+_u32", n) for n in fields), "sort-depth table")
    return [int(n.split()[1].removesuffix("_u32")) for n in fields]

def verify(dump,repo):
    validate_heuristics(dump)
    verify_scores(dump)
    scalar_scores(dump)
    verify_loops(dump)
    eval_sources(dump)
    return PROGRAM.replace("@DEPTHS@",", ".join(map(str,endgame_sort_depths(dump))))

def render_weights(dump):
    """Render the ordering table and weights shared by the extracted and hand-written models."""
    squares,weights=constants(dump)
    lines=[f"def squareValue : Vector Int 64 := #v[{', '.join(map(str,squares))}]",
           f"def squareMax : Int := {max(squares)}"]
    lines+=[f"def {name} : Int := {value}" for name,value in weights.items()]
    return WEIGHTS_MODULE.replace("@DEFS@","\n".join(lines))

WEIGHTS_MODULE="""/-! Move-ordering table and weights extracted from the rustc MIR of `move_list::ordering` by
`verification/lean/scripts/extract_endgame_order_mir.py`. Regenerate with
`check_endgame_mir.py --write` instead of editing. -/

namespace NeuralReversi.MoveList.OrderWeights

@DEFS@

end NeuralReversi.MoveList.OrderWeights
"""

PROGRAM="""import NeuralReversi.EndgameSearch.MoveOrder
import NeuralReversi.MoveList.Ordering
import NeuralReversi.MoveList.OrderWeights

namespace NeuralReversi.PvsParallel.EndgameOrderMir

open Bitboard.Moves (X86Config)
open PvsBoardTree (Position)
open MoveList.Ordering (sortByKeyDesc swapList)
open MoveList.OrderWeights

/-- Numeric result origins for search ordering; several control paths select or negate another origin. -/
inductive Estimate where
  | network (raw : Int)
  | cacheHit (packed : BitVec 16)
  | terminal (b : Position)
  | finalCount (board : Board.Board)
  | maximum
  | infinity
  | neg (child : Estimate)
  | higher (a b : Estimate)

def Estimate.read : Estimate → Int
  | .network raw => min 16383 (max (-16383) raw)
  | .cacheHit packed => packed.toInt
  | .terminal b => (PvsBoardTree.terminalScore b).val
  | .finalCount b => 256 * (2 * ((Bitboard.count b.player).toNat : Int) - 64)
  | .maximum => 16384
  | .infinity => 32767
  | .neg x => -x.read
  | .higher x y => max x.read y.read

inductive Heuristic where
  | fast
  | skip
  | search (estimate : Estimate)

def minSortDepth : Vector Nat 64 := #v[@DEPTHS@]

def useFast (depth : BitVec 32) (empty : Fin 64) : Bool := decide (depth.toNat < minSortDepth[empty])

def raw (method : Heuristic) (mc : X86Config) (next : Board.Board) (sq : Bitboard.Square) : Int :=
  let st : Int := (Bitboard.cornerStability next.opponent).toNat
  match method with
  | .fast => squareValue[sq] * squareWeight + st * stabilityWeight + (36 - ((Bitboard.cornerWeightedCount (Board.getMoves mc next)).toNat : Int)) * mobilityWeight
  | .skip => -((Bitboard.cornerWeightedCount (Board.getMoves mc next)).toNat : Int)
  | .search score =>
    let both := Board.getMovesAndPotential mc next
    score.read + st * searchStabilityWeight - ((Bitboard.cornerWeightedCount both.1).toNat : Int) * searchMobilityWeight -
      ((Bitboard.cornerWeightedCount both.2).toNat : Int) * searchPotentialWeight

def tt : Int := 1073741824

def actualKey [DecidableEq Move] (priority : Move → Int) (hint : Option Move) (m : Move) : Int :=
  if some m = hint then tt else priority m

/-- A witness for the enumeration of ties by the standard library's unstable sort. -/
def sortMoves {Move : Type} (priority : Move → Int) (tie : Move → Fin 64) (moves : List Move) : List Move :=
  if moves.length ≤ 3 then sortByKeyDesc priority moves
  else sortByKeyDesc (fun m => 64 * priority m + (tie m).val) moves

def ranked (priority : Move → Int) (tie : Move → Fin 64) (length : Nat) (m : Move) : Int :=
  if length ≤ 3 then priority m else 64 * priority m + (tie m).val

def ordinary {Move : Type} [DecidableEq Move] (priority : Move → Int) (tie : Move → Fin 64)
    (presearched : Bool) (hint : Option Move) (moves : List Move) : List Move :=
  if 1 < moves.length - (if presearched then 1 else 0) then sortByKeyDesc (actualKey (ranked priority tie moves.length) hint) moves
  else match moves with
    | [a,b] => if some a = hint then [a,b] else swapList [a,b] 0 1
    | _ => moves

end NeuralReversi.PvsParallel.EndgameOrderMir
"""





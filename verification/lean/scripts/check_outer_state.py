"""Verify the public endgame entry and the writes and comparisons of its line stamps."""

import re
from mir_common import blocks, CALL, GOTO, checked


def eval_body(path, inputs, active=False, empty=False):
    env={f"_{i}":v for i,v in enumerate(inputs,1)}
    cfg=blocks(path.read_text(encoding="utf-8"))
    actions=[]
    pc=0
    def expr(s):
        if s=="const ()":
            return "unit"
        if s=="SearchResult::NoLegalMove":
            return "noLegal"
        if s in ("Selectivity::Level1","Option::<(u32, Selectivity)>::None","Option::<&RootMove>::None"):
            return s
        if m:=re.fullmatch(r"const (\d+)_usize",s):
            return int(m[1])
        if m:=re.fullmatch(r"(?:copy |move |&mut |&)(_\d+)|(?:copy |move |&mut |&)\(\*(_\d+)\)",s):
            return env[m[1] or m[2]]
        if m:=re.fullmatch(r"(?:copy |&mut |&)\(\(\*(_\d+)\)\.(\d+): (.+)\)|(?:copy |&)\((_\d+)\.(\d+): (.+)\)",s):
            base=env[m[1] or m[4]]
            index,ty=(m[2],m[3]) if m[1] else (m[5],m[6])
            return {
                ("task","0","board::Board"):"board",
                ("task","1","probcut::Selectivity"):"mid",
                ("task","2","std::sync::Arc<transposition_table::TranspositionTable>"):"ttArc",
                ("task","4","std::sync::Arc<eval::Eval>"):"evalArc",
                ("task","6","bool"):int(active),
                ("ctx","7","probcut::Selectivity"):"sel",
                ("completed","0","std::vec::Vec<std::option::Option<(u32, probcut::Selectivity)>>"):"lines",
                ("row","1","score::Score"):"previousScore",
            }[(base,index,ty)]
        if m:=re.fullmatch(r"&\(\*\((?:\(\*(_\d+)\)|(_\d+))\.(\d+): &(.+)\)\)",s):
            return {
                ("ctx","1","search::root_move::RootMoves"):"roots",
                ("closure","0","search::root_move::RootMoves"):"roots",
                ("closure","0","std::option::Option<(u32, probcut::Selectivity)>"):"levelStamp",
            }[(env[m[1] or m[2]],m[3],m[4])]
        if m:=re.fullmatch(r"copy \(\*\((_\d+)\.0: &usize\)\)",s):
            checked(env[m[1]]=="closure","wrong clock-step capture")
            return "step"
        if m:=re.fullmatch(r"\(move (_\d+), move (_\d+)\)",s):
            return (env[m[1]],env[m[2]])
        if m:=re.fullmatch(r"Option::<\(u32, Selectivity\)>::Some\(move (_\d+)\)",s):
            checked(env[m[1]]==("depth","sel"),"wrong published stamp")
            return "levelStamp"
        if m:=re.fullmatch(r"Option::<&RootMove>::Some\(move (_\d+)\)",s):
            checked(env[m[1]]=="best","wrong completion")
            return "saved"
        if m:=re.fullmatch(r"(\{closure@[^}]+\}) \{ level: move (_\d+) \}",s):
            checked(env[m[2]]=="levelStamp","wrong equality closure")
            return ("compareAll",m[1])
        if m:=re.fullmatch(r"Eq\((.+), (.+)\)",s):
            checked([expr(m[1]),expr(m[2])]==["count",0],"wrong empty-root guard")
            return int(empty)
        raise ValueError(f"unexpected expression {s}")

    calls={
        "RootMoves::new":(["board"],"roots"),
        "<Arc<TranspositionTable> as Deref>::deref":(["ttArc"],"tt"),
        "<Arc<Eval> as Deref>::deref":(["evalArc"],"eval"),
        "SearchContext::<'_>::new":(["board","mid","roots","tt","eval"],"ctx"),
        "RootMoves::count":(["roots"],"count"),
        "estimate_aspiration_base_score":(["ctx","board"],"base"),
        "SearchTask::pv_count":(["task","count"],"nLines"),
        "CompletedState::new":(["nLines"],"completed"),
        "solve_root":(["ctx","task","thread","base","completed","Selectivity::Level1","last"],"solved"),
        "<Vec<Option<(u32, Selectivity)>> as DerefMut>::deref_mut":(["lines"],"slice"),
        "<Vec<Option<(u32, Selectivity)>> as Deref>::deref":(["lines"],"slice"),
        "core::slice::<impl [Option<(u32, Selectivity)>]>::fill":(["slice","Option::<(u32, Selectivity)>::None"],"unit"),
        "RootMoves::pv_idx":(["roots"],"idx"),
        "<Vec<Option<(u32, Selectivity)>> as IndexMut<usize>>::index_mut":(["lines","idx"],"slot"),
        "core::slice::<impl [Option<(u32, Selectivity)>]>::iter":(["slice"],"iter"),
        "CompletedState::iteration_completed":(["completed","ctx","depth","endgame"],"best"),
        "<Option<(u32, Selectivity)> as PartialEq>::eq":(["lineStamp","levelStamp"],"compared"),
        "RootMoves::get":(["roots","previousIndex"],"previousRow"),
        "EndgameClock::<'_>::should_stop":(["clock","step"],"clockStop"),
        "TimeManager::check_time":(["time"],"timeout"),
    }
    seen=set()
    while pc not in seen:
        seen.add(pc)
        *stmts,end=cfg[pc]
        for ins in stmts:
            if re.fullmatch(r"Storage(?:Live|Dead)\(_\d+\);|FakeRead\(ForLet\(None\), _\d+\);",ins):
                continue
            if m:=re.fullmatch(r"\(\*(_\d+)\) = (.+);",ins):
                checked(env[m[1]]=="slot" and expr(m[2])=="levelStamp","wrong line write")
                actions.append("setStamp")
                continue
            m=re.fullmatch(r"(_\d+) = (.+);",ins)
            checked(m,f"unexpected instruction {ins}")
            env[m[1]]=expr(m[2])
        if m:=re.fullmatch(CALL,end):
            fn=m[2]
            args=[expr(a) for a in m[3].split(", ") if a]
            if c:=re.fullmatch(r"<std::slice::Iter<'_, Option<\(u32, Selectivity\)>> as Iterator>::all::<(\{closure@[^}]+\})>",fn):
                expect=["iter",("compareAll",c[1])]
                result=int(active)
                actions.append("all")
            else:
                expect,result=calls[fn]
            checked(args==expect,f"{fn} expected {expect}, got {args}")
            env[m[1]]=result
            if fn in ("solve_root","CompletedState::iteration_completed","core::slice::<impl [Option<(u32, Selectivity)>]>::fill"):
                actions.append(fn)
            pc=int(m[4])
        elif m:=re.fullmatch(r"switchInt\(move (_\d+)\) -> \[0: bb(\d+), otherwise: bb(\d+)\];",end):
            checked(env[m[1]] in (0,1),"unknown branch")
            pc=int(m[3] if env[m[1]] else m[2])
        elif m:=re.fullmatch(GOTO,end):
            pc=int(m[1] or m[2])
        elif m:=re.fullmatch(r"drop\((_\d+)\) -> \[return: bb(\d+), unwind: bb\d+\];",end):
            checked(env[m[1]] in ("completed","ctx","roots","task","row"),"unhandled destructor")
            pc=int(m[2])
        elif end=="return;":
            return actions,env["_0"]
        else:
            raise ValueError(f"unhandled terminator {end}")
    raise ValueError(f"nonterminating metadata accessor {path}")


def check(dump):
    def path(name):
        return dump/("reversi_core.search-"+name+".SimplifyCfg-initial.after.mir")
    checked(eval_body(path("endgame-search_root"),["task","last","thread"],empty=True)==([],"noLegal"),"no-move entry")
    checked(eval_body(path("endgame-search_root"),["task","last","thread"])==(["solve_root"],"solved"),"direct entry")
    for multi in (False,True):
        checked(eval_body(path("{impl#0}-pv_count"),["task","count"],active=multi)==([],"count" if multi else 1),"PV count")
    checked(eval_body(path("result-{impl#1}-begin_phase"),["completed"])==(["core::slice::<impl [Option<(u32, Selectivity)>]>::fill"],"unit"),"phase start")
    checked(eval_body(path("result-{impl#1}-line_completed-{closure#0}"),["closure","lineStamp"])==
            ([],"compared"),"line-stamp equivalence")
    for all_equal in (False,True):
        actions,result=eval_body(path("result-{impl#1}-line_completed"),["completed","ctx","depth","endgame"],all_equal)
        checked(actions==["setStamp","all"]+(["CompletedState::iteration_completed"] if all_equal else []),"line completion")
        checked(result==("saved" if all_equal else "Option::<&RootMove>::None"),"published reference")
    for idx,arg,result in [(0,"previousIndex","previousRow"),(1,"row","previousScore"),(2,"clock","clockStop"),(3,"time","timeout")]:
        checked(eval_body(path(f"endgame-solve_root-{{closure#{idx}}}"),["closure",arg])==([],result),"ladder callback")


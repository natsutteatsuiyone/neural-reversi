"""Translate receiver/borrow transfers along the native worker and endgame call graph."""

import re
from cache_owner_mir import Body, CLOSURE, THREAD, WORKER
from mir_common import CALL, checked
from cache_owner_queue import helper_queue, main_queue
from cache_owner_frontier import frontier
from extract_cache_boot import boot, record


MID = "search-midgame-"
END = "search-endgame-"
STRATEGY = "search-strategy-"
LAZY = MID+"lazy_smp-"
GET = THREAD+"::endgame_caches"
NODES = {
    WORKER+"main_thread_loop": "main", WORKER+"idle_loop": "idle", WORKER+"run_helper_job": "job",
    WORKER+"search_active_split_point": "active", WORKER+"dispatch_search": "dispatch",
    WORKER+"split": "split", "search-search_root": "root", END+"search_root": "endRoot",
    END+"solve_root": "solve", END+"aspiration_search": "endWindow", MID+"search_root": "midRoot",
    MID+"iterate": "iterate", MID+"aspiration_search": "midWindow",
    MID+"aspiration_search-{closure#0}": "midClosure", MID+"try_probcut": "midProb",
    LAZY+"{impl#3}-run": "helper", LAZY+"{impl#1}-start-{closure#0}-{closure#0}": "helperClosure",
    "search-pvs-search": "pvs", "search-pvs-search_split_point": "splitSearch",
    STRATEGY+"{impl#0}-try_probcut": "midStrategyProb", STRATEGY+"{impl#1}-shallow_search": "boundary",
    END+"null_window_search": "deep", END+"search_move_nws_ec": "deepChild",
    END+"shallow_search": "shallow", END+"shallow_search_moves": "shallowMoves",
    END+"shallow_search_move": "shallowChild",
}


def targets(fn):
    fn = re.sub(r"^(shallow_search(?:_moves?)?)::<(?:5|6|EMPTIES)>$", r"\1", fn)
    exact = {
        THREAD+"::idle_loop":("idle",0), THREAD+"::run_helper_job":("job",0),
        THREAD+"::dispatch_search":("dispatch",0), THREAD+"::split":("split",0),
        THREAD+"::search_active_split_point":("active",0), "search::search_root":("root",1),
        "search::endgame::search_root":("endRoot",2), "midgame::search_root":("midRoot",2),
        "solve_root":("solve",2), "iterate":("iterate",5),
        "search::endgame::aspiration_search":("endWindow",4), "midgame::aspiration_search":("midWindow",6),
        "HelperSearch::run":("helper",1), "midgame::try_probcut":("midProb",6),
        "<EndGameStrategy as SearchStrategy>::shallow_search":("boundary",5),
        "<SS as SearchStrategy>::shallow_search":("boundary",5),
        "<SS as SearchStrategy>::try_probcut":("midStrategyProb",6),
        GET:("access",0), "shallow_search":("shallow",3), "shallow_search_move":("shallowChild",4),
        "shallow_search_moves":("shallowMoves",6),
    }
    if fn.startswith("search::<"):
        return "pvs",5
    if fn.startswith("search_split_point::<"):
        return "splitSearch",3
    if fn.startswith("null_window_search::<"):
        return "deep",3
    if fn.startswith("search_move_nws_ec::<"):
        return "deepChild",5
    if re.fullmatch(r"EndGameCache::<(Deep|Shallow)>::(slot|probe|store)",fn):
        tag = fn.split("::<")[1].split(">")[0].lower()
        return tag+fn.split("::")[-1].capitalize(),0
    if fn.startswith("<dyn for<'a> Fn(&'a Arc<threading::thread::Thread>)") and fn.endswith("::call"):
        return "helperClosure",1
    return exact.get(fn)


def mid_closure(body,dump):
    matches = [(reg,s) for reg,defs in body.defs.items() for s in defs if isinstance(s,str)
               and re.match(CLOSURE+r" \{",s)]
    checked(len(matches)==1,"midgame aspiration closure")
    reg,s=matches[0]
    ty=s.split(" { ")[0]
    members=record(body,reg,ty)
    checked(list(members)==["ctx","board","depth","thread"] and
            body.resolve(members["thread"],{body.arg("thread"):"self"})=="self","aspiration closure capture")
    retry=body.named("retry_midgame::<"+ty+">")
    checked(retry[2][3]=="move "+reg,"closure argument")
    runner=Body(dump,"search-aspiration-retry_midgame")
    callback="<impl FnMut(Score, Score) -> (Score, bool) as FnMut<(Score, Score)>>::call_mut"
    checked(runner.resolve(runner.named(callback)[2][0],{"_4":"callback"})=="callback",
            "retry must invoke supplied closure")


def check_borrow_uses(body):
    arguments = re.findall(r"(_\d+): ([^,)]*)",body.signature)
    locals_ = re.findall(r"^\s*let (?:mut )?(_\d+): (.*);$",body.source,re.M)
    cache_regs = {reg for reg,ty in arguments+locals_ if "EndGameCache" in ty}
    checked(cache_regs,body.stem+": missing cache borrow operands")
    for _,statement in body.ops:
        if not (cache_regs & set(re.findall(r"_\d+",statement))):
            continue
        if statement.startswith(("StorageLive(", "StorageDead(", "FakeRead(", "PlaceMention(", "AscribeUserType(")):
            continue
        call = re.fullmatch(CALL,statement)
        if call:
            target = targets(call[2])
            checked(target is not None,body.stem+": a cache reference escaped to "+call[2])
            if call[1] in cache_regs:
                checked(call[2]==GET,body.stem+": untracked cache borrow producer")
            used = [i for i,operand in enumerate(call[3].split(", "))
                    if cache_regs & set(re.findall(r"_\d+",operand))]
            checked(not used or used==[target[1]],body.stem+": unexpected second reference argument")
            continue
        pattern = (r"(_\d+) = &(?:mut )?(?:\(\*(_\d+)\)|\(\(\*(_\d+)\)\.[01]: "
                   r"search::endgame::cache::EndGameCache<search::endgame::cache::(?:Deep|Shallow)>\));")
        alias=re.fullmatch(pattern,statement)
        checked(alias and alias[1] in cache_regs and (alias[2] or alias[3]) in cache_regs,
                body.stem+": cache reference escaped or aliased a foreign local: "+statement)


def routing(dump,stem):
    body=Body(dump,stem)
    node=NODES[stem]
    if node in ("boundary","deep","deepChild","shallow","shallowMoves","shallowChild"):
        check_borrow_uses(body)
    if node=="midClosure":
        capture="((*_1).3: &std::sync::Arc<search::threading::thread::Thread>)"
        seeds={capture:"thread"}
    elif node in ("deep","deepChild"):
        seeds={body.arg("caches"):"pair",body.arg("cutoff"):"poll"}
    elif node in ("shallow","shallowMoves","shallowChild"):
        seeds={body.arg("sc"):"shallow"}
    else:
        inputs=re.findall(r"(_\d+): (?:&)?Arc<threading::thread::Thread>",body.signature)
        checked(len(inputs)==1,f"{node} thread argument")
        seeds={inputs[0]:"thread"}
    if node=="boundary":
        seeds=body.produced(GET,"pair",seeds)
        start=body.named(GET)[0]
        checked(all(start not in body.reachable(int(m[1])) for x in [body.cfg[start][-1]]
                    if (m:=re.search(r"return: bb(\d+)",x))),"second loan on a single invocation")
    rows=[]
    for pc,fn,args,_ in body.calls:
        target=targets(fn)
        if target:
            to,i=target
            origin=body.resolve(args[i],seeds)
            if to=="helperClosure":
                checked(origin==("one","thread"),"helper callback receiver")
                origin="thread"
            if node=="boundary" and to=="deep" and "IgnoreSplitPointCutoff" not in fn:
                checked(body.resolve(args[4],seeds)=="thread","cutoff poller does not own this borrow")
            if node in ("deep","deepChild") and to in ("deep","deepChild"):
                pollarg=4 if to=="deep" else 6
                checked(body.resolve(args[pollarg],seeds)=="poll","changed cutoff poller")
            transform = {("boundary","pair"):"borrowPair",("boundary","shallow"):"borrowShallow",
                         ("deep","deep"):"deepField",("deepChild","shallow"):"shallowField"}.get((node,origin))
            if transform is None:
                expected={"deep":"pair","deepChild":"pair","shallow":"shallow","shallowChild":"shallow",
                          "shallowMoves":"shallow"}.get(node,"thread")
                checked(origin==expected,f"foreign receiver in {node}: {fn} {origin}")
                transform="forward"
            rows.append((pc,to,transform))
        elif node=="midWindow" and fn.startswith("retry_midgame::<"):
            mid_closure(body,dump)
            rows.append((pc,"midClosure","forward"))
        elif node in ("solve","iterate") and fn.startswith("<dyn Fn(SearchProgress)") and fn.endswith("::call"):
            rows.append((pc,"progress","callback"))
    checked(rows,f"no modeled route {node}")
    return rows


def render(dump,repo):
    main_queue(dump)
    helper_queue(dump)
    made=boot(dump)
    rows=[]
    for stem,source in NODES.items():
        for ordinal,(_,target,flow) in enumerate(routing(dump,stem)):
            rows.append(f"  ⟨.{source}, {ordinal}, .{target}, .{flow}⟩")
    return "\n".join(["module", "public import NeuralReversi.EndgameSearch.CacheOwner.Syntax","",
        "@[expose] public section", "", "namespace NeuralReversi.EndgameSearch.CacheOwner","",*made,
        "def transfers : List Edge := [",",\n".join(rows)+"]","",
        "end NeuralReversi.EndgameSearch.CacheOwner",""])


def check_full_frontier(dump):
    frontier(dump, targets, NODES)

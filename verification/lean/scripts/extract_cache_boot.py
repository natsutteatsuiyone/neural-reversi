"""Check storage allocation, Arc registration, OS-thread capture and loop receiver."""

import re

from cache_owner_mir import Body, CLOSURE, POOL, THREAD, WORKER
from mir_common import checked


def record(body, target, prefix):
    defs = body.defs[target]
    checked(len(defs) == 1 and isinstance(defs[0],str) and defs[0].startswith(prefix + " { ") and
            defs[0].endswith(" }"), f"{body.stem}: wrong {prefix} record")
    return dict(field.split(": ",1) for field in defs[0][len(prefix)+3:-2].split(", "))


def normal_paths(body, selected):
    outcomes = set()
    sites = {n:fn for n,fn,args,out in body.calls if fn in selected}
    def walk(pc, path, events):
        if pc in path:
            outcomes.add(("loop",events))
            return
        if pc in sites:
            events += (sites[pc],)
        end = body.cfg[pc][-1]
        if end == "return;":
            outcomes.add(("return",events))
            return
        if end.startswith(("falseEdge","falseUnwind")):
            targets = re.findall(r"real: bb(\d+)",end)
        else:
            targets = re.findall(r"(?:return: |success: |otherwise: |\d+: |goto -> )bb(\d+)",end)
        for t in targets:
            if int(t) in body.cfg:
                walk(int(t),path|{pc},events)
    walk(0,set(),())
    return outcomes


def sequence(body, calls):
    checked(normal_paths(body,calls) == {("return",tuple(calls))}, f"{body.stem}: event order")


def construction(dump):
    worker = Body(dump,WORKER+"new")
    pair = "EndGameCaches::for_thread_count"
    storage = "SyncUnsafeCell::<EndGameCaches>::new"
    sequence(worker,[pair,storage])
    size=worker.arg("pool_size")
    _,_,inputs,_=worker.named(pair)
    checked(worker.resolve(inputs[0],{size:"count"})=="count","cache allocator size")
    alloc=worker.produced(pair,"newPair",{})
    _,_,inputs,_=worker.named(storage)
    checked(worker.resolve(inputs[0],alloc)=="newPair","cell must contain fresh pair")
    cell=worker.produced(storage,"newCell",{})
    data=record(worker,"_0","threading::thread::Thread")
    resident = worker.resolve(data["endgame_caches"],cell)
    checked(list(data)[13]=="endgame_caches" and resident=="newCell","resident cache")
    idx=worker.arg("idx")
    checked(worker.resolve(data["idx"],{idx:"idx"})=="idx","wrong stored worker index")

    access=Body(dump,WORKER+"endgame_caches")
    getter="SyncUnsafeCell::<EndGameCaches>::get"
    sequence(access,[getter])
    _,_,inputs,_=access.named(getter)
    projection = "((*_1).13: std::cell::SyncUnsafeCell<search::endgame::EndGameCaches>)"
    checked(access.resolve(inputs[0],{projection:"own"})=="own","cache getter receiver field")
    raw=access.produced(getter,"own",{})
    checked(access.resolve("_0",raw)=="own","cache getter returned a different reference")

    add=Body(dump,POOL+"add_thread")
    new=THREAD+"::new"
    arc=f"Arc::<{THREAD}>::new"
    clone=f"<Arc<{THREAD}> as Clone>::clone"
    push=f"Vec::<Arc<{THREAD}>>::push"
    sequence(add,[new,arc,clone,push])
    _,_,args,_=add.named(new)
    checked(add.resolve(args[0],{add.arg("idx"):"idx"})=="idx","registration index")
    checked(add.resolve(args[4],{"((*_1).2: usize)":"count"})=="count","registration pool size")
    seed=add.produced(new,"freshThread",{})
    checked(add.resolve(add.named(arc)[2][0],seed)=="freshThread","Arc must wrap fresh Thread")
    seed=add.produced(arc,"created",{})
    destination="((*_1).0: std::vec::Vec<std::sync::Arc<search::threading::thread::Thread>>)"
    checked(add.resolve(add.named(push)[2][0],{destination:"registry"})=="registry","thread list")
    registered=add.resolve(add.named(push)[2][1],seed)
    returned=add.resolve("_0",seed)
    checked(registered=="created" and returned=="created","registered/returned object")

    init=Body(dump,POOL+"init")
    sequence(init,["ThreadPool::create_main_thread","ThreadPool::create_worker_threads",
                   "ThreadPool::wait_for_threads_ready"])
    for name in ["ThreadPool::create_main_thread","ThreadPool::create_worker_threads"]:
        _,_,args,_=init.named(name)
        checked(init.resolve(args[0],{"_1":"pool"})=="pool","init dispatch pool")

    pool=Body(dump,POOL+"new-{closure#0}")
    public=Body(dump,POOL+"new")
    cyclic=[fn for _,fn,_,_ in public.calls if fn.startswith("Arc::<ThreadPool>::new_cyclic::<")]
    checked(len(cyclic)==1,"pool allocation constructor")
    constructor=cyclic[0]
    sequence(public,[constructor])
    closure=constructor.removeprefix("Arc::<ThreadPool>::new_cyclic::<").removesuffix(">")
    checked(closure in pool.signature,"pool initialization closure")
    (arg,)=public.named(constructor)[2]
    captured=record(public,arg.split()[1],closure)
    checked(list(captured)==["n_threads"] and public.resolve(captured["n_threads"],{"_1":"size"})=="size",
            "pool size closure capture")
    checked(public.resolve("_0",public.produced(constructor,"pool",{}))=="pool","returned pool")
    new_vec=f"Vec::<Arc<{THREAD}>>::new"
    sequence(pool,[new_vec,"ThreadPool::init"])
    candidates=[reg for reg,defs in pool.defs.items() if any(isinstance(d,str) and d.startswith("ThreadPool {") for d in defs)]
    checked(len(candidates)==1,"constructor body")
    obj,=candidates
    entry=record(pool,obj,"ThreadPool")
    checked(pool.resolve(entry["size"],{"(_1.0: &usize)":"size"})=="size","pool size field")
    checked(pool.resolve(entry["threads"],pool.produced(new_vec,"empty",{}))=="empty","thread list must start empty")
    src=("checked",pool.defs[obj][0],"newPool")
    checked(pool.resolve(pool.named("ThreadPool::init")[2][0],{obj:src})=="newPool" and
            pool.resolve("_0",{obj:src})=="newPool","initialized pool identity")
    return [{"newCell":".fresh","created":".newArc"}[item] for item in [resident,registered,returned]]


def starts(dump,kind):
    main = kind=="main"
    stem=POOL+("create_main_thread" if main else "create_worker_threads")
    creator=Body(dump,stem)
    new="ThreadPool::add_thread"
    _,_,args,_=creator.named(new)
    checked(creator.resolve(args[0],{"_1":"pool"})=="pool","add_thread receiver")
    if main:
        checked(creator.resolve(args[1],{})=="0_usize","main thread ordinal")
    else:
        rr="std::ops::Range::<usize> { start: const 1_usize, end: move _5 }"
        checked(creator.defs["_4"]==[rr] and creator.resolve("_5",{"((*_1).2: usize)":"count"})=="count",
                "worker range")
        some="((_9 as Some).0: usize)"
        checked(creator.resolve(args[1],{some:"i"})=="i","worker ordinal")
        next_call="<std::ops::Range<usize> as Iterator>::next"
        _,_,args_n,iter_result=creator.named(next_call)
        into="<std::ops::Range<usize> as IntoIterator>::into_iter"
        checked(iter_result=="_9" and creator.resolve(args_n[0],creator.produced(into,"range",{}))=="range"
                and creator.named(into)[2]==["move _4"],"worker iterator origin")
    spawn=[c for c in creator.calls if c[1].startswith("spawn::<")]
    checked(len(spawn)==1,"OS thread spawn count")
    _,fn,closure_args,_=spawn[0]
    closure_id=re.fullmatch(r"spawn::<("+CLOSURE+r"), \(\)>",fn)[1]
    _,closure_reg=closure_args[0].split()
    captures=record(creator,closure_reg,closure_id)
    checked(list(captures)==(["thread","receiver"] if main else ["thread"]),"closure capture shape")
    captured=creator.resolve(captures["thread"],creator.produced(new,"returned",{}))
    checked(captured=="returned","captured thread")
    if main:
        checked(creator.resolve(captures["receiver"],{"_3":"rx"})=="rx","main inbox capture")
        sequence(creator,[new,fn])
    else:
        checked(normal_paths(creator,[new,fn])=={("return",()),("loop",(new,fn))},"worker spawn loop")
    closure=Body(dump,stem+"-{closure#0}")
    call=THREAD+("::main_thread_loop" if main else "::idle_loop")
    sequence(closure,[call])
    checked(closure_id in closure.signature,"invoked closure mismatched captured type")
    receiver="(_1.0: std::sync::Arc<search::threading::thread::Thread>)"
    invoked=closure.resolve(closure.named(call)[2][0],{receiver:"captured"})
    checked(invoked=="captured","boot receiver")
    return ["."+captured,"."+invoked,".main" if main else ".worker"]


def boot(dump):
    create=construction(dump)
    return ["def bootMain : Boot := ⟨"+", ".join(create+starts(dump,"main"))+"⟩",
            "def bootWorker : Boot := ⟨"+", ".join(create+starts(dump,"worker"))+"⟩",""]

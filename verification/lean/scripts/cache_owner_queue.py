"""Trace the main inbox and helper mailbox from construction to invocation."""

import re

from cache_owner_mir import Body, CLOSURE, POOL, THREAD, WORKER
from mir_common import checked
from extract_cache_boot import record


JOB = "Arc<dyn for<'a> Fn(&'a Arc<threading::thread::Thread>) + Send + Sync>"
FULL_JOB = "std::sync::Arc<dyn for<'a> std::ops::Fn(&'a std::sync::Arc<search::threading::thread::Thread>) + std::marker::Send + std::marker::Sync>"
TASK = "search::SearchTask"
MESSAGE = "search::threading::pool::Message"
RESULT = "search::result::SearchResult"
EXECUTE = "search-{impl#1}-execute_search"
LAUNCH = "search-midgame-lazy_smp-{impl#1}-start-{closure#0}"


def one_definition(body, pattern):
    found = [(reg, expr, match) for reg, definitions in body.defs.items() for expr in definitions
             if isinstance(expr, str) and (match := re.fullmatch(pattern, expr))]
    checked(len(found) == 1, f"{body.stem}: expected single {pattern}")
    reg, expr, match = found[0]
    checked(body.defs[reg] == [expr], f"conflicting {reg} in {body.stem}")
    return reg, expr, match


def check_origin(body, operand, seeds, expected):
    checked(body.resolve(operand, seeds) == expected, f"{body.stem}: expected {expected} at {operand}")


def passed(body, consumer, i, seeds, expected):
    check_origin(body, body.named(consumer)[2][i], seeds, expected)


def channel(body, short, full):
    fn = f"std::sync::mpsc::channel::<{short}>"
    result = body.named(fn)[3]
    checked(body.defs[result] == [("call",fn,[])], body.stem + ": channel result reassigned")
    return {f"({result}.{i}: std::sync::mpsc::{ty}<{full}>)": ty for i,ty in enumerate(("Sender","Receiver"))}


def main_queue(dump):
    new = Body(dump, POOL+"new-{closure#0}")
    ends = channel(new,"Message", MESSAGE)
    pool, _, _ = one_definition(new, r"ThreadPool \{ .+ \}")
    fields = record(new, pool, "ThreadPool")
    check_origin(new, fields["sender"], ends, "Sender")
    passed(new, "ThreadPool::init", 2, ends, "Receiver")

    init = Body(dump, POOL+"init")
    passed(init, "ThreadPool::create_main_thread", 2, {"_3":"Receiver"}, "Receiver")
    start = Body(dump, POOL+"create_main_thread-{closure#0}")
    recv = "(_1.1: std::sync::mpsc::Receiver<"+MESSAGE+">)"
    passed(start, THREAD+"::main_thread_loop", 1, {recv:"Receiver"}, "Receiver")

    task = Body(dump, POOL+"start_thinking")
    pair = channel(task, "SearchResult", RESULT)
    send = "std::sync::mpsc::Sender::<Message>::send"
    field = "((*_1).4: std::sync::mpsc::Sender<"+MESSAGE+">)"
    passed(task, send, 0, {field:"sender"}, "sender")
    reg, expr, message = one_definition(task, r"Message::StartThinking\((move _\d+), (move _\d+), (move _\d+)\)")
    check_origin(task, message[1], {"_2":"task"}, "task")
    check_origin(task, message[3], pair, "Sender")
    passed(task, send, 1, {reg:("checked",expr,"message")}, "message")
    check_origin(task, "_0", pair, "Receiver")

    api = Body(dump, EXECUTE)
    request = "ThreadPool::start_thinking"
    recv_result = "std::sync::mpsc::Receiver::<SearchResult>::recv"
    pool_field = "((*_1).1: std::sync::Arc<search::threading::pool::ThreadPool>)"
    passed(api, request, 0, {pool_field:"pool"}, "pool")
    passed(api, request, 1, {"_2":"task"}, "task")
    passed(api, recv_result, 0, api.produced(request,"reply",{}), "reply")

    worker = Body(dump, WORKER+"main_thread_loop")
    receive = "std::sync::mpsc::Receiver::<Message>::recv"
    passed(worker, receive, 0, {"_2":"inbox"}, "inbox")
    tag = worker.named(receive)[3]
    checked(worker.defs[tag] == [("call",receive,worker.named(receive)[2])], "message overwritten")
    case = f"((({tag} as Ok).0: {MESSAGE}) as StartThinking)"
    passed(worker, "search::search_root", 0, {"("+case+".0: "+TASK+")":"task"}, "task")
    reply = "std::sync::mpsc::Sender::<SearchResult>::send"
    passed(worker, reply, 0, {"("+case+".2: std::sync::mpsc::Sender<"+RESULT+">)":"reply"}, "reply")
    passed(worker, reply, 1, worker.produced("search::search_root","score",{}), "score")


def erase(body, seeds):
    pat = r"(move _\d+) as "+re.escape(FULL_JOB)+r" \(PointerCoercion\(Unsize, Implicit\)\)"
    reg, expr, m = one_definition(body, pat)
    check_origin(body, m[1], seeds, "job")
    return {reg:("checked",expr,"job")}


def mailbox(body):
    field_ty = "std::sync::Mutex<std::option::Option<"+FULL_JOB+">>"
    reg, expr, field = one_definition(body,r"&\(\(\*(_\d+)\)\.6: "+re.escape(field_ty)+r"\)")
    check_origin(body, field[1], {"_1":"self"}, "self")
    seeds = {reg:("checked",expr,"mailbox")}
    lock = "std::sync::Mutex::<Option<"+JOB+">>::lock"
    unwrap = [fn for _,fn,_,_ in body.calls if fn.startswith("Result::<std::sync::MutexGuard<'_, Option<"+JOB) and fn.endswith(">::unwrap")]
    checked(len(unwrap)==1,"helper lock unwrap")
    deref = "<std::sync::MutexGuard<'_, Option<"+JOB+">> as DerefMut>::deref_mut"
    for fn in (lock, unwrap[0], deref):
        passed(body,fn,0,seeds,"mailbox")
        seeds=body.produced(fn,"mailbox",{})
    return seeds


def helper_queue(dump):
    create = Body(dump, LAUNCH)
    alloc = [fn for _,fn,_,_ in create.calls if re.fullmatch(r"Arc::<"+CLOSURE+r">::new",fn)]
    checked(len(alloc)==1,"helper closure allocation")
    closure = alloc[0][len("Arc::<"):-len(">::new")]
    callback = Body(dump, LAUNCH+"-{closure#0}")
    checked(closure in callback.signature,"unexpected callback type")
    _,_,args,_ = create.named(alloc[0])
    (operand,) = args
    checked(list(record(create, operand.split()[1],closure)) == ["search"],"helper closure captures worker")
    passed(create,"ThreadPool::start_helpers",1,erase(create,create.produced(alloc[0],"job",{})),"job")

    launch = Body(dump, POOL+"start_helpers")
    idx = "<Vec<Arc<"+THREAD+">> as Index<std::ops::RangeFrom<usize>>>::index"
    passed(launch,idx,0,{"((*_1).0: std::vec::Vec<std::sync::Arc<search::threading::thread::Thread>>)":"registered"},"registered")
    reg,expr,_ = one_definition(launch,r"std::ops::RangeFrom::<usize> \{ start: const 1_usize \}")
    passed(launch,idx,1,{reg:("checked",expr,"workers")},"workers")
    to_iter = "<&[Arc<"+THREAD+">] as IntoIterator>::into_iter"
    step = "<std::slice::Iter<'_, Arc<"+THREAD+">> as Iterator>::next"
    passed(launch,to_iter,0,launch.produced(idx,"slice",{}),"slice")
    passed(launch,step,0,launch.produced(to_iter,"iter",{}),"iter")
    some = "(("+launch.named(step)[3]+" as Some).0: &std::sync::Arc<search::threading::thread::Thread>)"
    passed(launch,THREAD+"::book_helper_job",0,{some:"worker"},"worker")
    clone = "<"+JOB+" as Clone>::clone"
    passed(launch,clone,0,{"_2":"job"},"job")
    passed(launch,THREAD+"::book_helper_job",1,erase(launch,launch.produced(clone,"job",{})),"job")

    book = Body(dump, WORKER+"book_helper_job")
    seeds = erase(book,{"_2":"job"})
    reg,expr,m = one_definition(book,re.escape("Option::<"+JOB+">::Some")+r"\((move _\d+)\)")
    check_origin(book,m[1],seeds,"job")
    stores = [match for _,s in book.ops if (match:=re.fullmatch(r"\(\*(_\d+)\) = (move _\d+);",s))]
    checked(len(stores)==1,"helper mailbox writes")
    check_origin(book,stores[0][1],mailbox(book),"mailbox")
    check_origin(book,stores[0][2],{reg:("checked",expr,"present")},"present")

    idle = Body(dump, WORKER+"idle_loop")
    take = "Option::<"+JOB+">::take"
    passed(idle,take,0,mailbox(idle),"mailbox")
    some = "(("+idle.named(take)[3]+" as Some).0: "+FULL_JOB+")"
    passed(idle,THREAD+"::run_helper_job",1,{some:"job"},"job")
    run = Body(dump, WORKER+"run_helper_job")
    invoke = [fn for _,fn,_,_ in run.calls if fn.endswith("::call")]
    checked(len(invoke)==1,"helper invocation")
    deref = "<"+JOB+" as Deref>::deref"
    passed(run,deref,0,{"_2":"job"},"job")
    passed(run,invoke[0],0,run.produced(deref,"job",{}),"job")

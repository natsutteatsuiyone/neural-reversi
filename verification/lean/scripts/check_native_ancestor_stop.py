"""Capture a nested physical split owner discarding its joined score after an ancestor cutoff."""

import argparse
from collections import defaultdict
from pathlib import Path
import re
import subprocess
import tempfile

from check_native_split_owner import instrument as instrument_owner
from check_serial_calls_trace import FIXTURES, LEAN, REPO, capture, replace_once
from normalize_pvs_trace import require


def instrument(src):
    instrument_owner(src)
    pvs = src / "search/pvs.rs"
    s = pvs.read_text() + """
thread_local! {
    pub(super) static POST_JOIN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    pub(super) static WORKER_POLL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static TRACE_ACTOR: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}
"""
    anchor = "            let stopped = thread.should_stop_after_split();"
    s = replace_once(s, anchor, "            POST_JOIN.set(split_id);\n" + anchor + "\n            POST_JOIN.set(0);")
    anchor = "            if stopped {\n                return Score::ZERO;"
    s = replace_once(s, anchor, """            if stopped {
                let selected = Score::ZERO;
                println!("N chosen {} {} {} {}",owner_tick(),thread.idx(),split_id,selected.value());
                return selected;""")
    anchor = "                    thread.mark_split_point_cutoff(sp);"
    s = replace_once(s, anchor, """                    println!("N cut {} {} {addr} {idx} {} {} {} {}",owner_tick(),thread.idx(),mv.sq.index(),
                        sp.trace_flag_addr(),score.value(),beta.value());
""" + anchor)
    anchor = "        let _guard = split_point.lock();\n\n        // Abort check\n        if thread.should_stop() {"
    s = replace_once(s,anchor,"""        let _guard = split_point.lock();
        // Abort check
        WORKER_POLL.set(true);
        let stopped = thread.should_stop();
        WORKER_POLL.set(false);
        if stopped {""")
    pvs.write_text(s)
    point = src / "search/threading/split_point.rs"
    s = point.read_text()
    anchor = "    /// Returns the cutoff flag."
    point.write_text(replace_once(s,anchor,"""    pub(in crate::search) fn trace_flag_addr(&self) -> usize {
        self.cutoff.as_ptr() as usize
    }

""" + anchor))
    th = src / "search/threading/thread.rs"
    s = th.read_text()
    anchor = "            sp_state.clear_cutoff();"
    s = replace_once(s,anchor,anchor+"""
            let parent = unsafe { sp.task() }.parent_split_point.as_ref().map_or(0,|p| Arc::as_ptr(p) as usize);
            println!("N init {} {} {} {} {} {parent}",crate::search::pvs::owner_tick(),self.idx,
                crate::search::pvs::OWNER_SPLIT.get(),Arc::as_ptr(sp) as usize,sp_state.trace_flag_addr());""")
    anchor = "            *self.active_split_point_mut() = unsafe { sp.task() }.parent_split_point.clone();"
    s = replace_once(s,anchor,anchor+"""
            let parent = self.active_split_point().as_ref().map_or(0,|p| Arc::as_ptr(p) as usize);
            println!("N restore {} {} {} {} {parent}",crate::search::pvs::owner_tick(),self.idx,
                crate::search::pvs::OWNER_SPLIT.get(),Arc::as_ptr(sp) as usize);""")
    anchor = "        if sp_state.mark_cutoff() {"
    s = replace_once(s,anchor,"""        let first = sp_state.mark_cutoff();
        println!("N mark {} {} {} {}",crate::search::pvs::owner_tick(),self.idx,sp_state.trace_flag_addr(),(!first) as u8);
        if first {""")
    anchor = "            if sp_state.cutoff() {\n                return true;"
    s = replace_once(s,anchor,"""            let found = sp_state.cutoff();
            let post = crate::search::pvs::POST_JOIN.get();
            if post != 0 {
                println!("N flag {} {} {post} {} {} {}",crate::search::pvs::owner_tick(),self.idx,
                    Arc::as_ptr(sp) as usize,sp_state.trace_flag_addr(),found as u8);
            }
            if found {
                return true;""")
    anchor = "            current = unsafe { sp.task() }.parent_split_point.as_ref();"
    s = replace_once(s,anchor,anchor+"""
            if post != 0 {
                println!("N link {} {} {post} {} {}",crate::search::pvs::owner_tick(),self.idx,
                    Arc::as_ptr(sp) as usize,current.map_or(0,|p| Arc::as_ptr(p) as usize));
            }""")
    anchor = "    pub(in crate::search) fn should_stop_after_split(&self) -> bool {"
    s = replace_once(s,anchor,anchor+"""
        let id = crate::search::pvs::POST_JOIN.get();
        let top = self.active_split_point().as_ref().map_or(0,|p| Arc::as_ptr(p) as usize);
        println!("N poll {} {} {id} {top}",crate::search::pvs::owner_tick(),self.idx);""")
    anchor = "        if self.cutoff_occurred() {\n            self.local_chain_cutoff.store(true, Ordering::Relaxed);"
    s = replace_once(s,anchor,anchor+"""
            println!("N latch {} {} {id}",crate::search::pvs::owner_tick(),self.idx);""")
    anchor = "    pub(super) fn idle_loop(self: &Arc<Self>) {"
    s = replace_once(s,anchor,anchor+"\n        crate::search::pvs::TRACE_ACTOR.set(self.idx);")
    anchor = "        let task = SplitPointTask::new(&req, self.active_split_point().clone(), ctx);"
    s = replace_once(s,anchor,anchor+"""
        crate::search::pvs::TRACE_ACTOR.set(self.idx);
        let fa = sp.state().trace_flag_addr();
        let g = crate::search::pvs::OWNER_SPLIT.get();
        let copied = task.parent_split_point.as_ref().map_or(0,|p| p.state().trace_flag_addr());
        println!("T copy {} {} {g} {fa} {copied}",crate::search::pvs::owner_tick(),self.idx);""")
    anchor = "            sp.set_task(task);"
    s = replace_once(s,anchor,anchor+"""
            let installed = unsafe {sp.task()}.parent_split_point.as_ref().map_or(0,|p| p.state().trace_flag_addr());
            println!("T install {} {} {g} {fa} {installed}",crate::search::pvs::owner_tick(),self.idx);""")
    anchor = "            *self.active_split_point_mut() = Some(assignment);"
    s = replace_once(s,anchor,anchor+"""
            println!("T active {} {} {g} {fa}",crate::search::pvs::owner_tick(),self.idx);""")
    anchor = "        sp_state.helpers_mask.set(self.idx);\n        *self.active_split_point_mut() = Some(assignment);"
    s = replace_once(s,anchor,anchor+"""
        println!("T book {} {} {} {}",crate::search::pvs::owner_tick(),crate::search::pvs::TRACE_ACTOR.get(),
            self.idx,sp_state.trace_flag_addr());""")
    anchor = "        let sp = self.active_split_point_mut().take();"
    s = replace_once(s,anchor,anchor+"""
        if let Some(old) = &sp {
            println!("T detach {} {} {}",crate::search::pvs::owner_tick(),self.idx,old.state().trace_flag_addr());
        }""")
    anchor = "        sp.clear_task();"
    s = replace_once(s,anchor,anchor+"""
        println!("T clearTask {} {} {} {}",crate::search::pvs::owner_tick(),self.idx,
            crate::search::pvs::OWNER_SPLIT.get(),sp.state().trace_flag_addr());""")
    anchor = "    pub(in crate::search) fn split_point_cutoff_occurred(&self) -> bool {"
    s = replace_once(s,anchor,anchor+"""
        if crate::search::pvs::WORKER_POLL.get() {
            let active = self.active_split_point().as_ref().map_or(0,|p| p.state().trace_flag_addr());
            println!("T poll {} {} {active}",crate::search::pvs::owner_tick(),self.idx);
        }""")
    anchor = "        if self.cutoff_occurred() {\n            self.local_chain_cutoff.store(true, Ordering::Relaxed);"
    s = replace_once(s,anchor,"""        let walked = self.cutoff_occurred();
        println!("T endWalk {} {} {id} {}",crate::search::pvs::owner_tick(),self.idx,walked as u8);
        if walked {
            self.local_chain_cutoff.store(true, Ordering::Relaxed);""")
    anchor = "            self.cutoff_epoch.fetch_add(1, Ordering::Release);"
    s = replace_once(s,anchor,"""            let prev = self.cutoff_epoch.fetch_add(1, Ordering::Release);
            println!("T increment {} {} {} {prev}",crate::search::pvs::owner_tick(),self.idx,self.cutoff_epoch.as_ptr() as usize);""")
    anchor = """        self.local_seen_cutoff_epoch.store(
            self.cutoff_epoch.load(Ordering::Relaxed).wrapping_sub(1),
            Ordering::Relaxed,
        );"""
    s = replace_once(s,anchor,"""        let epoch = self.cutoff_epoch.load(Ordering::Relaxed);
        let loaded = crate::search::pvs::owner_tick();
        self.local_seen_cutoff_epoch.store(epoch.wrapping_sub(1),Ordering::Relaxed);
        println!("T reset {} {} {} {epoch} {loaded}",crate::search::pvs::owner_tick(),self.idx,self.cutoff_epoch.as_ptr() as usize);""")
    th.write_text(s)


def observed_views(rows, voter, post, read, setter):
    """Independently follow every assignment and retirement up to the sampled window."""
    pointers={r[5]:r[6] for r in rows if r[:2]==["N","init"]}
    marks={r[4]:(r[3],r[2]) for r in rows if r[:2]==["O","mark"]}
    slots,active,reset,views={},{},{},{}
    def head(path):
        return path[0][0] if path else 0
    for row in rows:
        tag,op,t,w,*a=row
        here=active.get(w,())
        if t in (voter[2],post[2],read[2]):
            observed = a[0] if tag=="T" else pointers.get(a[1],0)
            require(head(here)==observed,"raw active pointer does not match the complete assignment stream")
            views[t]=(here,reset.get(w))
        if t==setter[2]:
            require(here == slots[a[0]],"cutoff reporter was not assigned to the flagged generation")
            views[t]=here[0]
        if t>=max(read[2],setter[2]):
            break
        if tag=="T":
            if op=="copy":
                require(head(here)==a[2],"copied active pointer differs from current worker")
            elif op=="install":
                g,flag,parent=a
                require(flag not in slots and head(here)==parent,"install changed the captured parent")
                slots[flag]=((flag,*marks[g]),*here)
            elif op=="active":
                require(here==slots[a[1]][1:],"owner activated under another parent")
                active[w]=slots[a[1]]
            elif op=="book":
                helper,flag=a
                require(not active.get(helper),"booked helper still has an active task")
                active[helper]=slots[flag]
            elif op=="detach":
                require(head(here)==a[0],"detached a different physical split")
                active[w]=()
            elif op=="clearTask":
                del slots[a[1]]
            elif op=="reset":
                reset[w]=row
        elif tag=="N" and op=="restore":
            _,child,parent=a
            chain=slots[pointers[child]][1:]
            require(not here and head(chain)==(pointers[parent] if parent else 0),"restore does not match retained task")
            active[w]=chain
    return views


def shape(rows,d,clear,swap,after,parent_mark):
    child=d["init"]
    owner = child[3]
    chain = [clear]
    while chain[0][7] != 0:
        here = chain[0]
        gen = next((r for r in reversed(rows) if r[:2] == ["N","init"] and r[5] == here[7] and r[2]<here[2]),None)
        if gen is None or any(gen[6]==r[6] for r in chain):
            return None
        chain.insert(0,gen)
    earliest = next(r for r in rows if r[:2] == ["T","copy"] and r[4] == chain[0][4])
    page = [r for r in rows if earliest[2] <= r[2] <= d["returned"][2]]
    def one(phase, gen):
        hits=[r for r in page if r[:2] == ["T",phase] and r[4] == gen]
        return hits[0] if len(hits)==1 else None
    def setup(init,mark):
        g = init[4]
        copied,installed,active = (one(op,g) for op in ("copy","install","active"))
        if not all((copied,installed,active)):
            return None
        require(copied[5]==installed[5]==active[5]==init[6] and copied[6]==installed[6],
                "installed task differs from the native parent copy")
        return [copied,mark,installed,init,active]
    setups, bridges, active = [], [], {}
    for init in [*chain,child]:
        mark = next(r for r in page if r[:2] == ["O","mark"] and r[4] == init[4])
        sequence = setup(init,mark)
        if not sequence:
            return None
        w,expected = init[3],sequence[0][6]
        if active.get(w,0) != expected:
            if w in active:
                return None
            link = next((r for r in reversed(page) if r[:2] == ["T","book"] and r[4]==w and
                         r[5]==expected and setups[-1][-1][2]<r[2]<sequence[0][2]),None)
            if not link or link[3]>3:
                return None
            bridges.append(link)
        active[w]=init[6]
        setups.append(sequence)
    parent,nested = setups[-2:]
    require(setups[0][0][6]==0 and nested[0][6]==clear[6],"wrong parent field at initialization")
    childflag=child[6]
    polls=[r for r in page if r[:2]==["T","poll"] and r[4]==childflag and child[2]<r[2]<d["restore"][2]]
    # Prefer an actual second worker booked into the child. This vote can only
    # come from should_stop() for that ticket, where the active flag is sampled.
    for voter in sorted(polls,key=lambda r:r[3] == owner):
        worker=voter[3]
        writer=swap[3]
        if worker == writer:
            continue
        books=[r for r in page if r[:2]==["T","book"] and r[4]==worker and r[5]==childflag and
               child[2]<r[2]<voter[2]]
        if worker != owner and (worker in active or not books or books[-1][3] > 3):
            continue
        entry = books[-1] if worker != owner else nested[-1]
        def detaching(w,start):
            return next((r for r in page if r[:2]==["T","detach"] and r[3]==w and r[4]==childflag and
                         start<r[2]<d["restore"][2]),None)
        left=detaching(owner,nested[-1][2])
        helper_left=detaching(worker,voter[2])
        if left is None or helper_left is None or (worker==owner and left!=helper_left):
            continue
        if any(r[:2]==["T","detach"] and r[3]==worker and entry[2]<r[2]<voter[2] for r in page):
            continue
        closed = one("clearTask",child[4])
        finished = one("endWalk",child[4])
        if closed is None or finished is None:
            continue
        require(closed[5]==childflag and finished[-1]==1,"different task or incomplete true ancestor walk")
        kept = [*(r for setup_rows in setups for r in setup_rows),*bridges,swap,voter,left,helper_left,d["restore"],closed,d["poll"],d["flag"],finished,d["latch"]]
        if active.get(writer) != clear[6]:
            if writer in active:
                continue
            booked = next((r for r in reversed(page) if r[:2] == ["T","book"] and
                           r[4:6] == [writer,clear[6]] and clear[2] < r[2] < swap[2]),None)
            if booked is None:
                continue
            kept.append(booked)
        if worker != owner:
            kept.append(entry)
        # Earlier resets overwrite one another. The complete assignment replay
        # supplies the last actual reset at each of the two views used by window.
        views = observed_views(rows,voter,d["poll"],d["flag"],swap)
        resets = [views[q[2]][1] for q in (voter,d["poll"])]
        if any(r is None for r in resets):
            continue
        adds = [r for r in rows if r[:2]==["T","increment"]]
        increments = {(r[4],r[5]+1):r for r in adds}
        require(len(adds)==len(increments) and all(0 <= r[5] < 2**64-1 for r in adds),
                "cutoff epoch wrapped or produced an ambiguous reset source")
        if any((r[4],r[5]) not in increments for r in resets):
            continue
        expected=[(r[3][6],r[1][3],r[1][2]) for r in reversed(setups)]
        require(views[voter[2]][0] == tuple(expected) and views[d["poll"][2]][0] == tuple(expected[1:]) and
                views[d["flag"][2]] == views[d["poll"][2]] and views[swap[2]] == expected[1],
                "compact path disagrees with complete task or reset history")
        epochs = [(r,increments[r[4],r[5]]) for r in resets]
        kept += resets
        return sorted({r[2]:r for r in kept}.values(),key=lambda r:r[2]),epochs,voter,finished,expected,views
    return None


def choose(log):
    rows = sorted(([tag,op,*map(int,args)] for tag,op,*args in map(str.split,log.splitlines())),
                  key=lambda r:r[2])
    require(len({r[2] for r in rows})==len(rows),"duplicate observation timestamp")
    groups, writes = defaultdict(list), defaultdict(list)
    pending, reporters, child_calls, installed = {}, {}, {}, {}
    for row in rows:
        tag,op,t,w,key,*a = row
        if tag == "N":
            if op == "cut":
                key3 = (w,key,a[0])
                child,incumbent = child_calls[key3],installed[key3]
                require(child[6] == incumbent[6] == a[1] and child[-2] == -a[3] and
                        incumbent[-1] == a[3] and child[2] < incumbent[2] < t,
                        "reported cutoff did not use the returned move score and its strict store")
                pending[w] = row,child,incumbent
            elif op == "mark":
                captured = pending.pop(w)
                require(captured[0][7]==key,"cutoff swap does not address its reporting ticket")
                reporters[t] = captured
                writes[key].append(row)
            else:
                groups[key].append(row)
                if op == "init":
                    writes[a[1]].append(row)
        elif tag != "O":
            continue
        elif op == "child":
            child_calls[w,key,a[0]] = row
        elif op == "wrote":
            installed[w,key,a[0]] = row
        elif op in ("pvs","call","mark","load","leave","stop","scored","returned"):
            groups[t if op=="call" else key].append(row)
    require(not pending,"cutoff call did not return from swap")
    found = []
    for group in groups.values():
        d = {r[1]:r for r in group}
        # First physical flag after restoring the enclosing split is already set.
        if [r[1] for r in group] != ["pvs","call","mark","init","restore","load","leave","poll","flag","latch","stop","chosen","returned"]:
            continue
        p, call, mark, child, restore, load, leave, poll, read, latch, stop, chosen, returned = group
        if p[11] or not p[14] or p[7] <= 12 or load[-1] == 0:
            continue
        w,addr,parent,flag = p[3],child[5],child[7],read[6]
        require(all(r[3]==w for r in group) and call[4]==returned[5]==p[2],"owner activation mismatch")
        require(read[-1] == stop[-1] == 1 and chosen[-1]==returned[-1]==0 and load[-1]==leave[-1],
                "flag latch or early zero return differs")
        require((mark[6],restore[5],load[6])==(addr,addr,addr) and mark[5]==load[5] and
                parent==restore[6]==poll[5]==read[5] and parent!=addr,"restored parent differs")
        changes = writes[flag]
        clears = [i for i,r in enumerate(changes) if r[1]=="init" and r[2] < read[2]]
        i = clears[-1]
        # Each clear and swap was sampled inside the existing split's lock.
        # Require that the *reader's own* two clears bracket its relaxed flag load.
        if i+2 >= len(changes):
            continue
        clear,swap,after = changes[i:i+3]
        if clear[3]!=w or after[3]!=w or swap[1]!="mark" or swap[3]==w or after[1]!="init" or after[2] <= read[2]:
            continue
        require(clear[5] == after[5] == parent and clear[2]<child[2] and swap[-1]==0,"wrong active parent era")
        old = next((j for j in reversed(range(i)) if changes[j][1]=="mark"),None)
        if old is None:
            continue
        cause,actual_child,strict = reporters[swap[2]]
        require(cause[4]==parent and cause[7]==flag and cause[2]<swap[2],"wrong cutoff award")
        parent_mark = next(r for r in groups[clear[4]] if (r[:2]==["O","mark"]))
        next_mark = next(r for r in groups[after[4]] if (r[:2]==["O","mark"]))
        require(parent_mark[3]==next_mark[3]==w and parent_mark[5:7]==next_mark[5:7] and
                parent_mark[6]==parent,"parent bank changed between clears")
        path = shape(rows,d,clear,swap,after,parent_mark)
        if path is None:
            continue
        found.append(((path[2][3]!=w,p[13]),d,changes[old:i],clear,swap,after,cause,parent_mark,next_mark,actual_child,strict,path))
    return max(found,key=lambda x:x[0])[1:] if found else None


def task_fields(path):
    rows,epochs,voter,finished,expected,views = path
    by_reset={r[2]:src for r,src in epochs}
    by_gen={r[4]:r for r in rows if r[:2]==["N","init"]}
    address={r[5]:r[6] for r in by_gen.values()}
    def optional(n):
        return f"(some {n})" if n else "none"
    def event(w,t):
        return f"⟨{w},{t}⟩"
    ops,slots = [],{}
    for row in rows:
        tag,op,t,w,*a = row
        if tag=="O":
            flag=by_gen[a[0]][6]
            require(slots.setdefault((w,a[1]),flag)==flag,"mark maps to another physical flag")
            ops.append(f".splitWord {w} {t} (.mark {a[1]})")
            continue
        if tag=="N":
            if op=="init": word=f".storeFlag {a[2]} false"
            elif op=="mark": word=f".storeFlag {a[0]} true"
            elif op=="restore": word=f".restore {address[a[1]]} {optional((address[a[2]] if a[2] else 0))}"
            elif op=="poll": word=f".afterSplit {optional((address[a[1]] if a[1] else 0))}"
            elif op=="flag": word=f".loadFlag {a[2]} {str(bool(a[3])).lower()} rf"
            elif op=="latch": word=".latch true"
            else: raise ValueError(op)
        elif op in ("copy","install"):
            method="copyParent" if op=="copy" else "install"
            word=f".{method} {a[1]} {optional(a[2])}"
        elif op=="active": word=f".ownerActive {a[1]}"
        elif op=="book": word=f".book {a[0]} {a[1]}"
        elif op=="detach": word=f".detach {a[0]}"
        elif op=="clearTask": word=f".clearTask {a[1]}"
        elif op=="poll": word=".poll"
        elif op=="endWalk": word=f".endWalk {str(bool(a[1])).lower()}"
        elif op=="reset":
            src=by_reset[t]
            require(src[4]==a[0] and src[5]+1==a[1],"epoch source differs from loaded word")
            word=f".resetCache {event(w,a[2])} {event(src[3],src[2])} {a[1]}"
        else: raise ValueError(op)
        ops.append(f".taskWord {w} {t} ({word})")
    require(rows[-1][:2] == ["N","latch"],"unterminated postJoin")
    parents=[r for r in rows if r[:2]==["N","init"]][:-1]
    post=next(r for r in rows if r[:2]==["N","poll"])
    vr,pr=views[voter[2]][1],views[post[2]][1]
    return dict(EXPECTEDPATH="["+",".join(f"⟨{f},⟨{w},{t}⟩⟩" for f,w,t in expected[1:])+"]",
        VOTERESET=event(vr[3],vr[2]),POSTRESET=event(pr[3],pr[2]),TASKOPS="[\n  "+",\n  ".join(ops)+"]",TASKMAP="\n".join(
        f"  | {w}, {bank} => some {flag}" for (w,bank),flag in slots.items()),
        TASKPATH="["+",".join(str(r[6]) for r in reversed(parents))+"]",
        VOTER=voter[3],VOTETIME=voter[2])


def replay(selected,directory,output):
    d,older,clear,swap,after,cause,parent_mark,next_mark,actual_child,strict,path = selected
    p,call,mark,child,restore,load,leave,poll,read,latch,stop,chosen,returned = d.values()
    def fw(r):
        return f"⟨⟨{r[3]},{r[2]}⟩,{str(r[1]=='mark').lower()}⟩"
    fields = dict(PLAYER=p[5],OPPONENT=p[6],DEPTH=p[7],ALPHA=p[8],BETA=p[9],PV=str(bool(p[10])).lower(),
                  CUT=str(bool(p[12])).lower(),LEVEL=p[13],WORKER=p[3],OPENED=p[2],CALL=call[2],MARK=mark[2],
                  BANK=mark[5],LOAD=load[2],RAW=load[-1],LEAVE=leave[2],CHOSEN=chosen[2],VALUE=chosen[-1],
                  RETURNED=returned[2],NORMAL=returned[-1],CLEAR=clear[2],WRITER=swap[3],SWAP=swap[2],
                  AFTER=after[2],POLL=poll[2],READ=read[2],LATCH=latch[2],STOP=stop[2],OLD=older[0][2],
                  FLAG=read[6],OLDER="["+",".join(fw(r) for r in older)+"]",PICK=len(older)+1,
                  CHILDADDR=child[5],CHILDFLAG=child[6],CHILDPARENT=child[7],RESTORECHILD=restore[5],
                  RESTORED=restore[6],ACTIVE=poll[5],SCANNED=read[5],PARENT=clear[5],PARENTFLAG=clear[6],
                  PARENTGEN=clear[4],CURRENTGEN=parent_mark[4],AFTERGEN=after[4],NEXTGEN=next_mark[4],
                  BANKEVENT=parent_mark[2],LATERBANK=next_mark[2],CUTADDR=cause[4],CUTFLAG=cause[7],
                  CUTSCORE=cause[8],CUTBETA=cause[9],CUTWORKER=cause[3],CUTSQUARE=cause[6],
                  CUTTICK=cause[2],CHILDGEN=child[4],RESTOREGEN=restore[4],FLAGVALUE=str(bool(read[-1])).lower(),
                  STOPPED=str(bool(stop[-1])).lower(),OLDWORKER=older[0][3],CHILDVALUE=actual_child[-2],
                  STRICTBEFORE=strict[-2],STRICTAFTER=strict[-1],REPORTSQUARE=strict[6],
                  RETURNSQUARE=actual_child[6],REMOTECHILD=actual_child[2],STRICTSTORE=strict[2],
                  RESTORESTEP=restore[2])
    fields.update(task_fields(path))
    template = (Path(__file__).parent/"serial_calls/AncestorStop.lean.in").read_text()
    file = directory/"NativeAncestorStop.lean"
    file.write_text(re.sub(r'\b[A-Z]+\b',lambda m:str(fields.get(m[0],m[0])),template))
    subprocess.run(["lake","env","lean","-DwarningAsError=true",str(file)],cwd=LEAN,check=True)
    if output:
        rows = [*d.values(),*older,clear,swap,after,cause,parent_mark,next_mark,actual_child,strict]
        rows.extend(path[0])
        rows.extend(src for _,src in path[1])
        unique = {r[2]:r for r in rows}
        output.write_text("".join(" ".join(map(str,r))+"\n" for _,r in sorted(unique.items())))
    print(f"hard-20 #166, level {p[13]}, worker {p[3]}, nested bank {mark[5]}, raw score {load[-1]}. "
          f"Restored parent bank {parent_mark[5]}, flag {read[6]:#x}, read true at {read[2]}. "
          f"Same-worker clears {clear[2]} and {after[2]} enclose the one true swap ({swap[3]},{swap[2]}). "
          f"The stopped PVS returned {returned[-1]} at {returned[2]}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-native-ancestor-stop-") as tmp:
        folder=Path(tmp)
        binary=capture(folder,args.target_dir.resolve(),instrument,fixture=FIXTURES/"parallel_tickets.rs",serial_trace=False)
        for _ in range(5):
            native=subprocess.run([binary,REPO],text=True,capture_output=True,check=True)
            matched=choose(native.stdout)
            if matched is not None:
                replay(matched,folder,args.output)
                break
        else:
            raise AssertionError("no remote ancestor mark bounded by the stopping worker's adjacent generations")


if __name__=="__main__":
    main()

"""Replay the source candidates of native split-worker alpha reads on a reused slot."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile

from check_native_split_owner import instrument as instrument_owner
from check_serial_calls_trace import FIXTURES, REPO, capture, replace_once
from normalize_pvs_trace import require


HERE = Path(__file__).resolve().parent


def instrument(src):
    instrument_owner(src)
    pvs = src / "search/pvs.rs"
    text = pvs.read_text() + """
thread_local! { pub(super) static ALPHA_ACTOR: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) }; }
"""
    store = "                    sp.set_alpha(score);"
    pvs.write_text(replace_once(text, store, store + """
                    println!("A store {} {} {addr} {}",owner_tick(),thread.idx(),score.value());"""))
    th = src / "search/threading/thread.rs"
    text = th.read_text()
    for anchor in ("        let task = SplitPointTask::new", "        let this_sp = self.active_split_point().clone();"):
        text = replace_once(text, anchor, "        crate::search::pvs::ALPHA_ACTOR.set(self.idx);\n" + anchor)
    seed = "            sp_state.set_alpha(req.alpha);"
    th.write_text(replace_once(text, seed, seed + """
            println!("A seed {} {} {} {}",crate::search::pvs::owner_tick(),self.idx,
                Arc::as_ptr(sp) as usize,req.alpha.value());"""))
    point = src / "search/threading/split_point.rs"
    point.write_text(replace_once(point.read_text(), "self.mutex.lock()", "self.mutex.lock().alpha_trace(self as *const _ as usize)"))
    search = src / "search.rs"
    with search.open("a") as file:
        file.write("""
pub(crate) fn alpha_lock_event(addr: usize, acquired: bool) {
    println!("A {} {} {} {}",if acquired { "lock" } else { "unlock" },
        pvs::owner_tick(),pvs::ALPHA_ACTOR.get(),addr);
}
""")
    spin = src / "util/spinlock.rs"
    text = replace_once(spin.read_text(), "SpinLockGuard<'a>(&'a SpinLock);", """SpinLockGuard<'a>(&'a SpinLock, Option<usize>);
impl SpinLockGuard<'_> {
    pub(crate) fn alpha_trace(mut self, addr: usize) -> Self {
        self.1 = Some(addr);
        crate::search::alpha_lock_event(addr,true);
        self
    }
}""")
    text = replace_once(text, "        SpinLockGuard(self)", "        SpinLockGuard(self,None)")
    spin.write_text(replace_once(text, "        self.0.unlock();",
        "        if let Some(addr) = self.1 { crate::search::alpha_lock_event(addr,false); }\n        self.0.unlock();"))


@dataclass(frozen=True)
class Row:
    op: str
    tick: int
    w: int
    addr: int
    args: tuple


def parse(log):
    slots = defaultdict(list)
    mapping = {}
    for line in log.splitlines():
        p = line.split()
        if p[:2] == ["O","mark"]:
            t,w,_,bank,addr = map(int,p[2:])
            require(mapping.setdefault(addr,(w,bank)) == (w,bank), "address mapped to different banks")
        if not (p[0:1] == ["A"] or p[:2] in (["O","sample"],["O","child"])):
            continue
        t,w,addr,*args = map(int,p[2:])
        slots[addr].append(Row(p[1],t,w,addr,tuple(args)))
    return mapping, slots


def candidates(mapping, slots):
    options = []
    for address, unsorted in slots.items():
        owner, bank = mapping[address]
        # Only per-worker program order and the order of records INSIDE the
        # same split mutex are used below. The relaxed alpha load is outside it.
        rows = sorted(unsorted, key=lambda r: r.tick)
        guards, writes, opened = [], [], None
        for x in rows:
            if x.op == "lock":
                require(opened is None, "two successful acquisitions overlap on one split mutex")
                opened = x
            elif x.op == "unlock":
                require(opened and opened.w == x.w, "split mutex released by wrong actor")
                guards.append((opened,x))
                opened = None
            elif x.op in ("seed","store"):
                require(opened and opened.w == x.w, "alpha store outside its split lock")
                writes.append((x,len(guards)))
        require(opened is None, "unfinished lock in completed run")
        seeds = [i for i,(x,_) in enumerate(writes) if x.op == "seed"]
        if len(seeds) < 2:
            continue
        for worker in {x.w for x in rows if x.op == "sample"} - {owner}:
            local = [x for x in rows if x.w == worker and x.op in ("sample","child")]
            own_guards = [i for i,(a,_) in enumerate(guards) if a.w == worker]

            def bounds(read):
                before = max((k for k in own_guards if guards[k][1].tick < read.tick), default=-1)
                after = min((k for k in own_guards if read.tick < guards[k][0].tick), default=len(guards))
                lo = max((i for i,(_,g) in enumerate(writes) if g <= before), default=-1)
                hi = min((i for i,(_,g) in enumerate(writes) if after <= g), default=len(writes))
                return lo,hi

            for j in range(0,len(local)-3):
                first,scout,second,wide = local[j:j+4]
                if ([r.op for r in (first,scout,second,wide)] != ["sample","child","sample","child"]
                        or first.args[2] != 0 or second.args[2] != 1
                        or not all(r.args[:2] == first.args[:2] for r in (scout,second,wide))):
                    continue
                narrow, fresh = first.args[-1], second.args[-1]
                require(scout.args[4:6] == (wide.args[4],0) and wide.args[5] == 1
                        and scout.args[6:8] == (-(narrow+1),-narrow) and wide.args[7] == -fresh
                        and first.tick < scout.args[-1] < scout.tick < second.tick < wide.args[-1] < wide.tick,
                        "native child window or local issue order differs from its sampled alpha")
                bs = [bounds(first),bounds(second)]
                for birth,next_ in zip(seeds[1:],seeds[2:]):
                    if not all(birth <= lo < hi <= next_ for lo,hi in bs):
                        continue
                    cs = [[i for i in range(lo,hi) if writes[i][0].args[0] == r.args[-1]]
                          for r,(lo,hi) in zip([first,second],bs)]
                    pairs = [(a,b) for a in cs[0] for b in cs[1] if a <= b]
                    require(pairs, "native reads have no source respecting lock bounds and read-read coherence")
                    stale = next((i for i in range(birth-1,-1,-1) if writes[i][0].args[0] in (narrow,fresh)), None)
                    if stale is None:
                        continue
                    lower_guard,upper_guard = writes[birth][1],writes[next_][1]
                    locked = guards[lower_guard:upper_guard+1]
                    # Every section of this address between the two seeds is retained.
                    # All stores, including those outside these sections, remain in MO.
                    score = (narrow != fresh, len(pairs) == 1, any(writes[b][0].w != worker for _,b in pairs),
                             -len(locked),-len(writes))
                    options.append((score,dict(owner=owner,bank=bank,address=address,worker=worker,
                        birth=birth,nextSeed=next_,stale=stale,
                        writes=[dict(actor=r.w,pc=r.tick,value=r.args[0],seed=r.op=="seed") for r,_ in writes],
                        guards=[dict(actor=a.w,acquired=a.tick,released=b.tick) for a,b in locked],
                        reads=[dict(actor=r.w,pc=r.tick,value=r.args[-1],seed=False) for r in (first,second)],
                        children=[dict(entered=r.args[-1],returned=r.tick,lo=r.args[6],hi=r.args[7],raw=r.args[-2],
                            ticket=r.args[0],square=r.args[1],pv=bool(r.args[5])) for r in (scout,wide)],
                        ticket=first.args[0],square=first.args[1],pairs=[list(x) for x in pairs])))
                    break
    return max(options,key=lambda item:item[0])[1] if options else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="retain the native operands as replayable JSON")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-native-split-alpha-") as tmp:
        directory = Path(tmp)
        binary = capture(directory, args.target_dir.resolve(), instrument,
                         fixture=FIXTURES / "parallel_tickets.rs", serial_trace=False)
        selected = None
        for _ in range(3):
            run = subprocess.run([binary, REPO], text=True, stdout=subprocess.PIPE, check=True)
            witness = candidates(*parse(run.stdout))
            if witness is not None:
                selected = witness
                if witness["reads"][0]["value"] != witness["reads"][1]["value"]:
                    break
        require(selected is not None, "no helper PV retry bracketed by its reused split mutex")
        payload = json.dumps(selected, separators=(",",":"))
        receipts = directory / "operands.json"
        receipts.write_text(payload)
        program = directory / "NativeSplitAlpha.lean"
        program.write_text((HERE / "serial_calls/SplitAlpha.lean.in").read_text())
        subprocess.run(["lake", "env", "lean", "-DwarningAsError=true", "--run", str(program),
                        str(receipts)], cwd=HERE.parent, check=True)
        if args.output is not None:
            args.output.write_text(payload + "\n")
        print(f"hard-20 #166, four workers, all Multi-PV scores matched OBF; "
              f"physical address {selected['address']:#x}, {len(selected['writes'])} stores, "
              f"{len(selected['guards'])} contiguous guard sections, "
              f"ticket {selected['ticket']} square {selected['square']}.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replay a reused split mask from four-worker public Search::run on hard-20 #166."""
import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

from check_native_split_owner import instrument as instrument_owner
from check_serial_calls_trace import FIXTURES, LEAN, REPO, capture, replace_once
from normalize_pvs_trace import require


def instrument(src):
    instrument_owner(src)
    pvs = src / "search/pvs.rs"
    with pvs.open("a") as file:
        file.write("""
thread_local! { pub(super) static MASK_ACTOR: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) }; }
""")
    bits = src / "util/bitset.rs"
    body = bits.read_text()
    body = replace_once(body, "impl AtomicBitSet {", """impl AtomicBitSet {
    pub(crate) fn trace_address(&self) -> usize { self.data.as_ptr() as usize }
    pub(crate) fn trace_acquire(&self) -> u64 { self.data.load(Ordering::Acquire) }
""")
    body = replace_once(body, "self.data.load(Ordering::Acquire) == 0", "self.trace_acquire() == 0")
    for method, operation in (("set", "fetch_or(1 << index, Ordering::Relaxed)"),
                              ("reset", "fetch_and(!(1 << index), Ordering::Release)")):
        body = replace_once(body, f"pub fn {method}(&self, index: usize) {{",
                            f"pub fn {method}(&self, index: usize) -> u64 {{")
        body = replace_once(body, f"self.data.{operation};", f"self.data.{operation}")
    bits.write_text(body)
    point = src / "search/threading/split_point.rs"
    anchor = "    /// Returns the cutoff flag."
    point.write_text(replace_once(point.read_text(),anchor,"""    pub(in crate::search) fn trace_flag_addr(&self) -> usize {
        self.cutoff.as_ptr() as usize
    }

""" + anchor))
    thread = src / "search/threading/thread.rs"
    body = thread.read_text()
    clock = "crate::search::pvs::owner_tick()"
    actor = "crate::search::pvs::MASK_ACTOR"
    for anchor in ("        let task = SplitPointTask::new", "        let this_sp = self.active_split_point().clone();"):
        body = replace_once(body, anchor, f"        {actor}.set(self.idx);\n" + anchor)
    anchor = "            sp_state.helpers_mask.clear();"
    body = replace_once(body, anchor, anchor + f"""
            let mask_addr = sp_state.helpers_mask.trace_address();
            println!("M clear {{}} {{}} {{mask_addr}}", {clock}, self.idx);""")
    # All mask modifications execute inside the existing split lock. The stamp
    # stays inside the same guard, even when the physical store is a no-op.
    mark = "            sp_state.helpers_mask.set(self.idx);"
    body = replace_once(body, mark, f"""            let old = sp_state.helpers_mask.set(self.idx);
            let mark_stamp = {clock};
            println!("M mark {{mark_stamp}} {{}} {{mask_addr}} {{}} {{old}} {{}}", self.idx, self.idx, old | 1u64 << self.idx);""")
    body = replace_once(body, 'println!("O mark {} {} {} {} {}", crate::search::pvs::owner_tick(),',
                        'println!("P addresses {mark_stamp} {} {bank} {mask_addr} {} {}", self.idx, '
                        'sp_state.trace_flag_addr(), Arc::as_ptr(sp) as usize);\n'
                        '            println!("O mark {} {} {} {} {}", mark_stamp,')
    body = replace_once(body, "        sp_state.helpers_mask.set(self.idx);", f"""        let mask_addr = sp_state.helpers_mask.trace_address();
        let old = sp_state.helpers_mask.set(self.idx);
        println!("M book {{}} {{}} {{mask_addr}} {{}} {{old}} {{}}", {clock}, {actor}.get(), self.idx, old | 1u64 << self.idx);""")
    for state in ("sp_state", "state"):
        original = f"            {state}.helpers_mask.reset(self.idx);"
        body = replace_once(body, original, f"""            let mask_addr = {state}.helpers_mask.trace_address();
            let old = {state}.helpers_mask.reset(self.idx);
            println!("M finish {{}} {{}} {{mask_addr}} {{}} {{old}} {{}}", {clock}, {actor}.get(), self.idx, old & !(1u64 << self.idx));""")
    # Preserve the short circuit and log the existing exit and mask loads once.
    anchor = """        while !self.exit.load(Ordering::Acquire) {
            // Check if we're the owner of a split point and all searchers have finished
            if let Some(ref sp) = this_sp
                && sp.state().helpers_mask.none()
            {
                break;
            }"""
    body = replace_once(body, anchor, f"""        loop {{
            let closed = self.exit.load(Ordering::Acquire);
            if let Some(ref sp) = this_sp {{
                println!("M shutdown {{}} {{}} {{}} {{}}", {clock},self.idx,
                    sp.state().helpers_mask.trace_address(),closed as u8);
            }}
            if closed {{ break; }}
            if let Some(ref sp) = this_sp {{
                let addr = sp.state().helpers_mask.trace_address();
                let word = sp.state().helpers_mask.trace_acquire();
                println!("M load {{}} {{}} {{addr}} {{word}}", {clock},self.idx);
                if word == 0 {{ break; }}
            }}""")
    anchor = """        self.idle_loop();

        std::mem::forget(unwind_guard);"""
    body = replace_once(body, anchor, f"""        self.idle_loop();
        println!("M returned {{}} {{}} {{}}", {clock},self.idx,sp.state().helpers_mask.trace_address());

        std::mem::forget(unwind_guard);""")
    anchor = "        sp.clear_task();\n        sp.clear_move_iter();"
    body = replace_once(body, anchor, anchor + f"""
        println!("M closed {{}} {{}} {{}}", {clock},self.idx,sp.state().helpers_mask.trace_address());""")
    thread.write_text(body)


MODS = {"clear", "mark", "book", "finish"}


@dataclass(frozen=True)
class Row:
    op: str
    tick: int
    actor: int
    addr: int
    args: tuple

    def word(self):
        return self.args[-1] if self.op != "clear" else 0


def row(tagged):
    op, tick, actor, addr, *args = tagged
    return Row(op, int(tick), int(actor), int(addr), tuple(map(int, args)))


def select(log):
    masks, owners, interior, addresses = [], {}, set(), {}
    for line in log.splitlines():
        parts = line.split()
        if parts[0] == "M":
            masks.append(row(parts[1:]))
        elif parts[:2] == ["P", "addresses"]:
            tick, worker, bank, mask, cutoff, point = map(int, parts[2:])
            addresses[tick] = (worker, bank, mask, cutoff, point)
        elif parts[:2] == ["O", "pvs"]:
            if parts[11] == "0" and parts[-1] == "1":
                interior.add(int(parts[4]))
        elif parts[:2] == ["O", "mark"]:
            tick, worker, call, bank, point = map(int, parts[2:])
            owners[tick] = (worker, call, bank, point)
    masks.sort(key=lambda m: m.tick)
    require(len({m.tick for m in masks}) == len(masks), "duplicate native mask instruction")
    owner_ends = {}
    for line in log.splitlines():
        parts = line.split()
        if parts[0] == "O" and parts[1] in {"load", "leave", "stop", "scored", "returned"}:
            tick, worker, call, *args = map(int, parts[2:])
            owner_ends.setdefault(call, {})[parts[1]] = (tick, worker, *args)
    slots = {}
    for m in masks:
        require(m.actor < 4, "mask modification executed by unregistered actor")
        slots.setdefault(m.addr, []).append(m)
    choices = []
    for addr, events in slots.items():
        generations, active, current, identity = [], None, 0, None
        for m in events:
            if m.op == "clear":
                # An owner may already have observed zero when the helper
                # records the RMW result under its lock. Its log receipt can
                # follow close. Only the owner's close-before-reuse is required.
                require(active is None or sum(r.op == "closed" for r in active) == 1,
                        "reuse before the previous owner closed its slot")
                active = [m]
                generations.append(active)
                current = 0
            else:
                require(active is not None, "mask activity before first clear")
                active.append(m)
            if m.op in {"mark", "book", "finish"}:
                bit, old, new = m.args
                require(bit < 4 and old == current, "RMW does not read its locked predecessor")
                require(new == (old & ~(1 << bit) if m.op == "finish" else old | 1 << bit),
                        "wrong mask RMW operand")
                current = new
            if m.op == "mark":
                worker, _, bank, point = owners[m.tick]
                require(worker == m.actor == m.args[0] and m.args[1] == 0,
                        "seed mark mismatches owner or prior clear")
                require(addresses[m.tick][:3] == (worker, bank, addr) and addresses[m.tick][-1] == point,
                        "native mask, split pointer, and owner mapping disagree")
                if identity is None:
                    identity = addresses[m.tick]
                require(identity == addresses[m.tick], "one mask aliased distinct owner slots")
        for prev, gen, following in zip(generations, generations[1:], generations[2:]):
            if gen[1].op != "mark" or following[0].op != "clear":
                continue
            mark, owner = gen[1], gen[1].actor
            _, call, bank, point = owners[mark.tick]
            ends = owner_ends[call]
            if (call not in interior or gen[-1].op != "closed" or "scored" not in ends or ends["stop"][-1] != 0
                    or not any(m.op=="book" and m.args[0]!=owner for m in gen)):
                continue
            probes = [m for m in gen if m.op == "load"]
            if not probes or probes[-1].word() != 0:
                continue
            # Modification order from the existing split lock, never a remote
            # writer-to-reader bound from the logging clock.
            mods = [m for m in [*gen, following[0]] if m.op in MODS]
            candidates = []
            for load in probes:
                lower = max(i for i,m in enumerate(mods) if m.actor == owner and m.tick < load.tick)
                upper = min(i for i,m in enumerate(mods) if m.actor == owner and m.tick > load.tick)
                candidates.append([m for m in mods[lower:upper] if m.word() == load.word()])
            require(all(candidates), "no coherent source for a native acquire")
            if any(len(cs) != 1 for cs in candidates) or candidates[-1][0].actor == owner:
                continue
            # The explicit SAT witness below uses the clock as one topological
            # order of already identified release/acquire edges. It is never
            # used by the independent source-identification theorem.
            if any(sum(m.tick < read.tick for m in mods) != mods.index(cs[0]) + 1
                   for read,cs in zip(probes,candidates)):
                continue
            prev_zero = [m for m in prev if m.op=="finish" and m.word()==0][-1]
            require(gen[-2].op == "returned" and gen[-1].tick < ends["load"][0] < ends["leave"][0]
                    < ends["stop"][0] < ends["scored"][0] < ends["returned"][0]
                    and ends["load"][2:4] == (bank, point)
                    and len({ends[op][-1] for op in ("load","leave","scored","returned")}) == 1,
                    "retirement did not feed the native PVS result")
            choices.append((len(gen), gen, following[0], prev_zero, [cs[0] for cs in candidates],
                            owner, call, bank, point, addresses[mark.tick][3], ends))
    return min(choices, key=lambda c:c[0], default=None)


def event(m):
    return f"⟨{m.actor},{m.tick}⟩"


def lean_list(items):
    return "[\n  " + ",\n  ".join(items) + "]"


def flag(x):
    return str(bool(x)).lower()


def render(log, selected, output):
    import re
    _, gen, nxt, old_zero, sources, owner, call, bank, point, cutoff, ends = selected
    p = next(list(map(int, parts[2:])) for parts in map(str.split,log.splitlines())
             if parts[:2] == ["O", "pvs"] and int(parts[4]) == call)
    opened, _, _, player, opponent, depth, alpha, beta, pv, root, cut, level, endgame = p
    require(endgame and not root and ends["returned"][-2] == opened, "expected interior PVS frame")
    mark = gen[1]
    occupied = [old_zero] + [m for m in gen if m.op in MODS] + [nxt]
    probes = [m for m in gen if m.op == "load"]
    readings = [f"⟨{event(m)},{m.word()},{event(w)}⟩" for m,w in zip(probes,sources)]
    # This projected generation includes every native guard and modification.
    # Prior/following generations are used by the RF proof's independent mask slice.
    tape = []
    def add(tick, form):
        tape.append((tick, form))
    add(opened, f".engine (.localCall {owner} (.entry {opened} 0 .pvs .none none) (some args))")
    add(call, f".engine (.localCall {owner} (.entry {call} 5 .split .none none) none)")
    it = iter(sources)
    for m in [old_zero,*gen,nxt]:
        if m.op == "closed":
            continue
        if m.op in {"clear", "shutdown", "returned", "load"}:
            word = ""
            if m.op == "shutdown":
                word = f" {flag(m.word())}"
            if m.op == "load":
                source = next(it)
                name = "remote" if m == probes[-1] else event(source)
                word = f" {name} {m.word()} true"
            op = f".maskWord {owner} {m.tick} (.{m.op} {bank}{word})"
        elif m.op == "mark":
            op = f".splitWord {owner} {m.tick} (.mark {bank})"
        else:
            bit, previous, stored = m.args
            op = (f".splitWord {m.actor} {m.tick} (.maskRmw {owner} {bank} {bit} "
                  f"{flag(m.op=='book')} {previous} {stored})")
        add(m.tick,op)
    word = f"⟨BitVec.ofInt 32 ({ends['scored'][-1]})⟩"
    load, leave, scored, ret = (ends[op][0] for op in ("load", "leave", "scored", "returned"))
    add(load, f".splitWord {owner} {load} (.result {bank} (BitVec.ofInt 32 ({ends['load'][-1]})))")
    add(leave, f".engine (.localCall {owner} (.leave {leave} {call} .split none none) none)")
    add(scored, f".engine (.localCall {owner} (.scored {scored} {word}) none)")
    add(ret, f".engine (.localCall {owner} (.leave {ret} {opened} .pvs none (some {word})) none)")
    fields = dict(ACTIONS=len(occupied)-4,
                  EXIT=gen[-2].tick, OLD=old_zero.tick, OLDWORKER=old_zero.actor,
                  WRITES=lean_list(f"⟨{event(m)},{m.word()},{flag(m.op=='finish')},{flag(m.op!='clear')}⟩"
                                  for m in occupied),
                  RECEIPTS=lean_list(readings), TERMINAL=readings[-1],
                  LAST=len(occupied)-1, COUNT=len(occupied), PLAYER=player, OPPONENT=opponent,
                  ALPHA=alpha, BETA=beta, PV=flag(pv), CUT=flag(cut), DEPTH=depth, LEVEL=level,
                  TAPE=lean_list(op for _,op in sorted(tape)), OWNER=owner, BANK=bank, NEXT=nxt.tick, FLAG=cutoff, MASK=mark.addr, POINT=point,
                  MARK=mark.tick, SCORELOAD=load, SCORE=ends["scored"][-1])
    template = (Path(__file__).parent/"serial_calls/MaskGeneration.lean.in").read_text()
    output.write_text(re.sub(r'\b[A-Z]+\b',lambda m:str(fields.get(m[0],m[0])),template))
    return (f"Hard-20 #166: public exact +1536; all MultiPV rows matched the OBF. "
            f"Owner {owner}, bank {bank}, split {point:#x}, mask {mark.addr:#x}, cutoff {cutoff:#x}, mark {mark.tick}. "
            f"{len(probes)} acquires; final remote release {event(sources[-1])}, "
            f"loaded {probes[-1].tick}; prior zero {event(old_zero)}, reuse clear {event(nxt)}. "
            f"Idle returned {gen[-2].tick}, slot closed {gen[-1].tick}; PVS returned {ends['returned'][-1]}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-native-mask-") as tmp:
        directory=Path(tmp)
        binary=capture(directory,args.target_dir.resolve(),instrument,fixture=FIXTURES/"parallel_tickets.rs",
                       serial_trace=False)
        for _ in range(5):
            run=subprocess.run([binary,REPO],capture_output=True,text=True,check=True)
            selected=select(run.stdout)
            if selected is not None:
                output=args.output.resolve() if args.output else directory/"Mask.lean"
                summary=render(run.stdout,selected,output)
                subprocess.run(["lake","env","lean","-DwarningAsError=true",str(output)],cwd=LEAN,check=True)
                print(summary)
                return
        raise AssertionError("no reused slot with distinct remote release and uniquely bounded loads")


if __name__ == "__main__":
    main()

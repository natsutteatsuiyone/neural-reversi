"""Trace a helper’s last strict improvement through an interior split owner’s final load and PVS return."""

import argparse
from collections import defaultdict
from pathlib import Path
import re
import subprocess
import tempfile

from check_serial_calls_trace import FIXTURES, LEAN, REPO, capture, replace_once
from normalize_pvs_trace import require


def instrument(src):
    pvs = src / "search/pvs.rs"
    text = pvs.read_text() + """
static OWNER_CLOCK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
pub(super) fn owner_tick() -> u64 {
    OWNER_CLOCK.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
thread_local! { pub(super) static OWNER_SPLIT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }
"""
    anchor = "    let all_node = !NT::PV_NODE && !cut_node;"
    text = replace_once(text, anchor, """    let opening = if SS::IS_ENDGAME && depth >= endgame::MIN_SPLIT_DEPTH {
        owner_tick()
    } else { 0 };
    let mut split_id = 0;
    let result = (|| {
""" + anchor)
    anchor = "            let (s, m, c) = thread.split("
    text = replace_once(text, anchor, """            split_id = owner_tick();
            let outer = OWNER_SPLIT.replace(split_id);
            println!("O pvs {} {} {} {} {} {} {} {} {} {} {} {} {}", opening, thread.idx(), split_id,
                board.player().bits(), board.opponent().bits(), depth, org_alpha.value(), beta.value(),
                NT::PV_NODE as u8, NT::ROOT_NODE as u8, cut_node as u8, ctx.selectivity.as_u8(), SS::IS_ENDGAME as u8);
            println!("O call {} {} {}", split_id, thread.idx(), opening);
""" + anchor)
    anchor = "            best_score = s;"
    text = replace_once(text, anchor, """            println!("O leave {} {} {} {}", owner_tick(), thread.idx(), split_id, s.value());
            OWNER_SPLIT.set(outer);
""" + anchor)
    anchor = "            if thread.should_stop_after_split() {"
    text = replace_once(text, anchor, """            let stopped = thread.should_stop_after_split();
            println!("O stop {} {} {} {}", owner_tick(), thread.idx(), split_id, stopped as u8);
            if stopped {""")
    anchor = "    let bound = Bound::classify::<NT>(best_score, org_alpha, beta);"
    text = replace_once(text, anchor, """    if split_id != 0 {
        println!("O scored {} {} {} {}", owner_tick(), thread.idx(), split_id, best_score.value());
    }
""" + anchor)
    text = replace_once(text, "    best_score\n}\n", """    best_score
    })();
    if split_id != 0 {
        println!("O returned {} {} {} {} {}", owner_tick(), thread.idx(), split_id, opening, result.value());
    }
    result
}
""")
    head, body = text.split("pub(super) fn search_split_point",1)
    body, tail = body.split("/// Performs enhanced transposition cutoff",1)
    anchor = "    while let Some((mv, move_count)) = move_iter.next() {"
    body = replace_once(body,anchor,anchor + """
        let addr = Arc::as_ptr(split_point) as usize;
        let idx = move_count - 1;
        println!("O ticket {} {} {addr} {idx} {}", owner_tick(),thread.idx(),mv.sq.index());""")
    for node in ("PV","NonPV"):
        body = replace_once(body,f"search::<{node}, EndGameStrategy>(",f"owner_child::<{node}>(addr,idx,mv.sq,")
    anchor = "        let alpha = split_point.state().alpha();"
    require(body.count(anchor) == 2, "split worker alpha reads changed")
    sample = 'println!("O sample {} {} {addr} {idx} {} {} {}",owner_tick(),thread.idx(),mv.sq.index(),PHASE,alpha.value());'
    body = body.replace(anchor,anchor+"\n        "+sample.replace("PHASE","0"),1)
    wide = "            let alpha = split_point.state().alpha();"
    body = replace_once(body,wide,wide+"\n            "+sample.replace("PHASE","1"))
    anchor = "            split_point.state().set_all_helpers_searching_locked(false);"
    body = replace_once(body,anchor,'            println!("O discard {} {} {addr} {idx} {}",owner_tick(),thread.idx(),mv.sq.index());\n'+anchor)
    anchor = "        if score > sp.best_score() {"
    body = replace_once(body,anchor,"""        let old = sp.best_score();
        println!("O decision {} {} {addr} {idx} {} {} {}",owner_tick(),thread.idx(),mv.sq.index(),old.value(),score.value());
        if score > old {""")
    anchor = "            sp.set_best_score(score);"
    body = replace_once(body,anchor,anchor + """
            println!("O wrote {} {} {addr} {idx} {} {} {}",owner_tick(),thread.idx(),mv.sq.index(),old.value(),score.value());""")
    child = """
fn owner_child<NT: NodeType>(addr: usize, idx: usize, sq: crate::square::Square,
    ctx: &mut SearchContext, board: &Board, depth: Depth, alpha: Score, beta: Score,
    thread: &Arc<Thread>, cut: bool) -> Score {
    let entered = owner_tick();
    let score = search::<NT,EndGameStrategy>(ctx,board,depth,alpha,beta,thread,cut);
    println!("O child {} {} {addr} {idx} {} {} {} {depth} {} {} {} {} {} {entered}",
        owner_tick(),thread.idx(),sq.index(),board.player().bits(),board.opponent().bits(),NT::PV_NODE as u8,
        alpha.value(),beta.value(),cut as u8,score.value());
    score
}
"""
    pvs.write_text(head + child + "pub(super) fn search_split_point" + body + "/// Performs enhanced transposition cutoff" + tail)
    thread = src / "search/threading/thread.rs"
    text = thread.read_text()
    anchor = "            sp_state.helpers_mask.set(self.idx);"
    receipt = """
            let bank = self.split_points.iter().position(|point| Arc::ptr_eq(point,sp)).unwrap();
            println!("O mark {} {} {} {} {}", crate::search::pvs::owner_tick(), self.idx,
                crate::search::pvs::OWNER_SPLIT.get(), bank, Arc::as_ptr(sp) as usize);"""
    text = replace_once(text, anchor, anchor + receipt)
    anchor = "        let move_iter = ConcurrentMoveIterator::from_offset(req.move_list, req.move_count);"
    text = replace_once(text,anchor,"""        let ordered: Vec<_> = req.move_list.iter().map(|m| m.sq.index().to_string()).collect();
""" + anchor)
    anchor = "            sp_state.set_best_score(req.best_score);"
    text = replace_once(text,anchor,anchor + """
            println!("O seed {} {} {} {} {} {} {} {} {}",crate::search::pvs::owner_tick(),self.idx,
                crate::search::pvs::OWNER_SPLIT.get(),bank,Arc::as_ptr(sp) as usize,req.best_score.value(),
                req.alpha.value(),req.move_count,ordered.join(" "));""")
    anchor = "        (sp_state.best_score(), sp_state.best_move(), counters)"
    text = replace_once(text, anchor, """        let score = sp_state.best_score();
        let bank = self.split_points.iter().position(|point| Arc::ptr_eq(point,sp)).unwrap();
        println!("O load {} {} {} {} {} {}", crate::search::pvs::owner_tick(), self.idx,
            crate::search::pvs::OWNER_SPLIT.get(), bank, Arc::as_ptr(sp) as usize, score.value());
        (score, sp_state.best_move(), counters)""")
    thread.write_text(text)


def select(log):
    rows = sorted(([op, *map(int, args)] for tag, op, *args in map(str.split, log.splitlines())
                   if tag == "O"), key=lambda row: row[1])
    require(len(rows) == len({row[1] for row in rows}), "duplicate native instruction stamps")
    calls, lifetime, previous, addresses = defaultdict(list), {}, {}, {}
    traffic = {"ticket","sample","child","discard","decision","wrote"}
    for row in rows:
        op, tick, worker, arg, *_ = row
        if op == "mark":
            bank, addr = row[4:]
            require(addresses.setdefault((worker,bank),addr) == addr,"bank changed physical address")
            previous[arg] = lifetime.get(addr)
            lifetime[addr] = row
        if op in traffic:
            require(arg in lifetime,"helper touched an uninitialized physical split point")
            calls[lifetime[arg][3]].append(row)
        else:
            calls[tick if op == "call" else arg].append(row)
    require(len(set(addresses.values())) == len(addresses),"workers share a split-point bank")
    scouts, wides = [], []
    for generation, evidence in calls.items():
        owner = [r for r in evidence if r[0] not in traffic]
        if [r[0] for r in owner] != ["pvs","call","mark","seed","load","leave","stop","scored","returned"]:
            continue
        p, call, mark, seed, load, leave, stop, scored, returned = owner
        w = p[2]
        if p[10] or not p[13] or p[6] <= 12 or stop[-1]:
            continue
        prior = previous[generation]
        require(prior is None or prior[1] < call[1],"previous bank generation started inside this split call")
        require(p[1] == call[3] == returned[4] and all(r[2] == w for r in owner),"wrong parent activation")
        require(mark[4:6] == seed[4:6] == load[4:6],"slot changed during generation")
        require(load[-1] == leave[-1] == scored[-1] == returned[-1],"PVS owner changed the loaded score")
        by_ticket = defaultdict(list)
        for row in evidence:
            if row[0] in traffic:
                require(seed[1] < row[1] < load[1] and row[3] == mark[5],"traffic outside generation")
                by_ticket[row[4]].append(row)
        updates = [r for r in evidence if r[0] == "wrote"]
        if not updates or updates[-1][2] == w:
            continue
        winner = updates[-1]
        tie = next((r for r in evidence if r[0] == "decision" and r[1] > winner[1]
                    and r[-2] == r[-1] == winner[-1]),None)
        winning_children = [r for r in by_ticket[winner[4]] if r[0] == "child"]
        scout_case = prior is not None and not p[9] and p[12] == 5 and len(winning_children) == 1 and tie is not None
        wide_case = p[9] and len(winning_children) == 2
        if wide_case:
            first, second = winning_children
            wide_case = second[11] - second[10] > 1 and first[-2] != second[-2]
        if not (scout_case or wide_case):
            continue
        # Validate every outcome, including canceled tickets. The successful CAS
        # returned the 1-based index which Rust then converts to its move-list index.
        packets = []
        for index, path in by_ticket.items():
            claim,*rest = path
            who, addr, idx, sq = claim[2:6]
            require(claim[0]=="ticket" and seed[8] <= idx < len(seed[9:]) and seed[9+idx] == sq,
                    "native ticket differs from the installed move list")
            require(all(r[2:6] == [who,addr,idx,sq] for r in rest),"ticket actor/move changed")
            children = []
            while rest and rest[0][0] == "sample":
                alpha,child,*rest = rest
                require(child[0] == "child" and alpha[1] < child[-1] < child[1],"unpaired alpha/child call")
                children.append((alpha,child))
            require(children and rest and rest[0][0] in ("decision","discard"),"missing lock outcome")
            decision,*changed = rest
            require(len(changed) <= 1 and all(r[0] == "wrote" and r[-2:] == decision[-2:] for r in changed),
                    "score store differs from lock decision")
            packets.append((decision,claim,children,changed[0] if changed else None))
        packets.sort(key=lambda packet:packet[0][1])
        observation = owner,prior,packets,winner,tie,evidence
        if scout_case:
            scouts.append(observation)
        else:
            wides.append(observation)
    # Both generations use the identical live allocation, owner and physical bank.
    # Choose the highest observed selectivity and prefer a changed alpha value.
    pairs = [(s,w) for s in scouts for w in wides if s[0][2][4:6] == w[0][2][4:6]]
    if not pairs:
        return None
    def rank(pair):
        owner,_,packets,winner,_,_ = pair[1]
        children = next(path for _,claim,path,_ in packets if claim[4] == winner[4])
        return owner[0][12],children[0][0][-1] != children[1][0][-1]
    chosen = max(pairs,key=rank)
    for owner,_,_,_,_,_ in chosen:
        w,load,ret = owner[0][2],owner[4][1],owner[-1][1]
        require([r for r in rows if r[2]==w and load <= r[1] <= ret] == owner[4:],"non-adjacent owner return")
    return chosen


# Absolute stamps come from the native clock. The parent opens the scoped slice,
# so its own caller is outside the trace. Only the observed terminal events from
# this owner are used by ownerWindow. The same physical generation supplies all
# lock decisions, child returns and best-score stores in the publisher replay.
MODEL = """
import NeuralReversi.PvsParallel.Interleaving.JournalHelpers
import NeuralReversi.PvsParallel.Interleaving.BestPublisher
namespace NeuralReversi.PvsParallel.Interleaving.NativeSplitOwner
open PvsBoardTree (Position)
private def b : Position := ⟨⟨PLAYER,OPPONENT⟩,by unfold Board.wellFormed; decide⟩
private def args : NativeFrames.Args := ⟨b.val,DEPTH,⟨BitVec.ofInt 32 (ALPHA)⟩,⟨BitVec.ofInt 32 (BETA)⟩,
  true,LEVEL,PV,CUT,none⟩
private def events (wrongOrder : Bool) : Joint.Tape Unit :=
  let openFrame := fun step site node registers =>
    (Joint.Event.engine (.localCall WORKER (.entry step site node .none none) registers) : Joint.Event Unit)
  let splitExit : Joint.Event Unit := .engine (.localCall WORKER (.leave LEAVE CALL .split none none) none)
  let choice : Joint.Event Unit := .engine (.localCall WORKER (.scored SCORED ⟨BitVec.ofInt 32 (CHOSEN)⟩) none)
  (BEFORE : Joint.Tape Unit) ++ [openFrame OPENED 0 .pvs (some args)] ++ (DURING : Joint.Tape Unit) ++ [openFrame CALL 5 .split none,
   .splitWord WORKER MARK (.mark BANK),.splitWord WORKER LOAD (.result BANK (BitVec.ofInt 32 (VALUE)))] ++
   (if wrongOrder then [choice,splitExit] else [splitExit,choice]) ++
   [.engine (.localCall WORKER (.leave FINISH OPENED .pvs none (some ⟨BitVec.ofInt 32 (RETURN)⟩)) none)]
private def accepted (wrongOrder : Bool) (bank mark : Nat) (v : Int) := (show Option Unit from do
  let es := events wrongOrder
  let state ← CallReturns.captureTimed (es.map Joint.Event.frameSample)
  let [r] := state.allRows | none
  let activation ← state.calls[r.id]?
  guard (r.score = (RETURN : Int) ∧ activation.first = OPENED ∧ r.returned = some (es.length-1))
  let sc : CallReturns.Scope := ⟨state.allRows,r.id,r.chosen,state.calls⟩
  guard (JournalHelpers.ownerWindow es sc b false PV CUT LEVEL DEPTH (BETA) bank
    ⟨WORKER,mark⟩ ⟨WORKER,LOAD⟩ v false)
  return ()) |>.isSome
#guard accepted false BANK MARK (VALUE)
#guard !accepted true BANK MARK (VALUE)
STALECONTROL
#guard !accepted false (BANK+1) MARK (VALUE)
#guard !accepted false BANK MARK (VALUE+1)
end NeuralReversi.PvsParallel.Interleaving.NativeSplitOwner
"""


TIE_CONTROL = """
-- The later tied ticket has the same score and a distinct lock acquisition.
example : TIEVALUE = (VALUE : Int) ∧
    winning ≠ some (⟨⟨TIEWORKER,TIELOCK⟩,BitVec.ofInt 32 (TIEVALUE)⟩, some TIEINDEX) := by decide +kernel
"""

WIDE_CONTROL = """
private def wideRegisters : Option (Nat × Int × Int × Nat × Int × Int) := do
  let (writer,some idx) ← winning | none
  let ticket ← tickets.find? (fun (t : Ticket) => t.index == idx && t.worker == writer.event.worker)
  let [scout,wide] := ticket.children | none
  guard (PV && !scout.pv && wide.pv && scout.score != wide.score && decide (1 < wide.hi-wide.lo))
  guard (scout.returned < wide.sampled ∧ wide.sampled < wide.entered)
  return (scout.sampled,scout.alpha,scout.score,wide.sampled,wide.alpha,wide.score)
example : wideRegisters = some (SCOUTREAD,(OLDALPHA : Int),(SCOUTRAW : Int),
    WIDEREAD,(FRESHALPHA : Int),(WIDERAW : Int)) := by decide +kernel
example : acceptsSource WINNER WRITE INDEX (-(SCOUTRAW : Int)) = false := by decide +kernel

private def staleRead (t : Ticket) : Ticket :=
  if t.index == INDEX then
    match t.children with
    | [scout,wide] => {t with children := [scout,{wide with sampled := scout.sampled}]}
    | _ => t
  else t
example : winningWith (tickets.map staleRead) = none := by decide +kernel
"""

def replay(owner, previous, packets, winner, tie, directory, other_mark):
    parent, call, mark, seed, load, leave, _, scored, returned = owner
    _, opening, worker, _, player, opponent, depth, alpha, beta, pv, _, cut, level, _ = parent
    def flag(n):
        return str(bool(n)).lower()
    def children(rows):
        kept=[]
        for sample,child in rows:
            _,stop,w,addr,idx,sq,p,o,d,wide,lo,hi,c,v,start=child
            kept.append(f"⟨{w},{addr},{idx},{sq},{sample[1]},{sample[6]},({sample[7]}),{start},{stop},"
                        f"⟨{p},{o}⟩,{d},{flag(wide)},({lo}),({hi}),{flag(c)},({v})⟩")
        return "["+",".join(kept)+"]"
    kept=[]
    for decision,claim,rows,store in packets:
        _,tick,w,address,index,sq=claim
        result="none" if decision[0]=="discard" else f"some (({decision[-2]}),({decision[-1]}))"
        stored=f"some ⟨⟨{store[2]},{store[1]}⟩,BitVec.ofInt 32 ({store[-1]})⟩" if store else "none"
        kept.append(f"⟨{w},{address},{index},{sq},{tick},{children(rows)},{decision[1]},{result},{stored}⟩")
    fields = dict(PLAYER=player, OPPONENT=opponent, DEPTH=depth, ALPHA=alpha, BETA=beta, LEVEL=level,
                  PV=flag(pv), CUT=flag(cut), WORKER=worker,
                  BANK=mark[4], MARK=mark[1], OPENED=opening, CALL=call[1], LOAD=load[1], VALUE=load[-1],
                  LEAVE=leave[1], SCORED=scored[1], CHOSEN=scored[-1], FINISH=returned[1], RETURN=returned[-1],
                  START=seed[8],MOVES=str(seed[9:]),ADDRESS=mark[5],SEED=seed[1],INITIAL=seed[6],
                  PACKETS="["+ ",\n".join(kept)+"]",WINNER=winner[2],WRITE=winner[1],INDEX=winner[4],
                  OTHERWORKER=other_mark[2], OTHERBANK=other_mark[4], OTHERADDRESS=other_mark[5], OTHERMARK=other_mark[1])
    prior_word = [f".splitWord {worker} {previous[1]} (.mark {mark[4]})"] if previous else []
    before_parent = previous is not None and previous[1] < opening
    fields.update(BEFORE="["+",".join(prior_word if before_parent else [])+"]",
                  DURING="["+",".join(prior_word if not before_parent else [])+"]",
                  STALECONTROL=(f"#guard !accepted false {mark[4]} {previous[1]} ({load[-1]})" if previous else ""))
    winning_children = next(path for _,claim,path,_ in packets if claim[4] == winner[4])
    if pv:
        first, second = winning_children
        fields.update(SCOUTRAW=first[1][-2], WIDERAW=second[1][-2],
                      SCOUTREAD=first[0][1], WIDEREAD=second[0][1],
                      OLDALPHA=first[0][-1], FRESHALPHA=second[0][-1])
        control = WIDE_CONTROL
    else:
        fields.update(TIEVALUE=tie[-1],TIEWORKER=tie[2],TIELOCK=tie[1],TIEINDEX=tie[4])
        control = TIE_CONTROL
    source = directory / "NativeSplitOwner.lean"
    appendix=(Path(__file__).parent/"serial_calls/SplitPublisher.lean.in").read_text()
    program=MODEL.replace("end NeuralReversi.PvsParallel.Interleaving.NativeSplitOwner",
                          appendix.replace("CASECONTROL",control)+"\nend NeuralReversi.PvsParallel.Interleaving.NativeSplitOwner")
    source.write_text(re.sub(r'\b[A-Z]+\b', lambda m: str(fields.get(m[0], m[0])), program))
    subprocess.run(["lake", "env", "lean", "-DwarningAsError=true", str(source)], cwd=LEAN, check=True)
    print(f"Hard-20 #166, {'PV' if pv else 'NonPV'}, level {level}, depth {depth}, owner {worker}, bank {mark[4]} at {mark[5]:#x}, "
          f"mark {mark[1]}, seed {seed[6]}. Helper {winner[2]} ticket {winner[4]} square {winner[5]} "
          f"stored {winner[-2]} -> {winner[-1]} at {winner[1]}.")
    if pv:
        print(f"Scout raw {first[1][-2]} from window [{first[1][10]},{first[1][11]}], "
              f"alpha {first[0][-1]} at {first[0][1]}; wide raw {second[1][-2]} from "
              f"[{second[1][10]},{second[1][11]}], fresh alpha {second[0][-1]} at {second[0][1]}.")
    else:
        print(f"Later equal-score ticket {tie[4]} by worker {tie[2]} at {tie[1]} made no store.")
    print(f"Final load {load[1]}, split leave {leave[1]}, parent choice {scored[1]}, return {returned[1]}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write the selected native receipts")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reversi-native-split-owner-") as tmp:
        directory = Path(tmp)
        binary = capture(directory, args.target_dir.resolve(), instrument, fixture=FIXTURES / "parallel_tickets.rs",
                         serial_trace=False)
        for _ in range(3):
            result = subprocess.run([binary, REPO], text=True, capture_output=True, check=True)
            found = select(result.stdout)
            if found is not None:
                break
        else:
            raise AssertionError("three runs produced no scout/tie and wide-research witnesses on the same bank")
        if args.output:
            selected = {r[1]: r for (_,previous,_,_,_,evidence) in found for r in [previous,*evidence] if r is not None}
            args.output.write_text("".join("O " + " ".join(map(str,r)) + "\n" for _,r in sorted(selected.items())))
        for i, (owner,previous,packets,winner,tie,_) in enumerate(found):
            other_mark = found[1-i][0][2]
            replay(owner,previous,packets,winner,tie,directory,other_mark)


if __name__ == "__main__":
    main()

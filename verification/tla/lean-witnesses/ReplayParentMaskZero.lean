import NeuralReversi.PvsParallel.Interleaving.Mask
import NeuralReversi.PvsParallel.Interleaving.SlotSynchronization

namespace ParentMaskZero

open Lean NeuralReversi NeuralReversi.Threading NeuralReversi.PvsParallel
open AtomicWord (Event Write InReleaseSequence SequencedBefore)
open SplitCompletion (Worker Action)
open SlotFrameMir (RawMask ClosedAfterFirst)
open SlotFrameMir.RawProjection (lastEffective zero_source_iff rf back)
open Interleaving

structure Snapshot where
  phase : String
  pending : List Nat
  finishes : List Nat
  words : List Nat
  ownerFinish : Int
  effective : Int
  duplicate : Int
  source : Int
  deriving FromJson

private def generation (base : Nat) (actions : List Action) : Mask.Generation :=
  ⟨(base + ·),by omega,actions⟩

private def interval (g : RawMask ws) := (List.finRange ws.length).filter fun i =>
  (ws.get (lastEffective g)).value == 0 && (lastEffective g).val ≤ i.val &&
    i.val ≤ g.mark.val + g.actions.length

theorem mem_interval (g : RawMask ws) (closed : ClosedAfterFirst g) (i : Fin ws.length) :
    i ∈ interval g ↔ g.mark.val ≤ i.val ∧ i.val ≤ g.mark.val + g.actions.length ∧
      (ws.get i).value = 0 := by
  simpa [interval, and_assoc] using (zero_source_iff g closed i).symm

private def releaseSequence (ws : List Write) (a b : Fin ws.length) : Bool :=
  @decide (InReleaseSequence ws a b) (by unfold InReleaseSequence; infer_instance)

private def sb (x y : Event) : Bool :=
  @decide (SequencedBefore x y) (by unfold SequencedBefore; infer_instance)

/-- All writes in this litmus use the same physical split lock. The read does
    not take it. In this finite graph an incoming HB path has as its last edge
    either owner program order or release-sequence synchronization. Outbound
    paths start with owner program order. The mutex orders preceding and
    succeeding stores in both cases. -/
private def coherent (ws : List Write) (r : Event) (src : Fin ws.length) : Bool :=
  let all := List.finRange ws.length
  all.all fun i =>
    let before := all.any fun j => i.val ≤ j.val &&
      (sb (ws.get j).event r || releaseSequence ws j src)
    let after_ := all.any fun j => j.val ≤ i.val && sb r (ws.get j).event
    (!before || i.val ≤ src.val) && (!after_ || src.val < i.val)

-- Place lock/unlock sites on either side of each model write. Only consecutive
-- critical sections hand off this mutex. The final mask acquire also synchronizes
-- with every release head of its selected physical source.
private def onTape (ws : List Write) (r : Event) (src : Fin ws.length) : Bool := Id.run do
  let atWrite (e : Event) : Event := ⟨e.worker,4*e.step+1⟩
  let reader := atWrite r
  let mut regs : Joint.Tape Unit := []
  let mut old : BitVec 64 := 0
  for w in ws do
    let e := atWrite w.event
    if w.event == ⟨0,53⟩ then
      regs := regs ++ [.maskWord reader.worker (reader.step-1) (.shutdown 0 false),
        .maskWord reader.worker reader.step (.load 0 (atWrite (ws.get src).event) 0 true),
        .maskWord reader.worker (reader.step+1) (.returned 0)]
    let row : Joint.Event Unit ← if !w.rmw then pure (.maskWord e.worker e.step (.clear 0))
      else if w.release then pure (.splitWord e.worker e.step (.maskRmw 0 0 e.worker false old w.value))
      else if w.value = 1 then pure (.splitWord e.worker e.step (.mark 0))
      else do
        let [who] := (List.range 64).filter fun k => (old ^^^ w.value).getLsbD k | return false
        pure (.splitWord e.worker e.step (.maskRmw 0 0 who true old w.value))
    regs := regs ++ [.splitWord e.worker (e.step-1) (.lock 0 0),row,
      .splitWord e.worker (e.step+1) (.unlock 0 0)]
    old := w.value
  let base : CutoffVisibility.History := ⟨[],fun _ => [],[],[],fun _ _ => False⟩
  let some ⟨_,c⟩ := SlotSynchronization.capture base regs [] [] | return false
  let i := MaskGeneration.input c regs (fun o b => if o=0 ∧ b=0 then some 0 else none)
  return (i.generation ⟨0,atWrite ⟨0,4⟩⟩).any fun g => g.wait == reader

private def require (ok : Bool) (why : String) : Except String Unit :=
  unless ok do throw why

private def check (mode : String) (trace : List Snapshot) : Except String String := do
  let [start, one, two, empty, chosen, doneState] := trace | throw "wrong trace length"
  let mut resets : List Action := []
  let mut active := [0,1,2]
  let mut finished : List Nat := []
  let mut ownerRank : Int := -1
  let mut words : List Nat := [0,1,0,0,1,3,7]
  require (start.phase == "retiring" && start.words == words && start.pending == active &&
    start.finishes.isEmpty && start.duplicate == -1 && start.source == -1) "invalid initial generation"
  for row in [one,two,empty] do
    let [n] := active.filter (!row.pending.contains ·) | throw "no single participating worker retired"
    if h : n < 64 then
      resets := resets ++ [.finish ⟨n,h⟩ ⟨n,words.length⟩]
    else throw "invalid worker"
    active := active.erase n
    finished := finished ++ [n]
    if n == 0 then ownerRank := words.length
    words := words ++ [active.foldl (fun bits w => bits ||| (1 <<< w)) 0]
    require (row.phase == (if active.isEmpty then "empty" else "retiring") &&
      row.effective == (if active.isEmpty then Int.ofNat (words.length-1) else -1) &&
      row.ownerFinish == ownerRank && row.words == words && row.pending == active &&
      row.finishes == finished && row.duplicate == -1 && row.source == -1) "wrong reset update"
  let extra := chosen.duplicate
  if extra != -1 then
    require ((mode == "raw" && (extra == 1 || extra == 2)) || (mode == "ownerRaw" && extra == 0))
      "unexpected extra reset"
    if extra == 0 then ownerRank := words.length
    let w := extra.toNat
    if h : w < 64 then resets := resets ++ [.finish ⟨w,h⟩ ⟨w,words.length⟩]
    else throw "invalid extra worker"
  let prior := (generation 0 [.finish 0 ⟨0,2⟩]).words 0
  let current := generation 3 ([.book 1 ⟨0,5⟩,.book 2 ⟨0,6⟩] ++ resets)
  let next := [generation 53 []]
  let ws := Mask.tape prior 0 current next
  let g := Mask.raw prior 0 current next
  let rows := ws.map fun a => a.value.toNat
  let r : Event := ⟨0,50⟩
  require (chosen.phase == "choose" && doneState.phase == "done" && chosen.source == -1 &&
    [chosen,doneState].all fun st => st.finishes == finished && st.pending.isEmpty &&
      st.words == rows && st.duplicate == extra && st.effective == empty.effective &&
      st.ownerFinish == ownerRank) "physical mask tape differs"
  let some src := (List.finRange ws.length)[doneState.source.toNat]? | throw "invalid source"
  require (doneState.source == Int.ofNat src.val) "invalid signed rank"
  let input := Interleaving.OwnerMaskGuards.Operand.mask r (ws.get src).event 0 true
  match input.guard ws with
  | some (.mask rank) => require (rank == src) "RF event resolves to a different rank"
  | _ => throw "RF has no matching zero and acquire"
  let all := List.finRange ws.length
  let ok := all.filter fun i => (ws.get i).value == 0 && coherent ws r i
  let decoded := all.filter fun i => (ws.get i).value == 0 && onTape ws r i
  require (ok == decoded) "public retention checker and independent atomic enumeration disagree"
  require (@decide (ClosedAfterFirst g) (by unfold ClosedAfterFirst; infer_instance))
    "booking after first finish invalidates the zero interval theorem"
  let candidates := interval g
  let expected := if extra == 0 then candidates.filter (fun i => ownerRank ≤ Int.ofNat i.val)
    else candidates
  require (ok == expected && (lastEffective g).val == empty.effective.toNat)
    "atomic constraints and generation interval disagree"
  if extra == 0 then
    require (candidates.length == 2 && ok.length == 1) "owner reset did not tighten the read"
  let inRange := ok.contains src
  if mode == "valueOnly" then
    require (!inRange && (src.val == 2 || src.val == ws.length-2)) "missing wrong lifetime"
  else
    require (inRange && back g (rf g src) == lastEffective g) "incorrect effective release"
    for i in all do
      if (ws.get i).release && g.mark.val < i.val && i.val ≤ (lastEffective g).val then
        require (releaseSequence ws i src) "missing effective release in synchronization"
  return s!"physical RF {src.val}, last effective {(lastEffective g).val}, generation zeros {candidates.map Fin.val}, coherent zeros {ok.map Fin.val}"

def main (args : List String) : IO Unit := do
  let [mode,path] := args | throw (IO.userError "expected policy and TLC snapshots")
  let json ← IO.ofExcept (← Json.parse <$> IO.FS.readFile path)
  IO.println (← IO.ofExcept (fromJson? json >>= check mode))

end ParentMaskZero

def main := ParentMaskZero.main

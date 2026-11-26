import NeuralReversi.PvsParallel.SlotFrameMir.Program
import NeuralReversi.Threading.SplitReturn.Observation

open Lean NeuralReversi NeuralReversi.PvsParallel NeuralReversi.Threading
open CutoffVisibility (FlagWrite FlagRead History Walk)

namespace NestedCutoff

structure Word where
  generation : Nat
  value : Bool
  deriving FromJson, BEq
structure Job where
  generation : Nat
  parentSeed : Nat
  childSeed : Nat
  deriving FromJson, BEq
structure Snapshot where
  phase : String
  generation : Nat
  parent : List Word
  child : List Word
  bookings : List Job
  parentMask : List Nat
  helperActive : Int
  restored : Int
  assigned : Nat
  usedAssignment : Nat
  seen : Nat
  epoch : Nat
  cached : Bool
  lastSlot : Int
  lastSource : Int
  previousParentSource : Int
  nestedStop : Bool
  expectedNested : Bool
  afterStop : Bool
  parentSource : Int
  pollKind : String
  deriving FromJson, BEq

private def initial : Snapshot :=
  ⟨"start",0,[],[],[],[],-1,-1,0,0,0,1,false,-1,-1,-1,false,false,false,-1,""⟩

-- Independent of the generation labels on flag writes: a booking ends at the
-- next booking's slot initializer. Both polls are between helpful book and detach.
private def lastRank (s : Snapshot) (slot job : Nat) : Option Nat := do
  let chosen ← s.bookings[job]?
  let seed := if slot == 0 then Job.parentSeed else Job.childSeed
  let tape := if slot == 0 then s.parent else s.child
  let upper := ((s.bookings[job+1]?).map seed).getD tape.length
  if seed chosen < upper && upper ≤ tape.length then some (upper-1) else none

private def flags (slot : Nat) (tape : List Word) : List FlagWrite :=
  tape.map fun w => ⟨⟨if w.value then 2+slot else slot,32*w.generation⟩,w.value⟩

-- Arms use different physical rows even when the loaded bool is the same.
private def readAt (s : Snapshot) (slot rank : Nat) : Option { r : FlagRead (fun i =>
    if i == 0 then flags 0 s.parent else flags 1 s.child) // r.slot = slot } := do
  let tape := if slot == 0 then flags 0 s.parent else flags 1 s.child
  let tick := 32*s.generation + (if s.phase == "first" then 10 else 20)
  if within : rank < tape.length then some ⟨⟨slot,⟨0,tick + (1-slot)⟩,⟨rank,within⟩⟩,rfl⟩ else none

private def nested (s : Snapshot) (old : Bool) : Except String Snapshot := do
  let job := if old then 0 else s.bookings.length-1
  let some parent := (lastRank s 0 job).bind (readAt s 0) | throw "missing parent source"
  let some child := (lastRank s 1 job).bind (readAt s 1) | throw "missing child source"
  let path : {reads // Walk _ [child.val.slot,parent.val.slot] reads} :=
    ⟨if child.val.value then [child.val] else [child.val,parent.val],by
      cases c : child.val.value <;> simp only [Bool.false_eq_true,ite_false,ite_true]
      · cases p : parent.val.value
        · exact .descend _ c (.descend _ p .empty)
        · exact .descend _ c (.found _ [] p)
      · exact .found _ _ c⟩
  let loaded := path.val
  let oldCache : SplitCutoff.Cache := ⟨BitVec.ofNat 64 s.seen,s.cached⟩
  let epoch : SplitCutoff.Epoch := BitVec.ofNat 64 s.epoch
  let update := SplitCutoff.poll oldCache ⟨epoch,loaded.map FlagRead.value⟩
  let touched := !oldCache.latched && epoch != oldCache.seen
  let row := loaded.find? (·.value)
  let label := if touched then "walk" else if s.cached then "latch" else "equalEpoch"
  let some chosen := s.bookings[job]? | throw "missing booking"
  let some nowP := s.parent.getLast? | throw "missing current parent"
  let some nowC := s.child.getLast? | throw "missing current child"
  return { s with
    phase := if s.phase == "first" then "second" else "retire"
    usedAssignment := chosen.generation, seen := update.seen.toNat, cached := update.latched,
    lastSlot := if touched then (row.map (fun r => Int.ofNat r.slot)).getD (-1) else s.lastSlot,
    lastSource := if touched then (row.map (fun r => Int.ofNat r.source.val)).getD (-1) else s.lastSource,
    nestedStop := update.latched, expectedNested := nowP.value || nowC.value, pollKind := label}

private def afterStop (s : Snapshot) : Option Bool := do
  let slot := SlotFrameMir.restored (some (0,0))
  let some parent := slot | none
  let src := s.parent.length - 1
  let fs := fun _ : Nat => flags 0 s.parent
  if h : src < (fs parent.2).length then
    let tick := 32*s.generation+26
    let r : FlagRead fs := ⟨parent.2,⟨1,tick+1⟩,⟨src,h⟩⟩
    let hist : History := ⟨[],fs,[],[r],fun _ _ => False⟩
    have path : Walk fs [r.slot] [r] := by
      cases val : r.value
      · exact .descend r val .empty
      · exact .found r [] val
    let scan : SplitReturn.Observation.AncestorScan hist :=
      ⟨⟨1,tick⟩,[r.slot],[r],path,by
        intro read mem
        obtain rfl := List.mem_singleton.mp mem
        exact ⟨by simp [hist],rfl,Nat.lt_succ_self tick⟩⟩
    -- Generation 1, requested generation 0; no external or helper-job abort.
    let abort : SplitReturn.Observation.AbortObservation := ⟨[⟨⟨1,0⟩,(1#32 ++ 0#32),false,false⟩],⟨⟨1,tick+2⟩,0,false⟩⟩
    return (⟨scan,some abort,by simp,false⟩ : SplitReturn.Observation.OwnerCheck hist).stopped
  else none

private def transition (s next : Snapshot) (old : Bool) : Except String Snapshot := do
  if s.phase == next.phase then
    let parent := s.parent != next.parent
    let tape := if parent then s.parent else s.child
    unless ["first","second"].contains s.phase || (parent && s.phase == "after") do throw "mark outside live interval"
    let some prev := tape.getLast? | throw "mark without initializer"
    let (first,word) := SplitCutoff.mark prev.value
    unless first do throw "duplicate mark"
    let tape' := tape ++ [⟨s.generation,word⟩]
    return { s with
      epoch := (SplitCutoff.publish first (BitVec.ofNat 64 s.epoch)).toNat,
      parent := if parent then tape' else s.parent, child := if parent then s.child else tape' }
  match s.phase with
  | "done" =>
    unless s.generation == 1 do throw "both activations complete"
    return {s with phase := "closed", parentMask := [], restored := -1}
  | "start" | "closed" =>
    unless s.generation < 2 && s.parentMask.isEmpty do throw "parent slot still occupied"
    let g := s.generation+1
    return { s with
      phase := "parent", generation := g, helperActive := 0, restored := -1, parentMask := [0,1,2],
      parent := s.parent ++ [⟨g,false⟩], nestedStop := false, afterStop := false,
      previousParentSource := if s.nestedStop && s.lastSlot == 0 then s.lastSource else -1,
      parentSource := -1, assigned := 0, usedAssignment := 0 }
  | "parent" =>
    unless s.parentMask.contains 1 do throw "nested owner is not a helper"
    return { s with
      phase := "child", child := s.child ++ [⟨s.generation,false⟩] }
  | "child" =>
    let cleared := SplitCutoff.reset (BitVec.ofNat 64 s.epoch)
    return { s with
      phase := "idle", parentMask := s.parentMask.erase 0, helperActive := -1,
      seen := cleared.seen.toNat, cached := cleared.latched, lastSlot := -1, lastSource := -1, pollKind := "" }
  | "idle" =>
    unless WorkerFrameMir.canJoin (s.helperActive != -1) false true true (s.parentMask.contains 1) do
      throw "helpful-owner booking denied"
    let job := ⟨s.generation,s.parent.length-1,s.child.length-1⟩
    return { s with
      phase := "first", helperActive := 1, assigned := s.generation, bookings := s.bookings ++ [job] }
  | "first" | "second" => nested s old
  | "retire" => return {s with phase := "after", helperActive := -1, restored := 0}
  | "after" =>
    unless s.restored == 0 do throw "restored a different parent"
    let some stopped := afterStop s | throw "missing restored parent"
    return {s with phase := "done", parentSource := (Int.ofNat (s.parent.length-1)), afterStop := stopped}
  | _ => throw "unknown phase"

def replay (rows : List Snapshot) (mode : String) : Except String Snapshot := do
  unless mode == "step" || mode == "staleAssignment" do throw "unknown mode"
  let first :: tail := rows | throw "empty witness"
  unless first == initial do throw "initial state differs"
  let last ← tail.foldlM (fun s next => do
    let expected ← transition s next (mode == "staleAssignment")
    unless expected == next do throw s!"{s.phase} → {next.phase}: assignment, flag or cache register differs"
    pure next) first
  unless last.phase == "done" && last.generation == 2 do throw "second restored-parent check missing"
  return last

def main (args : List String) : IO Unit := do
  let [mode,path] := args | throw (IO.userError "expected policy and TLC snapshots")
  let json ← IO.ofExcept (← Json.parse <$> IO.FS.readFile path)
  let last ← IO.ofExcept (fromJson? json >>= fun rows => replay rows mode)
  IO.println s!"assignment {last.assigned}, selected {last.usedAssignment}, helpful poll {last.nestedStop} from ({last.lastSlot},{last.lastSource}), restored parent {last.afterStop} from (0,{last.parentSource})"

end NestedCutoff

def main := NestedCutoff.main

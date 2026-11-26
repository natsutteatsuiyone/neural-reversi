import NeuralReversi.PvsParallel.Interleaving.CallReturns
open Lean NeuralReversi NeuralReversi.PvsParallel
open Interleaving.CallReturns
open NativeFrames.EntryStream (Sample)

structure Receipt where
  id : Nat
  score : Int
  deriving FromJson, DecidableEq

structure Image where
  begins : List Nat
  returns : List Receipt
  consumed : List (List Nat)
  deriving FromJson, DecidableEq

private def blank : Image := ⟨[],[],[[],[]]⟩
private def require (b : Bool) (reason : String) : Except String Unit := unless b do throw reason
-- hard-20.obf, position 166, has minimax +6 discs. Both words are valid lower bounds.
private def board : PvsBoardTree.Position := ⟨⟨0x181c1f171300303e,0x2420e0686c7e0800⟩,rfl⟩
private def arg : NativeFrames.Args := ⟨board.val,20,⟨BitVec.ofInt 32 (-1)⟩,⟨0⟩,true,5,false,false,none⟩
private def q : SerialCalls.Request := ⟨board,20,false,false,-1,0⟩
private def sample (w : Nat) (ev : NativeFrames.Event) (args : Option NativeFrames.Args := none) : Sample :=
  ⟨.localCall w ev,args⟩
private def push (s : State) (ev : Sample) : Except String State :=
  (step s ev).elim (throw "invalid Rust stack transition") pure

private def replay (mode : String) (states : List Image) : Except String Nat := do
  let first :: tail := states | throw "empty witness"
  require (first == blank) "unexpected initial state"
  let mut s ← push {} (sample 0 (.entry 0 0 .pvs .none none) (some arg))
  s ← push s (sample 1 (.entry 0 0 .pvs .none none) (some arg))
  let mut prev := first
  let mut errors := 0
  let mut aliases := 0
  for now in tail do
    let b := prev.begins.length
    let r := prev.returns.length
    if let some w := now.begins[b]? then
      require (w < 2 && now == {prev with begins := prev.begins ++ [w]}) "corrupt call entry"
      require ((s.workers w).stack == [w]) "overlapping calls in one Rust worker"
      require ((prev.consumed[w]?.getD []).length == (prev.begins.filter (· == w)).length)
        "next synchronous call started before consuming the previous result"
      s ← push s (sample w (.entry (s.workers w).nextStep 6 .pvs .none none) (some arg))
    else if let some out := now.returns[r]? then
      require (now == {prev with returns := prev.returns ++ [out]} && (out.score == 512 || out.score == 1024)) "corrupt return"
      let some w := now.begins[out.id]? | throw "result for unknown call"
      let some a := s.calls[out.id+2]? | throw "missing Rust activation"
      require ((s.workers w).stack.head? == some (out.id+2)) "closing wrong activation"
      let v : ScoreRefresh.Score := ⟨BitVec.ofInt 32 out.score⟩
      s ← push s (sample w (.scored (s.workers w).nextStep v))
      s ← push s (sample w (.leave (s.workers w).nextStep a.first .pvs none (some v)))
    else
      let some w := [0,1].find? (fun w => prev.consumed[w]? != now.consumed[w]?) | throw "missing consumption"
      let used := prev.consumed[w]?.getD []
      let some chosen := (now.consumed[w]?.getD [])[used.length]? | throw "missing consumed ID"
      require (now == {prev with consumed := prev.consumed.set w (used ++ [chosen])}) "corrupt consumption"
      let scope : Scope := ⟨s.rows true,w,s.tick,s.calls⟩
      let some source := scope.atSite used.length .pvs 6 q | throw "missing actual ordinal"
      let receipts := now.returns.filter (fun o => now.begins[o.id]? == some w)
      let some real := receipts[used.length]? | throw "TLC has no expected return"
      require (source.id == real.id+2 && source.score == real.score) "TLC and Lean disagree"
      let some picked := now.returns.find? (fun o => o.id == chosen) | throw "no returned score for choice"
      let predicted := match mode with
        | "step" => some real.id
        | "firstInParent" => receipts.head?.map Receipt.id
        | "firstAcrossParents" | "equalAcrossParents" => now.returns.head?.map Receipt.id
        | _ => none
      if source.id != picked.id+2 && source.score == picked.score then
        aliases := aliases+1
        require (mode != "step") "equal score from a different invocation"
      require (predicted == some chosen) "choice does not follow the stipulated rule"
      if source.score != picked.score then errors := errors+1
    prev := now
  if mode == "step" then
    require (prev.consumed.all (fun rows => rows.length == 2) && prev.begins.take 2 == [0,1] &&
      (prev.returns.head?.map Receipt.id) == some 1) "required overlapping return missing"
    let words := prev.returns.filter (fun o => prev.begins[o.id]? == some 0)
    require ((words.head?.map Receipt.score) != (words.getLast?.map Receipt.score)) "required unequal scores missing"
  if mode == "equalAcrossParents" then
    require (aliases > 0 && errors == 0) "missing equal score from another worker"
  else
    require ((mode == "step") == (errors == 0)) "expected score difference missing"
  return errors

def main (args : List String) : IO Unit := do
  let [mode,path] := args | throw (IO.userError "expected policy and JSON trace")
  let v ← IO.ofExcept (Json.parse (← IO.FS.readFile path))
  let rows ← IO.ofExcept (fromJson? v : Except String (List Image))
  IO.println s!"{mode}: {rows.length} snapshots; {← IO.ofExcept (replay mode rows)} incorrect scores"

import NeuralReversi.PvsParallel.Interleaving.CutClosure
import NeuralReversi.PvsParallel.Interleaving.MultiSessionCarry

namespace CutBoundaryReplay

open Lean NeuralReversi NeuralReversi.PvsParallel.Interleaving
open CutClosure (minimalCut)
open JournalResident (checkDrained)

structure Snapshot where
  phase : String
  cutK : Int
  rfClosed : Bool
  session1Drained : Bool
  session2ReadInherited : Bool
  deriving FromJson

private def litmusHB : List (Nat × Nat) := [(0,1),(1,2),(2,6),(3,4),(4,5)]
private def litmusRF : List (Nat × Nat) := [(6,4)]

private def require (p : Bool) (message : String) : Except String Unit :=
  unless p do throw message

private def replay (mode : String) (trace : List Snapshot) : Except String String := do
  let some last := trace.getLast? | throw "empty TLC trace"
  let k : Nat := last.cutK.toNat
  let closedK := minimalCut litmusHB litmusRF 7 k
  let hbOnlyK := minimalCut litmusHB [] 7 k
  let records : List NeuralReversi.PvsParallel.NativeTableWrites.Journal.Record :=
    if last.session1Drained then
      [.writer ⟨0,.cas⟩,.writer ⟨0,.fence⟩,.writer ⟨0,.field .player⟩,
       .writer ⟨0,.field .opponent⟩,.writer ⟨0,.field .metadata⟩,.writer ⟨0,.finish⟩]
    else
      [.writer ⟨0,.cas⟩,.writer ⟨0,.fence⟩,.writer ⟨0,.field .player⟩]
  let drainedOK := checkDrained (fun _ => true) records
  let cutValid := (k == closedK) && drainedOK
  match mode with
  | "step" =>
    require (last.phase == "done" && cutValid && last.rfClosed && last.session1Drained)
      "normal CutBoundary trace failed Lean RF+HB closure or drained handoff check"
    return s!"normal: cutK={k}, minimalCut={closedK}, drained={drainedOK}, inheritedHit={last.session2ReadInherited}"
  | "hbOnlyCut" =>
    require (k == 6 && hbOnlyK == 6 && closedK == 7 && !cutValid)
      "expected HB-only cut k=6 to violate RF closure and expand to k'=7"
    return s!"hbOnlyCut k={k} rejected (hbOnly={hbOnlyK}, minimalCut expands to {closedK})"
  | "undrainedHandoff" =>
    require (!last.session1Drained && !drainedOK && !cutValid)
      "expected undrained session handoff to be rejected by JournalResident.checkDrained"
    return "undrainedHandoff rejected by Lean JournalResident.checkDrained"
  | _ => throw s!"unknown mode {mode}"

def main (args : List String) : IO Unit := do
  let [mode,path] := args | throw (IO.userError "expected mode and TLC snapshots")
  let json ← IO.ofExcept (← Json.parse <$> IO.FS.readFile path)
  IO.println (← IO.ofExcept (fromJson? json >>= replay mode))

end CutBoundaryReplay

def main := CutBoundaryReplay.main

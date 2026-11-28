module
public import NeuralReversi.PvsParallel.NativeStopReturns

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeStopReturns

open NativeFrames (Record)
open NativeLeafBranches (BoundaryResult)
open ScoreRefresh (Score)

abbrev Evidence (m : Memory) := Nat → Nat → Nat → Option (Trace m)

def one (i : Inputs) (supply : Evidence m) (worker opening ended : Nat) (raw : Score) : Bool :=
  match supply worker opening ended with
  | none => false
  | some tr =>
    let frame := tr.frame
    frame.worker == worker && frame.opening == opening && frame.finish == ended &&
      tr.decode i == some ⟨raw,false⟩

def check (i : Inputs) (supply : Evidence m) : Bool :=
  i.stream.all fun
    | .localCall worker (.discarded ended opening raw) => one i supply worker opening ended raw
    | _ => true

def normalize (i : Inputs) (supply : Evidence m) : Option (List Record) :=
  if check i supply then some i.stream else none

theorem preserves_stream (ok : normalize i supply = some stream) : stream = i.stream := by
  unfold normalize at ok
  split at ok <;> simp_all

theorem discarded_source {i : Inputs} {supply : Evidence m}
    (ok : normalize i supply = some stream)
    (member : Record.localCall worker (.discarded ended opening raw) ∈ stream) :
    ∃ tr, supply worker opening ended = some tr ∧ tr.frame.worker = worker ∧ tr.frame.opening = opening ∧
      tr.frame.finish = ended ∧ tr.decode i = some ⟨raw,false⟩ ∧
      ∃ t chain, tr.CutFrom t chain ∧ Cause m t chain := by
  unfold normalize at ok
  split at ok <;> try contradiction
  rename_i valid
  cases ok
  have seen := List.all_eq_true.mp valid _ member
  change one i supply worker opening ended raw = true at seen
  unfold one at seen
  split at seen <;> try contradiction
  rename_i tr selected
  simp only [Bool.and_eq_true,beq_iff_eq] at seen
  obtain ⟨t,chain,source,cause,_⟩ := cancellation_origin tr seen.2 rfl
  exact ⟨tr,selected,seen.1.1.1,seen.1.1.2,seen.1.2,seen.2,t,chain,source,cause⟩

end NeuralReversi.PvsParallel.NativeStopReturns

module
public import NeuralReversi.PvsParallel.FrameTableReads.Execution
public import NeuralReversi.PvsParallel.FrameTableReads.LookupExecution
public import NeuralReversi.PvsParallel.FrameTableReads.ProbeExecution

@[expose] public section

namespace NeuralReversi.PvsParallel.FrameTableReads.Operations.Execution

open TranspositionTable
open FrameReturns (Log)

inductive Method where
  | probe (query : ProbeRetry.Key)
  | lookup (board : PvsBoardTree.Position)

structure Call where
  method : Method
  allocation : Nat
  clusters : SeqLock.Word
  key : SeqLock.Word
  endian : Bool
  supply : Supply
  event : Nat → Nat
  origin : Nat → Nat

def Method.query : Method → ProbeRetry.Key
  | .probe q => q | .lookup b => Cutoffs.key b

def Call.returned (c : Call) : Option EntryData := match c.method with
  | .probe q =>
    match (Operations.probe c.supply c.allocation c.clusters c.key c.endian q 0).result with
    | .hit _ d => some d | .miss _ => none
  | .lookup b =>
    let pair := (Operations.lookup c.supply c.allocation c.clusters c.key c.endian b 0).pair c.supply c.endian b
    pair.1.or pair.2

def Call.ops (c : Call) := match c.method with
  | .probe q => (Operations.probe c.supply c.allocation c.clusters c.key c.endian q 0).emit c.supply c.endian q
  | .lookup b => (Operations.lookup c.supply c.allocation c.clusters c.key c.endian b 0).emit c.supply c.endian b

def Call.shortSlotHistories (c : Call) (modifications : Address → Nat) : Bool := c.ops.all fun
  | .slot r (.load .sequence .relaxed) => decide (modifications r.address < SeqLock.modulus)
  | _ => true

def Call.atomic (c : Call) (op : Op) (i : Nat) : Atomic :=
  ⟨c.event i,instruction op,c.origin i,match op with
    | .slot r (.load _ _) => c.supply.word r.address i
    | .generation a => (c.supply.generationByte a i).zeroExtend 64
    | _ => 0⟩

def Call.records (c : Call) := c.ops.zipIdx.map (fun (op,i) => c.atomic op i)

theorem Call.atIndex (c : Call) (i : Nat) : c.records[i]? = c.ops[i]?.map (fun op => c.atomic op i) := by
  simp [records,List.getElem?_zipIdx,Function.comp_def]

theorem Call.instructions (c : Call) : c.records.map Atomic.instruction = c.ops.map instruction := by
  apply List.ext_getElem?
  intro i
  simp only [List.getElem?_map,atIndex,Option.map_map]
  rfl

/-- Each reader turn issues its next instruction, while frame and writer
    events pass through to the shared search journal in their original order. -/
inductive Turn where
  | search (r : NativeTableWrites.Journal.Record)
  | reader (caller : Caller)

def dispatch (calls : Caller → Call) (cursor : Caller → Nat) : List Turn → List Record
  | [] => []
  | .search r :: rest => .search r :: dispatch calls cursor rest
  | .reader who :: rest =>
    let emitted := ((calls who).records[cursor who]?).map (Record.reader who)
    emitted.toList ++ dispatch calls (fun k => if k = who then cursor k+1 else cursor k) rest

def budget (who : Caller) (turns : List Turn) :=
  (turns.filter fun | .reader k => k == who | .search _ => false).length

theorem dispatched (calls : Caller → Call) (turns : List Turn) (cursor : Caller → Nat) (who : Caller) :
    selected who (dispatch calls cursor turns) = (((calls who).records).drop (cursor who)).take (budget who turns) := by
  induction turns generalizing cursor with
  | nil => simp [dispatch,selected,budget]
  | cons t rest ih =>
    cases t with
    | search => simpa [dispatch,selected,budget,List.filterMap_cons] using ih cursor
    | reader k =>
      simp only [dispatch,selected,List.filterMap_append]
      rw [show List.filterMap _ (dispatch calls _ rest) = _ from ih _]
      by_cases eq : who = k
      · subst k
        simp only [ite_true,budget,List.filter_cons,beq_self_eq_true,List.length_cons]
        rw [List.drop_eq_getElem?_toList_append (i:=cursor who)]
        cases h : (calls who).records[cursor who]? with
        | some a => simp
        | none =>
          have past := List.getElem?_eq_none_iff.mp h
          simp [List.drop_of_length_le (show (calls who).records.length ≤ cursor who + 1 by omega)]
      · cases (calls k).records[cursor k]? <;> simp [budget,eq,Ne.symm eq]

theorem dispatched_journal (calls : Caller → Call) (turns : List Turn) (cursor : Caller → Nat) :
    journal (dispatch calls cursor turns) = turns.filterMap (fun | .search r => some r | .reader _ => none) := by
  induction turns generalizing cursor with
  | nil => rfl
  | cons t rest ih =>
    cases t with
    | search => simp [journal,dispatch] at *; exact ih cursor
    | reader who =>
      simp only [dispatch,journal,List.filterMap_append]
      rw [show List.filterMap _ (dispatch calls _ rest) = _ from ih _]
      cases (calls who).records[cursor who]? <;> rfl

theorem complete (calls : Caller → Call) (turns : List Turn) (who : Caller) (finished : (calls who).ops.length ≤ budget who turns) :
    selected who (dispatch calls (fun _ => 0) turns) = (calls who).records := by
  rw [dispatched,List.drop_zero,List.take_of_length_le]
  simpa [Call.records] using finished

def Call.machine {log : Log mc fc} {t : Table log} (c : Call) (actual : Atomics t consumer c.records)
    (noWrap : NoWrap c.ops c.records) : Machine t consumer (issued c.ops c.records) c.supply :=
  Execution.machine c.ops c.records actual noWrap c.supply (fun r i loc mode a selected_ at_ => by
    rw [c.atIndex,selected_] at at_
    cases at_
    rfl)

private theorem Call.publishedFrom {log : Log mc fc} {t : Table log} {f : FrameReturns.Frame} (c : Call)
    (m : Machine t (log.identify f.use) issued_ c.supply) (ran : Runs issued_ 0 c.ops) (hit : c.returned = some d) :
    Published log (log.identify f.use) c.endian c.method.query d := by
  unfold Call.returned at hit
  cases kind : c.method with
  | probe q =>
    simp only [kind] at hit
    cases result : (Operations.probe c.supply c.allocation c.clusters c.key c.endian q 0).result with
    | miss => rw [result] at hit; contradiction
    | hit slot v =>
      rw [result] at hit
      cases hit
      exact probe_published (f:=f) m c.allocation c.clusters c.key c.endian q 0 (by simpa only [Call.ops,kind] using ran) result
  | lookup b =>
    simp only [kind] at hit
    exact cluster_published (f:=f) m _ _ _ _ _ _ (by simpa only [Call.ops,kind] using ran) hit

theorem Call.published {log : Log mc fc} {t : Table log} {f : FrameReturns.Frame} (c : Call)
    (actual : Atomics t (log.identify f.use) c.records) (noWrap : NoWrap c.ops c.records) (hit : c.returned = some d) :
    Published log (log.identify f.use) c.endian c.method.query d :=
  c.publishedFrom (f:=f) (c.machine actual noWrap) (run_of_instructions _ _ c.instructions) hit

/-- Reconstruct a completed call from the words of an observed instruction stream. -/
def observed (method : Method) (allocation : Nat) (clusters key : SeqLock.Word) (endian : Bool) (rs : List Atomic) : Call :=
  ⟨method,allocation,clusters,key,endian,supply rs,fun i => (rs[i]?.map Atomic.id).getD 0,fun i => (rs[i]?.map Atomic.source).getD 0⟩

theorem observed_published {log : Log mc fc} {t : Table log} {f : FrameReturns.Frame} (rs : List Atomic)
    (method : Method) (allocation : Nat) (clusters key : SeqLock.Word) (endian : Bool) :
    let c := observed method allocation clusters key endian rs
    rs.map Atomic.instruction = c.ops.map instruction → Atomics t (log.identify f.use) rs → NoWrap c.ops rs →
    c.returned = some d → Published log (log.identify f.use) endian method.query d := by
  intro c instructions memory short hit
  exact c.publishedFrom (f:=f) (fromRegisters c.ops rs memory short) (run_of_instructions _ _ instructions) hit

end NeuralReversi.PvsParallel.FrameTableReads.Operations.Execution

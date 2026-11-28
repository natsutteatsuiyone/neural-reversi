module
public import NeuralReversi.PvsParallel.SlotFrameMir.Owner

@[expose] public section

namespace NeuralReversi.PvsParallel.SlotFrameMir

open Threading
open AtomicWord (Event Read Write SequencedBefore InReleaseSequence)
open CutoffVisibility (History HappensBefore)
open SplitRebookingMir (Coherence)

/-- After the point on u's program clock, another thread can only set the busy
    flag. A normal store by u has a positive tick offset from the chosen point. -/
inductive FlagRecord
  | externalBusy (event : Event)
  | worker (offset : Nat) (bit : Bool)

def FlagRecord.write (guard : Event) : FlagRecord → Write
  | .externalBusy e => ⟨e,1,true,false⟩
  | .worker n bit => ⟨⟨guard.worker,guard.step+n+1⟩,if bit then 1 else 0,true,false⟩

def busyFlag (e : Event) : Write := ⟨e,1,true,false⟩

/-- The earlier worker history ends at a busy publication. The second interval
    enumerates the subsequent true publications up to A's initial helper guard.
    The assignment may race with any of the remaining flag stores. -/
structure BusySpan (guard : Event) (ws : List Write) where
  before : List Write
  start : Event
  early : List Event
  tail : List FlagRecord
  contents : ws = before ++ ((start :: early).map busyFlag ++ tail.map (FlagRecord.write guard))
  anchor : Fin ws.length
  index : anchor.val = before.length

theorem BusySpan.after_zero {guard : Event} (sp : BusySpan guard ws) (i : Fin ws.length) (later : sp.anchor.val ≤ i.val)
    (zero : (ws.get i).value = 0) : SequencedBefore guard (ws.get i).event ∧ (ws.get i).release = true := by
  have got : ws[i.val]? = some (ws.get i) := List.getElem?_eq_getElem i.isLt
  have atIndex := (congrArg (fun (xs : List Write) => xs[i.val]?) sp.contents).symm.trans got
  rw [List.getElem?_append_right (sp.index ▸ later)] at atIndex
  by_cases before : i.val-sp.before.length < ((sp.start :: sp.early).map busyFlag).length
  · rw [List.getElem?_append_left before,List.getElem?_map] at atIndex
    obtain ⟨e,_,same⟩ := Option.map_eq_some_iff.mp atIndex
    rw [← same] at zero
    exact ((by decide : (1:BitVec 64) ≠ 0) zero).elim
  · rw [List.getElem?_append_right (by omega),List.getElem?_map] at atIndex
    obtain ⟨r,_,same⟩ := Option.map_eq_some_iff.mp atIndex
    rw [← same] at zero ⊢
    cases r with
    | externalBusy e => exact ((by decide : (1:BitVec 64) ≠ 0) zero).elim
    | worker n bit => exact ⟨⟨rfl,by dsimp [FlagRecord.write]; omega⟩,rfl⟩

/-- Build the A.guard→B.set edge from the second reservation's successful locked
    Acquire and the reads-from source of that target worker's searching flag. -/
theorem BusySpan.before_book {guard : Event} (sp : BusySpan guard ws) (hist : History) (r : Read ws)
    (zero : (ws.get r.source).value = 0) (acq : r.acquire = true)
    (published : HappensBefore hist (ws.get sp.anchor).event r.event) (coherent : Coherence hist ws r)
    (sync : ∀ i : Fin ws.length, r.acquire = true → InReleaseSequence ws i r.source →
      hist.otherSync (ws.get i).event r.event)
    (ordered : SequencedBefore r.event book) : HappensBefore hist guard book := by
  have bounds := coherent.writeRead sp.anchor published
  obtain ⟨before,release⟩ := sp.after_zero r.source bounds zero
  have seq : InReleaseSequence ws r.source r.source := ⟨release,Nat.le_refl _,fun _ lt le => by omega⟩
  exact .trans (.sequenced before) (.trans (.other (sync r.source acq seq)) (.sequenced ordered))

end NeuralReversi.PvsParallel.SlotFrameMir

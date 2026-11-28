module
public import NeuralReversi.PvsParallel.Interleaving.OwnerMaskGuards

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.OwnerMaskGuards.Regression

open Threading
open AtomicWord (Event Write Read SequencedBefore)
open CutoffVisibility (History HappensBefore)
open SplitSchedulerMir (Clock Guard)
open SlotFrameMir (RawMask OwnerExit)

private def owner : Clock := ⟨0,fun n => 10+10*n,by omega⟩

/-- Helpers 1 and 2 overlap. The first mask load (at 80) may still see 7
    after helper 1 resets at 76. The second load (at 100) sees their retirement. -/
def ws : List Write := [
  ⟨⟨2,2⟩,0,true,true⟩, ⟨⟨0,10⟩,0,false,false⟩, ⟨⟨0,20⟩,1,false,true⟩,
  ⟨⟨0,61⟩,3,false,true⟩, ⟨⟨0,62⟩,7,false,true⟩, ⟨⟨1,76⟩,5,true,true⟩,
  ⟨⟨0,85⟩,4,true,true⟩, ⟨⟨2,95⟩,0,true,true⟩, ⟨⟨0,200⟩,0,false,false⟩,
  ⟨⟨0,204⟩,1,false,true⟩]

local instance : NeZero ws.length := ⟨by decide⟩

private def gen : RawMask ws where
  owner := 0
  mark := 2
  actions := [.book 1 ⟨0,61⟩,.book 2 ⟨0,62⟩,.finish 1 ⟨1,76⟩,.finish 0 ⟨0,85⟩,.finish 2 ⟨2,95⟩]
  bounded := by decide
  initial := by decide
  rmw := by decide

private def input (stamp : Event) : List Operand := [
  .shutdown ⟨0,70⟩ false, .mask ⟨0,80⟩ ⟨0,62⟩ 7 true,
  .shutdown ⟨0,90⟩ false, .mask ⟨0,100⟩ stamp 0 true]

private def log (s : Fin ws.length) : List (Guard ws) :=
  [.shutdown false,.mask 4,.shutdown false,.mask s]

private def path : List (Fin 80) := [
  1,2,3,4,5,6,7,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,
  29,30,32,34,35,36,38,39,19,20,21,22,23,40,41,42,43,44,76,77,4,5,6,7,9,
  10,11,12,13,14,15,16,78,79]

/-- Every release in the sequence ending at rank 7 can synchronize with
    the owner acquire at 100. No such edge goes back to the earlier busy load. -/
private abbrev cross (a b : Event) := a.step < 100 ∧ 100 ≤ b.step ∧ b.worker = 0
private def hist : History := ⟨[],fun _ => [],[],[],fun a b =>
  a ∈ [⟨1,76⟩,⟨0,85⟩,⟨2,95⟩] ∧ b = ⟨0,100⟩⟩

private theorem order {a b} (h : HappensBefore hist a b) : SequencedBefore a b ∨ cross a b := by
  induction h with
  | sequenced sb => exact Or.inl sb
  | synchronized idx => exact Fin.elim0 idx
  | other edge =>
    obtain ⟨disjunction,rfl⟩ := edge
    simp only [List.mem_cons,List.not_mem_nil,or_false] at disjunction
    rcases disjunction with rfl | rfl | rfl <;> exact .inr (by decide)
  | trans _ _ left right =>
    rcases left with ⟨w,earlier⟩ | ⟨lo,hi,middle⟩ <;>
      rcases right with ⟨v,later⟩ | ⟨before,after_,zero⟩
    · exact .inl ⟨w.trans v,Nat.lt_trans earlier later⟩
    · exact .inr ⟨Nat.lt_trans earlier before,after_,zero⟩
    · exact .inr ⟨lo,by omega,v ▸ middle⟩
    · omega

private def rank : Guard ws → Option Nat
  | .mask i => some i.val
  | .shutdown _ => none

private theorem coherent : Coherent hist ws (input ⟨2,95⟩) := by
  intro row member i decoded
  obtain ⟨k,rfl⟩ := List.get_of_mem member
  have check : ∀ k : Fin (input ⟨2,95⟩).length, ∀ i j : Fin ws.length,
      (((input ⟨2,95⟩).get k).guard ws).bind rank = some i.val →
      ((SequencedBefore (ws.get j).event ((input ⟨2,95⟩).get k).site ∨
        cross (ws.get j).event ((input ⟨2,95⟩).get k).site) → j.val ≤ i.val) ∧
      ((SequencedBefore ((input ⟨2,95⟩).get k).site (ws.get j).event ∨
        cross ((input ⟨2,95⟩).get k).site (ws.get j).event) → i.val < j.val) := by
    unfold SequencedBefore cross; decide
  have got : (((input ⟨2,95⟩).get k).guard ws).bind rank = some i.val := by rw [decoded]; rfl
  exact ⟨fun j hb => (check k i j got).1 (order hb),fun j hb => (check k i j got).2 (order hb)⟩

private def checked : OwnerExit hist gen :=
  ownerExit hist gen owner (input ⟨2,95⟩) (log := log 7) rfl path (by decide) (by decide)
    rfl (by decide) coherent

/-- The three zero-valued RF candidates all exercise the very same MIR branches.
    The earlier return and the next clear both violate ordinary atomic coherence. -/
example : follows SplitSchedulerMir.idle ws path 0 (log 0) ∧
    follows SplitSchedulerMir.idle ws path 0 (log 7) ∧
    follows SplitSchedulerMir.idle ws path 0 (log 8) := by decide

example : ¬ Coherent hist ws (input ⟨2,2⟩) := by
  intro h
  have read := h (.mask ⟨0,100⟩ ⟨2,2⟩ 0 true) (by simp [input]) 0 rfl
  exact (by decide : ¬ (2 : Nat) ≤ 0) (read.writeRead 2 (.sequenced (by unfold SequencedBefore; decide)))

example : ¬ Coherent hist ws (input ⟨0,200⟩) := by
  intro h
  have read := h (.mask ⟨0,100⟩ ⟨0,200⟩ 0 true) (by simp [input]) 8 rfl
  exact (Nat.lt_irrefl 8) (read.readWrite 8 (.sequenced (by unfold SequencedBefore; decide)))

/-- Excluding only the next mark (rank 9) still admits its clear (rank 8). -/
example : (ws.get 8).value = (ws.get 7).value ∧ (8 : Nat) < 9 ∧
    (¬ (8 : Nat) ≤ gen.mark.val+gen.actions.length) := by decide

example : source ws ⟨0,999⟩ = none ∧ source (ws ++ [ws.get 7]) ⟨2,95⟩ = none := by decide

end NeuralReversi.PvsParallel.Interleaving.OwnerMaskGuards.Regression

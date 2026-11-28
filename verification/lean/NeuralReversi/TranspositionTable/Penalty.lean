module
public import NeuralReversi.TranspositionTable.Packing

@[expose] public section

namespace NeuralReversi.TranspositionTable.Penalty

open Packing

theorem penalized_depth (e : EntryData) (positive : 0 < e.depth.toNat) :
    (penalized e).depth.toNat + 1 = e.depth.toNat := by
  have bound : e.depth.toNat < 256 := e.depth.toFin.isLt
  dsimp [penalized]
  change ((255 + e.depth.toNat) % 256) + 1 = e.depth.toNat
  omega

theorem penalized_canCut (e : EntryData) (positive : 0 < e.depth.toNat)
    (checked : Bool) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool) :
    canCut checked (penalized e) beta depth selectivity requireEndgame =
      if depth.toNat < e.depth.toNat then
        canCut checked e beta depth selectivity requireEndgame
      else .ok false := by
  have hd := penalized_depth e positive
  by_cases below : depth.toNat < e.depth.toNat
  · have gates : Eligible (penalized e) depth selectivity requireEndgame ↔
        Eligible e depth selectivity requireEndgame := by
      dsimp [Eligible, penalized]
      constructor <;> intro h <;> refine ⟨h.1, ?_, h.2.2⟩ <;> omega
    rw [ite_eq_left below, canCut_contract, canCut_contract]
    simp only [gates]
    rfl
  · rw [ite_eq_right below]
    apply canCut_ineligible
    intro eligible
    have enough := eligible.2.1
    omega

theorem penalized_no_new_cut (e : EntryData) (positive : 0 < e.depth.toNat)
    (checked : Bool) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool)
    (cut : canCut checked (penalized e) beta depth selectivity requireEndgame = .ok true) :
    canCut checked e beta depth selectivity requireEndgame = .ok true := by
  rw [penalized_canCut e positive] at cut
  split at cut
  · exact cut
  · cases cut

theorem penalized_boundSound (e : EntryData) (truth : Int) :
    BoundSound (penalized e) truth ↔ BoundSound e truth := by rfl

theorem penalized_cut_sound (e : EntryData) (positive : 0 < e.depth.toNat)
    (checked : Bool) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool) (truth : Int)
    (sound : BoundSound e truth)
    (cut : canCut checked (penalized e) beta depth selectivity requireEndgame = .ok true) :
    (beta.raw.toInt ≤ e.score.toInt ↔ beta.raw.toInt ≤ truth) ∧
    (e.score.toInt < beta.raw.toInt ↔ truth < beta.raw.toInt) := by
  exact canCut_sound checked e beta depth selectivity requireEndgame truth sound
    (penalized_no_new_cut e positive checked beta depth selectivity requireEndgame cut)

inductive Trace : Nat → EntryData → EntryData → Prop where
  | refl (e : EntryData) : Trace 0 e e
  | step {n : Nat} {first last : EntryData} (prior : Trace n first last)
      (positive : 0 < last.depth.toNat) : Trace (n + 1) first (penalized last)

theorem trace_depth {n : Nat} {first last : EntryData} (trace : Trace n first last) :
    last.depth.toNat + n = first.depth.toNat := by
  induction trace with
  | refl => omega
  | @step n first last prior positive ih =>
    have hd := penalized_depth last positive
    omega

theorem trace_length_bound {n : Nat} {first last : EntryData} (trace : Trace n first last) :
    n ≤ first.depth.toNat ∧ n ≤ 255 := by
  have hd := trace_depth trace
  have bound : first.depth.toNat < 256 := first.depth.toFin.isLt
  omega

theorem trace_no_new_cut {n : Nat} {first last : EntryData}
    (trace : Trace n first last) (checked : Bool) (beta : Score) (depth : Depth)
    (selectivity : Selectivity) (requireEndgame : Bool)
    (cut : canCut checked last beta depth selectivity requireEndgame = .ok true) :
    canCut checked first beta depth selectivity requireEndgame = .ok true := by
  induction trace with
  | refl => exact cut
  | @step n first last prior positive ih =>
    exact ih (penalized_no_new_cut last positive checked beta depth selectivity
      requireEndgame cut)

theorem trace_boundSound {n : Nat} {first last : EntryData}
    (trace : Trace n first last) (truth : Int) :
    BoundSound last truth ↔ BoundSound first truth := by
  induction trace with
  | refl => rfl
  | step prior positive ih =>
    rw [penalized_boundSound]
    exact ih

def tryPenalize (bigEndian : Bool) (player opponent : SeqLock.Word) (expected : EntryData)
    (observation : Option SeqLock.Snapshot) (current : SeqLock.Word) : Option EntryData :=
  if expected.depth = 0#8 then none else
    observation.bind fun slot =>
      if slot.player = player ∧ slot.opponent = opponent ∧ unpack bigEndian slot.metadata = expected then
        if (SeqLock.claim current slot.sequence).1 then some (penalized expected) else none
      else none

def WriteContract : Prop :=
  ∀ bigEndian player opponent expected observation current result,
    tryPenalize bigEndian player opponent expected observation current = some result ↔
      0 < expected.depth.toNat ∧ ∃ slot, observation = some slot ∧
        slot.player = player ∧ slot.opponent = opponent ∧ unpack bigEndian slot.metadata = expected ∧
        current = slot.sequence ∧ result = penalized expected

theorem writeContract_proven : WriteContract := by
  intro bigEndian player opponent expected observation current result
  have positive : expected.depth ≠ 0#8 ↔ 0 < expected.depth.toNat := by
    constructor
    · intro nonzero
      apply Nat.pos_of_ne_zero
      exact fun zero => nonzero (BitVec.eq_of_toNat_eq zero)
    · intro pos zero
      simp [zero] at pos
  cases observation with
  | none => by_cases zero : expected.depth = 0#8 <;> simp [tryPenalize, zero]
  | some slot =>
    by_cases zero : expected.depth = 0#8
    · simp [tryPenalize, zero]
    · have pos := positive.mp zero
      by_cases matching : slot.player = player ∧ slot.opponent = opponent ∧
          unpack bigEndian slot.metadata = expected
      · simp [tryPenalize, zero, pos, matching, SeqLock.claim_success_iff,
          show penalized expected = result ↔ result = penalized expected from eq_comm]
      · simp [tryPenalize, zero, matching]
        intro _ hp ho hd
        exact (matching ⟨hp, ho, hd⟩).elim

def CutContract : Prop :=
  ∀ bigEndian player opponent expected observation current result,
    tryPenalize bigEndian player opponent expected observation current = some result →
      result.depth.toNat + 1 = expected.depth.toNat ∧
      (∀ truth, BoundSound result truth ↔ BoundSound expected truth) ∧
      ∀ checked beta depth selectivity endgame,
        canCut checked result beta depth selectivity endgame = .ok true →
          canCut checked expected beta depth selectivity endgame = .ok true

theorem cutContract_proven : CutContract := by
  intro bigEndian player opponent expected observation current result written
  obtain ⟨positive, slot, observed, playerMatch, opponentMatch, dataMatch, stillCurrent, rfl⟩ :=
    (writeContract_proven bigEndian player opponent expected observation current result).mp written
  exact ⟨penalized_depth expected positive, penalized_boundSound expected,
    penalized_no_new_cut expected positive⟩

private def fixture (score : Int) (depth : Nat) : EntryData :=
  ⟨BitVec.ofInt 16 score, 64, 3, BitVec.ofNat 8 depth, 5, 0, 1⟩

example : (penalized (fixture 0 1)).depth = 0 := by rfl
example : (penalized (fixture 0 255)).depth = 254 := by rfl
example : canCut true (penalized (fixture 0 12)) ⟨0⟩ 12 ⟨5, by decide⟩ true =
    .ok false := by rfl
example : canCut true (penalized (fixture 0 12)) ⟨0⟩ 11 ⟨5, by decide⟩ true =
    .ok true := by rfl

example : canCut true (penalized (fixture 0 1)) ⟨0⟩ 0 ⟨5, by decide⟩ true =
    .ok true := by rfl
example : canCut true (penalized (fixture 0 255)) ⟨0⟩ 255 ⟨5, by decide⟩ true =
    .ok false := by rfl
example : canCut true (penalized { fixture 0 12 with selectivity := 4 })
    ⟨0⟩ 10 ⟨5, by decide⟩ true = .ok false := by rfl
example : canCut true (penalized { fixture 0 12 with flags := 0 })
    ⟨0⟩ 10 ⟨5, by decide⟩ true = .ok false := by rfl

example : canCut true (fixture (-32768) 12) ⟨0⟩ 12 ⟨5, by decide⟩ true =
    .error scoreError ∧
    canCut true (penalized (fixture (-32768) 12)) ⟨0⟩ 12 ⟨5, by decide⟩ true =
    .ok false := by constructor <;> rfl

example : canCut false (fixture 0 0) ⟨0⟩ 1 ⟨5, by decide⟩ true = .ok false ∧
    canCut false (penalized (fixture 0 0)) ⟨0⟩ 1 ⟨5, by decide⟩ true = .ok true := by
  constructor <;> rfl

example : tryPenalize false 7 8 (fixture 0 0)
    (some ⟨7, 8, 2, pack false (fixture 0 0)⟩) 2 = none := by decide

end NeuralReversi.TranspositionTable.Penalty

import NeuralReversi.TranspositionTable.Packing

namespace NeuralReversi.PvsEnhancedCutoff

open TranspositionTable

structure Query where
  depth : Depth
  selectivity : Selectivity
  endgame : Bool
  alpha : Score

abbrev Probe := Fin 64 × Option EntryData
abbrev Hit := Fin 64 × Int

def candidate (q : Query) (p : Probe) : Option Hit := do
  let e ← p.2
  let score := (-(e.score.signExtend 32)).toInt
  if (q.endgame = false ∨ isEndgame e = true) ∧ (q.depth - 1).toNat ≤ e.depth.toNat ∧
      q.selectivity.val ≤ selectivityRank e.selectivity ∧
      (e.bound = Packing.boundByte .exact ∨ e.bound = Packing.boundByte .upper) ∧
      q.alpha.raw.toInt < score then
    some (p.1, score)
  else none

def scan (q : Query) (probes : List Probe) : Option Hit :=
  probes.findSome? (candidate q)

def Witness (q : Query) (p : Probe) (hit : Hit) : Prop :=
  ∃ e, p.2 = some e ∧ Eligible e (q.depth - 1) q.selectivity q.endgame ∧
    (e.bound = Packing.boundByte .exact ∨ e.bound = Packing.boundByte .upper) ∧
    q.alpha.raw.toInt < -e.score.toInt ∧ (p.1, -e.score.toInt) = hit

private theorem negated_score (s : BitVec 16) :
    (-(s.signExtend 32)).toInt = -s.toInt := by
  have widened : (s.signExtend 32).toInt = s.toInt := BitVec.toInt_signExtend_of_le (by decide)
  have fits : s.signExtend 32 ≠ BitVec.intMin 32 := by
    intro h
    rw [h] at widened
    have lo := BitVec.le_toInt s
    change -2147483648 = s.toInt at widened
    change -32768 ≤ s.toInt at lo
    omega
  rw [BitVec.toInt_neg_of_ne_intMin fits, widened]

private theorem candidate_spec (q : Query) (p : Probe) (hit : Hit) :
    candidate q p = some hit ↔ Witness q p hit := by
  rcases p with ⟨sq, _ | e⟩
  · simp [candidate, Witness]
  · have flag : isEndgame e = e.flags.getLsbD 0 := (byte_masks e.flags).1
    simp [candidate, Witness, Eligible, flag, selectivityRank_eq_min, negated_score, and_assoc]

private theorem no_candidate (q : Query) (p : Probe) :
    candidate q p = none ↔ ∀ hit, ¬ Witness q p hit := by
  simp only [← candidate_spec]
  cases candidate q p <;> simp

def Contract : Prop :=
  ∀ (q : Query) (probes : List Probe),
    (scan q probes = none ↔ ∀ p ∈ probes, ∀ hit, ¬ Witness q p hit) ∧
    (∀ hit, scan q probes = some hit ↔ ∃ before p after, probes = before ++ p :: after ∧
      Witness q p hit ∧ ∀ prev ∈ before, ∀ other, ¬ Witness q prev other) ∧
    ∀ (childValue : Fin 64 → Int) (parentValue : Int),
      (∀ p ∈ probes, ∀ e, p.2 = some e → BoundSound e (childValue p.1)) →
      (∀ p ∈ probes, -(childValue p.1) ≤ parentValue) →
      ∀ hit, scan q probes = some hit →
        q.alpha.raw.toInt < hit.2 ∧ hit.2 ≤ -(childValue hit.1) ∧ hit.2 ≤ parentValue

theorem contract_proven : Contract := by
  intro q probes
  refine ⟨by simp only [scan, List.findSome?_eq_none_iff, no_candidate], ?_, ?_⟩
  · intro hit
    simp only [scan, List.findSome?_eq_some_iff, candidate_spec, no_candidate]
  · intro childValue parentValue sound dominates hit cut
    obtain ⟨p, member, selected⟩ := List.exists_of_findSome?_eq_some cut
    obtain ⟨e, lookup, _, tag, high, rfl⟩ := (candidate_spec q p hit).mp selected
    have upper := (sound p member e lookup).2 (by rcases tag with h | h <;> rw [h] <;> decide)
    have lower : -e.score.toInt ≤ -(childValue p.1) := by omega
    exact ⟨high, lower, Int.le_trans lower (dominates p member)⟩

end NeuralReversi.PvsEnhancedCutoff

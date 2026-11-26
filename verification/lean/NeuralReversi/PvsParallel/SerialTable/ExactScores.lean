import NeuralReversi.PvsParallel.SerialTable.Finite

namespace NeuralReversi.PvsParallel.SerialTable

open TranspositionTable
open PvsMoveLoop (scoreMax)

theorem exactScores_empty : ExactScores [] := fun _ _ mem => by cases mem

private theorem proposal_bound {b oldKey data old e} (made : Save.proposal b oldKey data old = some e) : e.bound = data.bound := by
  rw [(Save.contract_proven ..).1] at made
  split at made
  · contradiction
  · cases Option.some.inj made
    unfold Save.payload
    split <;> rfl

theorem Table.store_exactScores (t : Table l) (support : t.image.Supported t.history) (prior : ExactScores t.history)
    (slot : l.Index) (b : PvsBoardTree.Position) (sc : Score) (bound : Bound) (d : Depth)
    (move : Option Bitboard.Square) (level : Selectivity) (ending pv : Bool)
    (bounded : bound = .exact → -scoreMax ≤ sc.raw.toInt ∧ sc.raw.toInt ≤ scoreMax) :
    ExactScores (t.store slot b sc bound d move level ending pv).history := by
  rw [Table.store,WriterMir.store_input]
  by_cases fits : sc.raw.toInt.natAbs < 32767 <;> simp only [fits,ite_true,ite_false]
  · rw [t.save_eq _ _ _ support]
    split
    next e made =>
      intro board entry mem exactBound
      rcases List.mem_append.mp mem with old | fresh
      · exact prior board entry old exactBound
      · obtain ⟨rfl,rfl⟩ := Prod.mk.inj (List.mem_singleton.mp fresh)
        have byte := (proposal_bound made).symm.trans exactBound
        change Packing.boundByte bound = Packing.boundByte .exact at byte
        have tag : bound = .exact := by cases bound <;> first | rfl | contradiction
        have range := bounded tag
        rw [proposal_score made]
        change -scoreMax ≤ (sc.raw.setWidth 16).toInt ∧ (sc.raw.setWidth 16).toInt ≤ scoreMax
        rw [StoredBound.narrow_score sc (by unfold scoreMax at range; omega) (by unfold scoreMax at range; omega)]
        exact range
    · exact prior
  · exact prior

theorem Memory.apply_exactScores (m : Memory l) (prior : ExactScores m.tt.history) (slot : l.Index)
    (level : Selectivity) (proposal : Option Writes.Proposal)
    (bounded : ∀ w, proposal = some w → w.bound = .exact → -scoreMax ≤ w.score ∧ w.score ≤ scoreMax) :
    ExactScores (m.apply slot proposal level).tt.history := by
  cases proposal with
  | none => exact prior
  | some w =>
    apply m.tt.store_exactScores m.supported prior
    intro bound
    have range := bounded w rfl bound
    have signed : (Writes.word w.score).raw.toInt = w.score := BitVec.toInt_ofInt_eq_self (by decide)
      (by unfold scoreMax at range; omega) (by unfold scoreMax at range; omega)
    rwa [signed]

end NeuralReversi.PvsParallel.SerialTable

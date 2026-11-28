module
public import NeuralReversi.Game.History.VecModel
public import NeuralReversi.Game.History.ReferenceProperties

@[expose] public section

namespace NeuralReversi.Game.History.VecView

open NeuralReversi.Game.History.Reference

theorem decrement_exact (entryBytes : Nat) (v : HistoryVec) (valid : Valid entryBytes v)
    (nonzero : v.lengthWord ≠ 0) :
    (v.lengthWord - 1).toNat = v.lengthWord.toNat - 1 ∧
    (v.lengthWord - 1).toNat < v.lengthWord.toNat ∧
    (v.lengthWord - 1).toNat < v.allocation.capacityWord.toNat := by
  have positive : 0 < v.lengthWord.toNat := by
    by_cases zero : v.lengthWord.toNat = 0
    · apply False.elim
      apply nonzero
      apply BitVec.eq_of_toNat_eq
      exact zero
    · omega
  have oneLe : (1 : Word) ≤ v.lengthWord := by
    change 1 ≤ v.lengthWord.toNat
    omega
  have eq := BitVec.toNat_sub_of_le oneLe
  change (v.lengthWord - 1).toNat = v.lengthWord.toNat - 1 at eq
  have bound := valid.2.1
  exact ⟨eq,by omega,by omega⟩

theorem pop_valid_frame (entryBytes : Nat) (v : HistoryVec) (valid : Valid entryBytes v) :
    Valid entryBytes (pop entryBytes v valid).2 ∧
    (pop entryBytes v valid).2.allocation = v.allocation := by
  by_cases empty : v.lengthWord = 0
  · rw [pop,dite_eq_left empty]
    exact ⟨valid,rfl⟩
  · have exactLength := (decrement_exact entryBytes v valid empty).1
    have oldLength := valid.1
    have oldBound := valid.2.1
    rw [pop,dite_eq_right empty]
    dsimp only
    refine ⟨⟨?_,?_,valid.2.2⟩,rfl⟩
    · simpa only [List.length_dropLast,oldLength] using exactLength
    · change (v.lengthWord - 1).toNat ≤ v.allocation.capacityWord.toNat
      omega

def listPop (entries : List Entry) : Option Entry × List Entry :=
  match popLast entries with
  | none => (none,entries)
  | some (entry,earlier) => (some entry,earlier)

theorem pop_value_refinement (entryBytes : Nat) (v : HistoryVec) (valid : Valid entryBytes v) :
    ((pop entryBytes v valid).1,(pop entryBytes v valid).2.entries) = listPop v.entries := by
  rcases suffix_cases v.entries with empty | ⟨earlier,last,history⟩
  · have zero : v.lengthWord = 0 := by
      apply BitVec.eq_of_toNat_eq
      simpa [empty] using valid.1
    simp [pop,zero,listPop,empty,popLast]
  · have nonzero : v.lengthWord ≠ 0 := by
      intro zero
      have len := valid.1
      simp [zero,history] at len
    rw [pop,dite_eq_right nonzero]
    simp [listPop,history,popLast_append]

theorem confirm_valid_frame (entryBytes : Nat) (v : HistoryVec) (valid : Valid entryBytes v)
    (side : Disc) :
    Valid entryBytes (confirm entryBytes v valid side).2 ∧
    (confirm entryBytes v valid side).2.lengthWord = v.lengthWord ∧
    (confirm entryBytes v valid side).2.allocation = v.allocation := by
  by_cases empty : v.lengthWord = 0
  · rw [confirm,dite_eq_left empty]
    exact ⟨valid,rfl,rfl⟩
  · rw [confirm,dite_eq_right empty]
    dsimp only
    split
    · split
      · refine ⟨⟨?_,valid.2⟩,rfl,rfl⟩
        simpa only [List.length_set] using valid.1
      · exact ⟨valid,rfl,rfl⟩
    · exact ⟨valid,rfl,rfl⟩

theorem confirm_value_refinement (entryBytes : Nat) (v : HistoryVec) (valid : Valid entryBytes v)
    (board : Board) (currentSide requestedSide : Disc) :
    ((confirm entryBytes v valid requestedSide).1,
      project board currentSide (confirm entryBytes v valid requestedSide).2) =
      NeuralReversi.Game.History.Reference.confirm (project board currentSide v) requestedSide := by
  rcases suffix_cases v.entries with empty | ⟨earlier,last,history⟩
  · have zero : v.lengthWord = 0 := by
      apply BitVec.eq_of_toNat_eq
      simpa [empty] using valid.1
    rw [confirm,dite_eq_left zero]
    simp [project,NeuralReversi.Game.History.Reference.confirm,empty,popLast]
  · have nonzero : v.lengthWord ≠ 0 := by
      intro zero
      have len := valid.1
      simp [zero,history] at len
    have index : (v.lengthWord - 1).toNat = earlier.length := by
      rw [(decrement_exact entryBytes v valid nonzero).1,valid.1,history]
      simp
    change (v.lengthWord - 1#64).toNat = earlier.length at index
    rw [confirm,dite_eq_right nonzero]
    by_cases auto : last.action = .automaticPass
    · by_cases requested : last.side = requestedSide
      · simp [-BitVec.toNat_sub,project,NeuralReversi.Game.History.Reference.confirm,history,popLast_append,index,auto,requested]
      · simp [project,NeuralReversi.Game.History.Reference.confirm,history,popLast_append,auto,requested]
    · simp [project,NeuralReversi.Game.History.Reference.confirm,history,popLast_append,auto]

abbrev CheckedVec (entryBytes : Nat) := {v : HistoryVec // Valid entryBytes v}

def typedPop (entryBytes : Nat) (v : CheckedVec entryBytes) : Option Entry × CheckedVec entryBytes :=
  let result := pop entryBytes v.val v.property
  (result.1,⟨result.2,(pop_valid_frame entryBytes v.val v.property).1⟩)

theorem typedPop_frame (entryBytes : Nat) (v : CheckedVec entryBytes) :
    (typedPop entryBytes v).2.val.allocation = v.val.allocation :=
  (pop_valid_frame entryBytes v.val v.property).2

theorem typedPop_projection (entryBytes : Nat) (v : CheckedVec entryBytes) :
    ((typedPop entryBytes v).1,(typedPop entryBytes v).2.val.entries) = listPop v.val.entries :=
  pop_value_refinement entryBytes v.val v.property

theorem last_address_no_wrap (v : HistoryVec) (valid : Valid 24 v)
    (nonzero : v.lengthWord ≠ 0) :
    let index := v.lengthWord - 1
    (v.allocation.address + index * BitVec.ofNat 64 24).toNat =
      v.allocation.address.toNat + index.toNat * 24 := by
  dsimp only
  have smaller := (decrement_exact 24 v valid nonzero).2.2
  have bytes := valid.2.2.2.2.2.2.1
  have address := valid.2.2.2.2.2.2.2
  have delta : (v.lengthWord - 1).toNat * 24 < 2^64 := by omega
  have product : ((v.lengthWord - 1) * BitVec.ofNat 64 24).toNat =
      (v.lengthWord - 1).toNat * 24 := by
    apply BitVec.toNat_mul_of_lt
    exact delta
  rw [BitVec.toNat_add_of_lt (by rw [product]; omega),product]

end NeuralReversi.Game.History.VecView

import NeuralReversi.TranspositionTable.Packing
import NeuralReversi.Bitboard.MovesSpec

namespace NeuralReversi.TranspositionTable.Save

open Packing

abbrev Key := BitVec 64 × BitVec 64

def proposal (key storedKey : Key) (data stored : EntryData) : Option EntryData :=
  if data.bound = boundByte .exact then some data
  else if stored.bound = 0 then some data
  else
    let same := sameBoardBits key.1 key.2 storedKey.1 storedKey.2
    let depth := data.depth.setWidth 32
    if same = false ∨ stored.depth.toNat ≤ (depth + 2).toNat ∨
        (selectivityRank stored.selectivity < data.selectivity.toNat ∧
          stored.depth.toNat ≤ (depth + 4).toNat) ∨ data.generation ≠ stored.generation then
      if data.bestMove ≠ noMove ∨ same = false then some data
      else some (withBestMoveOf data stored)
    else none

def Reject (key storedKey : Key) (data stored : EntryData) : Prop :=
  data.bound ≠ boundByte .exact ∧ stored.bound ≠ 0 ∧ key = storedKey ∧
    data.generation = stored.generation ∧
    (if min stored.selectivity.toNat 5 < data.selectivity.toNat then 4 else 2) <
      stored.depth.toNat - data.depth.toNat

instance (key storedKey : Key) (data stored : EntryData) : Decidable (Reject key storedKey data stored) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

def payload (key storedKey : Key) (data stored : EntryData) : EntryData :=
  if data.bound ≠ boundByte .exact ∧ stored.bound ≠ 0 ∧ key = storedKey ∧
      data.bestMove = noMove then { data with bestMove := stored.bestMove }
  else data

private theorem widened_add (d : BitVec 8) (k : BitVec 32) (h : k.toNat ≤ 4) :
    (d.setWidth 32 + k).toNat = d.toNat + k.toNat := by
  have hd : d.toNat < 256 := d.toFin.isLt
  simp only [BitVec.toNat_add, depth_widen]
  apply Nat.mod_eq_of_lt
  omega

private theorem decision (key storedKey : Key) (data stored : EntryData) :
    proposal key storedKey data stored =
      if Reject key storedKey data stored then none else some (payload key storedKey data stored) := by
  have same : sameBoardBits key.1 key.2 storedKey.1 storedKey.2 = true ↔ key = storedKey := by
    rw [sameBoardBits_exact, Prod.ext_iff]
  have different : sameBoardBits key.1 key.2 storedKey.1 storedKey.2 = false ↔ key ≠ storedKey := by
    simp [← same]
  simp only [proposal, Reject, payload, different, widened_add _ 2 (by decide),
    widened_add _ 4 (by decide), selectivityRank_eq_min]
  by_cases exactBound : data.bound = boundByte .exact
  · simp [exactBound]
  by_cases empty : stored.bound = 0
  · simp [empty]
  by_cases sameKey : key = storedKey
  · simp only [exactBound, empty, sameKey, true_and, ↓reduceIte, withBestMoveOf]
    split <;> rename_i gate
    · split <;> split <;> simp_all <;> intro gen <;> simp [gen] at gate <;> omega
    · split <;> simp_all <;> omega
  · simp [exactBound, sameKey]

def legal (key : Key) (move : BitVec 8) : Prop :=
  Bitboard.Moves.LegalMove key.1 key.2 move.toNat

def Contract : Prop :=
  ∀ key storedKey data stored,
    (proposal key storedKey data stored =
      if Reject key storedKey data stored then none else some (payload key storedKey data stored)) ∧
    ((data.bestMove ≠ noMove → legal key data.bestMove) →
      (stored.bound ≠ 0 → stored.bestMove ≠ noMove → legal storedKey stored.bestMove) →
      ∀ written, proposal key storedKey data stored = some written →
        written.bestMove ≠ noMove → legal key written.bestMove)

theorem contract_proven : Contract := by
  intro key storedKey data stored
  refine ⟨decision .., ?_⟩
  intro incoming old written result present
  rw [decision] at result
  split at result
  · contradiction
  · cases Option.some.inj result
    unfold payload at *
    split at * <;> simp_all

end NeuralReversi.TranspositionTable.Save

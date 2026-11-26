import NeuralReversi.Stability.Cutoff

namespace NeuralReversi.Stability

open Semantics

def EdgeMoveContract : Prop := ∀ p o sq, Edge.MoveCorrect p o sq

theorem edgeMoveContract_proven : EdgeMoveContract := Edge.applyEdgeMove_correct

private theorem shiftOr_eq_mul_add (hi lo : Byte) :
    hi.toNat * 256 + lo.toNat = (((hi.setWidth 16) <<< 8) ||| lo.setWidth 16).toNat := by
  have hhi := hi.isLt
  have hlo := lo.isLt
  simp only [BitVec.toNat_or, BitVec.toNat_shiftLeft, BitVec.toNat_setWidth, Nat.shiftLeft_eq]
  change hi.toNat * 256 + lo.toNat =
    ((hi.toNat % 65536 * 256) % 65536) ||| (lo.toNat % 65536)
  rw [Nat.mod_eq_of_lt (show hi.toNat < 65536 by omega),
    Nat.mod_eq_of_lt (show hi.toNat * 256 < 65536 by omega),
    Nat.mod_eq_of_lt (show lo.toNat < 65536 by omega)]
  simpa only [Nat.shiftLeft_eq] using
    (Nat.shiftLeft_add_eq_or_of_lt (i := 8) hlo hi.toNat)

def EdgeIndexContract : Prop :=
  (∀ sq : Fin 8, (Edge.bit8 sq.val).toNat = 2 ^ sq.val) ∧
  (∀ p o : Byte,
    Edge.edgeIndex p o = ((((p ||| o).setWidth 16) <<< 8) ||| p.setWidth 16).toNat ∧
    Edge.edgeIndex p o < Edge.initEdgeStability.size) ∧
  (∀ p occ : Byte,
    Edge.edgeLookupIndex p occ = (((occ.setWidth 16) <<< 8) ||| p.setWidth 16).toNat ∧
    Edge.edgeLookupIndex p occ < Edge.initEdgeStability.size)

theorem edgeIndexContract_proven : EdgeIndexContract := by
  refine ⟨?_, fun p o => ⟨shiftOr_eq_mul_add (p ||| o) p, Edge.initEdgeStability_index_safe p o⟩,
    fun p occ => ⟨shiftOr_eq_mul_add occ p, Edge.initEdgeStability_lookup_safe p occ⟩⟩
  decide +kernel

def EdgeTableContract : Prop :=
  Edge.initEdgeStability.size = 65536 ∧
  (∀ p o : Byte, p &&& o = 0 →
    Edge.edgeStability Edge.initEdgeStability p o = Edge.recursiveStable p o) ∧
  ∀ p occ : Byte,
    (p &&& ~~~occ ≠ 0 → Edge.edgeStabilityOcc Edge.initEdgeStability p occ = 0) ∧
    Edge.edgeStabilityOcc Edge.initEdgeStability p occ =
      Edge.checkedLookup Edge.initEdgeStability p occ (Edge.initEdgeStability_lookup_safe p occ)

theorem edgeTableContract_proven : EdgeTableContract :=
  ⟨Edge.initEdgeStability_size, Edge.initEdgeStability_eq_recursive, fun p occ =>
    ⟨Edge.initEdgeStability_invalid_zero p occ, Edge.initEdgeStability_lookup p occ⟩⟩

def EdgeChildIntersectionContract : Prop :=
  ∀ p o table, Edge.Subset (Edge.findEdgeStable p o table) p ∧
    ∀ sq : Fin 8, Edge.emptyAt p o sq = true →
      let a := Edge.firstChild p o sq
      let b := Edge.secondChild p o sq
      Edge.Subset (Edge.findEdgeStable p o table) (Edge.edgeStability table a.1 a.2) ∧
      Edge.Subset (Edge.findEdgeStable p o table) (Edge.edgeStability table b.1 b.2)

theorem edgeChildIntersectionContract_proven : EdgeChildIntersectionContract :=
  Edge.findEdgeStable_contract

def EdgePackingContract : Prop :=
  (∀ board, packA1A8Bmi2Word board = (packA1A8 board).setWidth 64 ∧
    packH1H8Bmi2Word board = (packH1H8 board).setWidth 64 ∧
    packA1A8 board = columnBits board 0 ∧ packH1H8 board = columnBits board 7) ∧
  (∀ x, unpackA2A7Bmi2 x = unpackA2A7 x ∧ unpackH2H7Bmi2 x = unpackH2H7 x) ∧
  (∀ x (r c : Fin 8), (unpackA2A7 x).getLsbD (8 * r.val + c.val) =
    (decide (c.val = 0 ∧ 0 < r.val ∧ r.val < 7) && x.getLsbD r.val) ∧
    (unpackH2H7 x).getLsbD (8 * r.val + c.val) =
    (decide (c.val = 7 ∧ 0 < r.val ∧ r.val < 7) && x.getLsbD r.val))

theorem edgePackingContract_proven : EdgePackingContract :=
  ⟨fun board => ⟨packA1A8Bmi2Word_eq board, packH1H8Bmi2Word_eq board,
      packA_eq_column board, packH_eq_column board⟩,
    fun x => ⟨unpackA2A7Bmi2_eq_unpackA2A7 x, unpackH2H7Bmi2_eq_unpackH2H7 x⟩,
    fun x r c => ⟨unpackA_bit x r c, unpackH_bit x r c⟩⟩

def FullLineContract : Prop :=
  ∀ disc axis sq, (field (getFullLines disc) axis).getLsbD sq.val = true ↔ FullAt disc axis sq

theorem fullLineContract_proven : FullLineContract := fullLines_correct

def ContactContract : Prop :=
  ∀ central previous full,
    let seed := previous ||| (full.intersection &&& central)
    let result := getStableByContact central previous full
    (seed = 0 → result = 0) ∧
    contactStep central result full = result ∧
    ∀ sq, seed.getLsbD sq = true → result.getLsbD sq = true

theorem contactContract_proven : ContactContract := by
  intro central previous full
  let seed := previous ||| (full.intersection &&& central)
  change (seed = 0 → getStableByContact central previous full = 0) ∧
    contactStep central (getStableByContact central previous full) full =
      getStableByContact central previous full ∧
    ∀ sq, seed.getLsbD sq = true → (getStableByContact central previous full).getLsbD sq = true
  by_cases zero : seed = 0
  · have hres : getStableByContact central previous full = 0 := by
      unfold getStableByContact
      exact (ite_eq_left zero).trans zero
    rw [hres]
    refine ⟨fun _ => rfl, ?_, fun sq h => by simp [zero] at h⟩
    simpa [contactStep, FullLines.intersection] using (BitVec.or_eq_zero_iff.mp zero).2
  · have hres : getStableByContact central previous full = contactClosure central seed full := by
      unfold getStableByContact
      exact ite_eq_right zero
    rw [hres]
    exact ⟨fun h => (zero h).elim, contactClosure_fixed central seed full, fun sq member =>
      contactClosure_preserves central seed full (fun x => x.getLsbD sq = true)
        (fun x present => contactStep_contains central x full sq present) member⟩

def StableDiscContract : Prop :=
  ∀ s t c p o, Represents s c p o → Reach s t →
    (∀ sq : Fin 64, (getStableEdge Edge.initEdgeStability p o).getLsbD sq.val = true →
      t.board (boardSquare sq) = some c) ∧
    (∀ sq : Fin 64, (getStableDiscs p o).getLsbD sq.val = true →
      t.board (boardSquare sq) = some c)

theorem stableDiscContract_proven : StableDiscContract := by
  intro s t c p o repr future
  exact ⟨fun sq marked => table_edge_sound _ initialized_table_agrees repr sq marked t future,
    fun sq marked => returned_disc_never_flips repr future sq marked⟩

end NeuralReversi.Stability

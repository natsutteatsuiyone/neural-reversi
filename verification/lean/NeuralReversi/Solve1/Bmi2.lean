import NeuralReversi.Solve1.Bmi2Proofs.Part7

namespace NeuralReversi.Solve1.Bmi2
set_option maxRecDepth 10000
set_option maxHeartbeats 0
theorem refs_correct (p : Word) (square : Fin 64) : refs p square.val = reference p square.val := by
  rcases square with ⟨x,hx⟩
  match x with
  | 0 => exact refs_0 p
  | 1 => exact refs_1 p
  | 2 => exact refs_2 p
  | 3 => exact refs_3 p
  | 4 => exact refs_4 p
  | 5 => exact refs_5 p
  | 6 => exact refs_6 p
  | 7 => exact refs_7 p
  | 8 => exact refs_8 p
  | 9 => exact refs_9 p
  | 10 => exact refs_10 p
  | 11 => exact refs_11 p
  | 12 => exact refs_12 p
  | 13 => exact refs_13 p
  | 14 => exact refs_14 p
  | 15 => exact refs_15 p
  | 16 => exact refs_16 p
  | 17 => exact refs_17 p
  | 18 => exact refs_18 p
  | 19 => exact refs_19 p
  | 20 => exact refs_20 p
  | 21 => exact refs_21 p
  | 22 => exact refs_22 p
  | 23 => exact refs_23 p
  | 24 => exact refs_24 p
  | 25 => exact refs_25 p
  | 26 => exact refs_26 p
  | 27 => exact refs_27 p
  | 28 => exact refs_28 p
  | 29 => exact refs_29 p
  | 30 => exact refs_30 p
  | 31 => exact refs_31 p
  | 32 => exact refs_32 p
  | 33 => exact refs_33 p
  | 34 => exact refs_34 p
  | 35 => exact refs_35 p
  | 36 => exact refs_36 p
  | 37 => exact refs_37 p
  | 38 => exact refs_38 p
  | 39 => exact refs_39 p
  | 40 => exact refs_40 p
  | 41 => exact refs_41 p
  | 42 => exact refs_42 p
  | 43 => exact refs_43 p
  | 44 => exact refs_44 p
  | 45 => exact refs_45 p
  | 46 => exact refs_46 p
  | 47 => exact refs_47 p
  | 48 => exact refs_48 p
  | 49 => exact refs_49 p
  | 50 => exact refs_50 p
  | 51 => exact refs_51 p
  | 52 => exact refs_52 p
  | 53 => exact refs_53 p
  | 54 => exact refs_54 p
  | 55 => exact refs_55 p
  | 56 => exact refs_56 p
  | 57 => exact refs_57 p
  | 58 => exact refs_58 p
  | 59 => exact refs_59 p
  | 60 => exact refs_60 p
  | 61 => exact refs_61 p
  | 62 => exact refs_62 p
  | 63 => exact refs_63 p
  | n + 64 => omega

theorem refsOpp_correct (p : Word) (square : Fin 64) : refsOpp p square.val = reference (~~~p) square.val := by
  rcases square with ⟨x,hx⟩
  match x with
  | 0 => exact refsOpp_0 p
  | 1 => exact refsOpp_1 p
  | 2 => exact refsOpp_2 p
  | 3 => exact refsOpp_3 p
  | 4 => exact refsOpp_4 p
  | 5 => exact refsOpp_5 p
  | 6 => exact refsOpp_6 p
  | 7 => exact refsOpp_7 p
  | 8 => exact refsOpp_8 p
  | 9 => exact refsOpp_9 p
  | 10 => exact refsOpp_10 p
  | 11 => exact refsOpp_11 p
  | 12 => exact refsOpp_12 p
  | 13 => exact refsOpp_13 p
  | 14 => exact refsOpp_14 p
  | 15 => exact refsOpp_15 p
  | 16 => exact refsOpp_16 p
  | 17 => exact refsOpp_17 p
  | 18 => exact refsOpp_18 p
  | 19 => exact refsOpp_19 p
  | 20 => exact refsOpp_20 p
  | 21 => exact refsOpp_21 p
  | 22 => exact refsOpp_22 p
  | 23 => exact refsOpp_23 p
  | 24 => exact refsOpp_24 p
  | 25 => exact refsOpp_25 p
  | 26 => exact refsOpp_26 p
  | 27 => exact refsOpp_27 p
  | 28 => exact refsOpp_28 p
  | 29 => exact refsOpp_29 p
  | 30 => exact refsOpp_30 p
  | 31 => exact refsOpp_31 p
  | 32 => exact refsOpp_32 p
  | 33 => exact refsOpp_33 p
  | 34 => exact refsOpp_34 p
  | 35 => exact refsOpp_35 p
  | 36 => exact refsOpp_36 p
  | 37 => exact refsOpp_37 p
  | 38 => exact refsOpp_38 p
  | 39 => exact refsOpp_39 p
  | 40 => exact refsOpp_40 p
  | 41 => exact refsOpp_41 p
  | 42 => exact refsOpp_42 p
  | 43 => exact refsOpp_43 p
  | 44 => exact refsOpp_44 p
  | 45 => exact refsOpp_45 p
  | 46 => exact refsOpp_46 p
  | 47 => exact refsOpp_47 p
  | 48 => exact refsOpp_48 p
  | 49 => exact refsOpp_49 p
  | 50 => exact refsOpp_50 p
  | 51 => exact refsOpp_51 p
  | 52 => exact refsOpp_52 p
  | 53 => exact refsOpp_53 p
  | 54 => exact refsOpp_54 p
  | 55 => exact refsOpp_55 p
  | 56 => exact refsOpp_56 p
  | 57 => exact refsOpp_57 p
  | 58 => exact refsOpp_58 p
  | 59 => exact refsOpp_59 p
  | 60 => exact refsOpp_60 p
  | 61 => exact refsOpp_61 p
  | 62 => exact refsOpp_62 p
  | 63 => exact refsOpp_63 p
  | n + 64 => omega


theorem count_refs (p : Word) (square : Nat) : count p square = BitVec.ofNat 8 (2 * refs p square) := by
  simp only [count, lookup, LineTable.countFlip_eq_ofNat, refs, Nat.mul_add, BitVec.ofNat_add]
  simp only [BitVec.add_assoc]

theorem countOpp_refs (p : Word) (square : Nat) : countOpp p square = BitVec.ofNat 8 (2 * refsOpp p square) := by
  simp only [countOpp, lookup, LineTable.countFlip_eq_ofNat, refsOpp, Nat.mul_add, BitVec.ofNat_add]
  simp only [BitVec.add_assoc]

theorem count_correct (p : Word) (square : Fin 64) :
    count p square.val = BitVec.ofNat 8 (2 * reference p square.val) := by
  rw [count_refs, refs_correct]
theorem countOpp_correct (p : Word) (square : Fin 64) :
    countOpp p square.val = BitVec.ofNat 8 (2 * reference (~~~p) square.val) := by
  rw [countOpp_refs, refsOpp_correct]


theorem refs_bound (p : Word) (square : Nat) : refs p square ≤ 24 := by
  unfold refs
  exact Nat.add_le_add
    (Nat.add_le_add (Nat.add_le_add (LineTable.reference_bound _ _) (LineTable.reference_bound _ _))
      (LineTable.reference_bound _ _)) (LineTable.reference_bound _ _)

theorem reference_bound (p : Word) (square : Fin 64) : reference p square.val ≤ 24 := by
  rw [← refs_correct]
  exact refs_bound p square.val

theorem count_toNat (p : Word) (square : Fin 64) :
    (count p square.val).toNat = 2 * reference p square.val := by
  rw [count_correct, BitVec.toNat_ofNat]
  have h := reference_bound p square
  apply Nat.mod_eq_of_lt
  omega

theorem countOpp_toNat (p : Word) (square : Fin 64) :
    (countOpp p square.val).toNat = 2 * reference (~~~p) square.val := by
  rw [countOpp_correct, BitVec.toNat_ofNat]
  have h := reference_bound (~~~p) square
  apply Nat.mod_eq_of_lt
  omega

theorem metadata_positions (square : Fin 64) :
    (unionParams square.val).rowPos < 8 ∧ (unionParams square.val).filePos < 8 ∧
    (unionParams square.val).d0Pos < 8 ∧ (unionParams square.val).d1Pos < 8 := by
  have all : ∀ s : Fin 64,
    (unionParams s.val).rowPos < 8 ∧ (unionParams s.val).filePos < 8 ∧
    (unionParams s.val).d0Pos < 8 ∧ (unionParams s.val).d1Pos < 8 := by decide
  exact all square


def lookupWide (u : UnionParams) (row file : Byte) (diag : Index) : BitVec 32 :=
  (LineTable.countFlip ⟨u.rowPos % 8, Nat.mod_lt _ (by decide)⟩ row).setWidth 32 +
  (LineTable.countFlip ⟨u.filePos % 8, Nat.mod_lt _ (by decide)⟩ file).setWidth 32 +
  (LineTable.countFlip ⟨u.d0Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex diag u.d0Map) +
   LineTable.countFlip ⟨u.d1Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex diag u.d1Map)).setWidth 32

theorem lookupWide_toNat (u : UnionParams) (row file : Byte) (diag : Index) :
    (lookupWide u row file diag).toNat = (lookup u row file diag).toNat := by
  have h0 := LineTable.countFlip_bound ⟨u.rowPos % 8, Nat.mod_lt _ (by decide)⟩ row
  have h1 := LineTable.countFlip_bound ⟨u.filePos % 8, Nat.mod_lt _ (by decide)⟩ file
  have h2 := LineTable.countFlip_bound ⟨u.d0Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex diag u.d0Map)
  have h3 := LineTable.countFlip_bound ⟨u.d1Pos % 8, Nat.mod_lt _ (by decide)⟩ (projectIndex diag u.d1Map)
  simp only [lookupWide, lookup, BitVec.toNat_add, BitVec.toNat_setWidth]
  omega

def countWide (p : Word) (square : Nat) : BitVec 32 :=
  let u := unionParams square
  lookupWide u ((p >>> u.rankShift).setWidth 8) ((pext p u.fileCells).setWidth 8)
    ((pext p u.unionCells).setWidth 16)
def countOppWide (p : Word) (square : Nat) : BitVec 32 :=
  let u := unionParams square
  lookupWide u (((p >>> u.rankShift).setWidth 8) ^^^ 255) (((pext p u.fileCells).setWidth 8) ^^^ 255)
    (((pext p u.unionCells).setWidth 16) ^^^ (BitVec.ofNat 16 (u.size - 1)))

theorem countWide_toNat (p : Word) (square : Fin 64) :
    (countWide p square.val).toNat = 2 * reference p square.val := by
  rw [countWide, lookupWide_toNat]
  exact count_toNat p square

theorem countOppWide_toNat (p : Word) (square : Fin 64) :
    (countOppWide p square.val).toNat = 2 * reference (~~~p) square.val := by
  rw [countOppWide, lookupWide_toNat]
  exact countOpp_toNat p square

theorem ray_eq_geometry (p : Word) (acc : Nat) (cells : List Nat) :
    ray p acc cells = Geometry.ray p acc cells := by
  induction cells generalizing acc with
  | nil => rfl
  | cons i cells ih => simp only [ray, Geometry.ray, ih]

theorem reference_eq_geometry (p : Word) (square : Nat) :
    reference p square = Geometry.reference p square := by
  simp only [reference, Geometry.reference, ray_eq_geometry]
  rfl

end NeuralReversi.Solve1.Bmi2

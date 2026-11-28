module
public import NeuralReversi.Bitboard.CornerProperties

@[expose] public section

namespace NeuralReversi.LegalMoves
open Bitboard

structure Raw where
  bits : Bits
  deriving DecidableEq, Repr

def empty : Raw := ⟨0⟩
def count (m : Raw) : Nat := (Bitboard.count m.bits).toNat
def bounded (m : Raw) : Prop := count m ≤ 35

def andNot (m : Raw) (mask : Bits) : Raw := ⟨m.bits &&& ~~~mask⟩
def corners (m : Raw) : Raw := ⟨Bitboard.corners m.bits⟩
def nonCorners (m : Raw) : Raw := ⟨Bitboard.nonCorners m.bits⟩
def bitand (m : Raw) (mask : Bits) : Raw := ⟨m.bits &&& mask⟩

def Subset (a b : Raw) : Prop :=
  ∀ i, a.bits.getLsbD i = true → b.bits.getLsbD i = true

theorem subset_refl (a : Raw) : Subset a a := by intro _ h; exact h

theorem subset_trans {a b c : Raw} (ab : Subset a b) (bc : Subset b c) :
    Subset a c := by intro i h; exact bc i (ab i h)

theorem bitand_subset (m : Raw) (mask : Bits) : Subset (bitand m mask) m := by
  intro i h
  exact (Bool.and_eq_true_iff.mp (by simpa [bitand] using h)).1

theorem andNot_subset (m : Raw) (mask : Bits) : Subset (andNot m mask) m :=
  bitand_subset m (~~~mask)

theorem corners_subset (m : Raw) : Subset (corners m) m :=
  bitand_subset m cornerMask

theorem nonCorners_subset (m : Raw) : Subset (nonCorners m) m :=
  bitand_subset m (~~~cornerMask)

theorem count_mono {a b : Raw} (h : Subset a b) : count a ≤ count b := by
  simp only [count, (populationCountContract_proven a.bits).1,
    (populationCountContract_proven b.bits).1]
  exact List.countP_mono_left (by intro i _ hi; exact h i hi)

theorem bounded_of_subset {a b : Raw} (h : Subset a b) (hb : bounded b) :
    bounded a := Nat.le_trans (count_mono h) hb

theorem empty_bounded : bounded empty := by unfold bounded; decide

inductive Filter where
  | andNot (mask : Bits)
  | corners
  | nonCorners
  | bitand (mask : Bits)

def applyFilter (f : Filter) (m : Raw) : Raw :=
  match f with
  | .andNot mask => andNot m mask
  | .corners => corners m
  | .nonCorners => nonCorners m
  | .bitand mask => bitand m mask

def filterMask : Filter → Bits
  | .andNot mask => ~~~mask
  | .corners => cornerMask
  | .nonCorners => ~~~cornerMask
  | .bitand mask => mask

theorem filter_bits_exact (f : Filter) (m : Raw) :
    (applyFilter f m).bits = m.bits &&& filterMask f := by
  cases f <;> rfl

theorem filter_membership_exact (f : Filter) (m : Raw) (i : Nat) :
    (applyFilter f m).bits.getLsbD i =
      (m.bits.getLsbD i && (filterMask f).getLsbD i) := by
  rw [filter_bits_exact, BitVec.getLsbD_and]

theorem bitand_membership_exact (m : Raw) (mask : Bits) (i : Nat) :
    (bitand m mask).bits.getLsbD i = (m.bits.getLsbD i && mask.getLsbD i) :=
  filter_membership_exact (.bitand mask) m i

theorem andNot_membership_exact (m : Raw) (mask : Bits) (i : Nat) (hi : i < 64) :
    (andNot m mask).bits.getLsbD i = (m.bits.getLsbD i && !mask.getLsbD i) := by
  simpa [filterMask, applyFilter, hi] using filter_membership_exact (.andNot mask) m i

theorem filter_subset (f : Filter) (m : Raw) : Subset (applyFilter f m) m := by
  cases f with
  | andNot mask => exact andNot_subset m mask
  | corners => exact corners_subset m
  | nonCorners => exact nonCorners_subset m
  | bitand mask => exact bitand_subset m mask

theorem filter_count_le (f : Filter) (m : Raw) : count (applyFilter f m) ≤ count m :=
  count_mono (filter_subset f m)

theorem filter_bounded (f : Filter) (m : Raw) (h : bounded m) :
    bounded (applyFilter f m) := bounded_of_subset (filter_subset f m) h

def applyFilters : List Filter → Raw → Raw
  | [], m => m
  | f :: fs, m => applyFilters fs (applyFilter f m)

theorem filters_subset (fs : List Filter) (m : Raw) : Subset (applyFilters fs m) m := by
  induction fs generalizing m with
  | nil => exact subset_refl m
  | cons f fs ih => exact subset_trans (ih _) (filter_subset f m)

theorem filters_count_le (fs : List Filter) (m : Raw) :
    count (applyFilters fs m) ≤ count m := count_mono (filters_subset fs m)

theorem filters_bounded (fs : List Filter) (m : Raw) (h : bounded m) :
    bounded (applyFilters fs m) := bounded_of_subset (filters_subset fs m) h

theorem corner_partition (m : Raw) :
    (corners m).bits ||| (nonCorners m).bits = m.bits ∧
    (corners m).bits &&& (nonCorners m).bits = 0 :=
  (cornerMaskContract_proven m.bits).2

theorem corner_partition_count (m : Raw) :
    count (corners m) + count (nonCorners m) = count m := by
  have h := count_or_of_disjoint (corners m).bits (nonCorners m).bits
    (corner_partition m).2
  rw [(corner_partition m).1] at h
  exact h.symm

theorem corner_count_le_four (m : Raw) : count (corners m) ≤ 4 :=
  cornerCount_le_four m.bits

def fromGenerated (debugAssertions : Bool) (bits : Bits) : Except String Raw :=
  if debugAssertions && decide (35 < count ⟨bits⟩) then
    .error "LegalMoves::new debug assertion failed"
  else .ok ⟨bits⟩

theorem debug_success_iff (bits : Bits) (m : Raw) :
    fromGenerated true bits = .ok m ↔ m = ⟨bits⟩ ∧ bounded m := by
  by_cases h : 35 < count ⟨bits⟩
  · simp [fromGenerated, h]
    intro eq
    subst m
    exact Nat.not_le_of_gt h
  · have hb : bounded ⟨bits⟩ := Nat.le_of_not_gt h
    simp only [fromGenerated, Bool.true_and, h, decide_false, Bool.false_eq_true,
      ↓reduceIte, Except.ok.injEq]
    constructor
    · intro eq; subst m; exact ⟨rfl, hb⟩
    · intro hm; exact hm.1.symm

theorem release_unconditional (bits : Bits) :
    fromGenerated false bits = .ok ⟨bits⟩ := rfl

theorem constructor_bounded_of_initial (debug : Bool) (bits : Bits) (m : Raw)
    (initial : bounded ⟨bits⟩) (accepted : fromGenerated debug bits = .ok m) :
    bounded m := by
  cases debug with
  | false => have eq : (⟨bits⟩ : Raw) = m := Except.ok.inj accepted; simpa [← eq] using initial
  | true => exact (debug_success_iff bits m).mp accepted |>.2

theorem constructor_filters_bounded (debug : Bool) (bits : Bits) (m : Raw)
    (initial : bounded ⟨bits⟩) (accepted : fromGenerated debug bits = .ok m)
    (fs : List Filter) : bounded (applyFilters fs m) :=
  filters_bounded fs m (constructor_bounded_of_initial debug bits m initial accepted)

def NonEmpty := { m : Raw // m.bits ≠ 0 }

def nonEmpty (m : Raw) : Option NonEmpty :=
  if h : m.bits = 0 then none else some ⟨m, h⟩

theorem nonEmpty_none_iff (m : Raw) : nonEmpty m = none ↔ m.bits = 0 := by
  by_cases h : m.bits = 0 <;> simp [nonEmpty, h]

theorem nonEmpty_get (m : Raw) (n : NonEmpty) (accepted : nonEmpty m = some n) :
    n.val = m := by
  by_cases h : m.bits = 0
  · simp [nonEmpty, h] at accepted
  · rw [nonEmpty, dite_eq_right h] at accepted
    have eq := Option.some.inj accepted
    subst eq
    rfl

example : fromGenerated false (~~~0#64) = .ok ⟨~~~0#64⟩ := rfl
example : ¬ bounded ⟨~~~0#64⟩ := by unfold bounded; decide
example : fromGenerated true (~~~0#64) =
    .error "LegalMoves::new debug assertion failed" := rfl
example : count (bitand ⟨~~~0#64⟩ (~~~0#64)) = 64 := rfl
example : count (corners ⟨~~~0#64⟩) = 4 := rfl
example : count (nonCorners ⟨~~~0#64⟩) = 60 := rfl

end NeuralReversi.LegalMoves

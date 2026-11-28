module
public import NeuralReversi.Solve1.Policy

@[expose] public section

namespace NeuralReversi.Solve1.Packed

structure Counts where
  low : BitVec 8
  high : BitVec 8
  deriving DecidableEq, Repr

def pair16 (p : Counts) : BitVec 16 :=
  p.low.setWidth 16 ||| (p.high.setWidth 16 <<< 8)

def entry32 (p : Counts) : BitVec 32 := (pair16 p).setWidth 32

theorem pair16_value (p : Counts) :
    (pair16 p).toNat = p.low.toNat + 256 * p.high.toNat := by
  have low := p.low.isLt
  have high := p.high.isLt
  simp only [pair16, BitVec.toNat_or, BitVec.toNat_shiftLeft, BitVec.toNat_setWidth,
    Nat.shiftLeft_eq]
  change (p.low.toNat % 65536) ||| ((p.high.toNat % 65536 * 256) % 65536) = _
  rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (show p.high.toNat < 65536 by omega),
    Nat.mod_eq_of_lt (by omega), Nat.or_comm]
  have sum := Nat.shiftLeft_add_eq_or_of_lt (i := 8) low p.high.toNat
  rw [Nat.shiftLeft_eq] at sum
  rw [← sum]
  omega

theorem entry32_value (p : Counts) :
    (entry32 p).toNat = p.low.toNat + 256 * p.high.toNat := by
  rw [entry32, BitVec.toNat_setWidth, pair16_value]
  have low := p.low.isLt
  have high := p.high.isLt
  exact Nat.mod_eq_of_lt (by omega)

def lowTotal : List Counts → Nat
  | [] => 0
  | p :: ps => p.low.toNat + lowTotal ps

def highTotal : List Counts → Nat
  | [] => 0
  | p :: ps => p.high.toNat + highTotal ps

def total (ps : List Counts) : Nat := lowTotal ps + 256 * highTotal ps

def packedSum (ps : List Counts) : BitVec 32 :=
  ps.foldl (fun acc p => acc + entry32 p) 0

def AddsSafe : List Counts → BitVec 32 → Prop
  | [], _ => True
  | p :: ps, acc => BitVec.uaddOverflow acc (entry32 p) = false ∧ AddsSafe ps (acc + entry32 p)

theorem fold_exact_safe (ps : List Counts) (acc : BitVec 32)
    (small : acc.toNat + total ps < 4294967296) :
    (ps.foldl (fun a p => a + entry32 p) acc).toNat = acc.toNat + total ps ∧
    AddsSafe ps acc := by
  induction ps generalizing acc with
  | nil => simp [total, lowTotal, highTotal, AddsSafe]
  | cons p ps ih =>
    have totals : total (p :: ps) = (entry32 p).toNat + total ps := by
      simp only [total, lowTotal, highTotal, entry32_value]
      omega
    have addSmall : acc.toNat + (entry32 p).toNat < 4294967296 := by omega
    have addExact : (acc + entry32 p).toNat = acc.toNat + (entry32 p).toNat := by
      rw [BitVec.toNat_add]
      exact Nat.mod_eq_of_lt addSmall
    have rest := ih (acc + entry32 p) (by omega)
    refine ⟨?_, ?_, rest.2⟩
    · simp only [List.foldl_cons]
      rw [rest.1, addExact, totals]
      omega
    · simp only [BitVec.uaddOverflow, decide_eq_false_iff_not]
      omega

def LineBounds (ps : List Counts) : Prop :=
  ∀ p ∈ ps, p.low.toNat ≤ 12 ∧ p.high.toNat ≤ 12

theorem totals_bounded (ps : List Counts) (bound : LineBounds ps) :
    lowTotal ps ≤ 12 * ps.length ∧ highTotal ps ≤ 12 * ps.length := by
  induction ps with
  | nil => simp [lowTotal, highTotal]
  | cons p ps ih =>
    have head := bound p (by simp)
    have tail := ih (by intro q h; exact bound q (by simp [h]))
    simp only [lowTotal, highTotal, List.length_cons]
    omega

theorem packed_value_safe (ps : List Counts) (lines : ps.length ≤ 4) (bound : LineBounds ps) :
    (packedSum ps).toNat = lowTotal ps + 256 * highTotal ps ∧
    AddsSafe ps 0 ∧ lowTotal ps ≤ 48 ∧ highTotal ps ≤ 48 := by
  have totals := totals_bounded ps bound
  have low : lowTotal ps ≤ 48 := by omega
  have high : highTotal ps ≤ 48 := by omega
  have fold := fold_exact_safe ps 0 (by change 0 + total ps < _; unfold total; omega)
  exact ⟨by simpa [packedSum, total] using fold.1, fold.2, low, high⟩

theorem unpack_exact (ps : List Counts) (lines : ps.length ≤ 4) (bound : LineBounds ps) :
    (packedSum ps &&& 0xff).toNat = lowTotal ps ∧
    (packedSum ps >>> 8).toNat = highTotal ps := by
  have value := packed_value_safe ps lines bound
  constructor
  · rw [BitVec.toNat_and]
    change (packedSum ps).toNat &&& (2 ^ 8 - 1) = _
    rw [Nat.and_two_pow_sub_one_eq_mod, value.1]
    omega
  · rw [BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow, value.1]
    omega

theorem policy_inputs (ps : List Counts) (lines : ps.length ≤ 4) (bound : LineBounds ps) :
    (packedSum ps &&& 0xff).toInt = (lowTotal ps : Int) ∧
    (packedSum ps >>> 8).toInt = (highTotal ps : Int) ∧
    lowTotal ps ≤ 96 ∧ highTotal ps ≤ 96 := by
  have unpack := unpack_exact ps lines bound
  have sizes := packed_value_safe ps lines bound
  refine ⟨?_, ?_, by omega, by omega⟩
  · rw [BitVec.toInt_eq_toNat_of_lt (by omega), unpack.1]
  · rw [BitVec.toInt_eq_toNat_of_lt (by omega), unpack.2]

example : (packedSum (List.replicate 4 ⟨12, 12⟩) &&& 0xff).toNat = 48 := rfl
example : (packedSum (List.replicate 4 ⟨12, 12⟩) >>> 8).toNat = 48 := rfl
example : (packedSum [⟨255, 0⟩, ⟨1, 0⟩] >>> 8).toNat = 1 := rfl

end NeuralReversi.Solve1.Packed

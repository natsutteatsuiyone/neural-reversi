import Std

namespace NeuralReversi.Solve1

abbrev ScoreWord := BitVec 32

def scoreBase (n : Nat) : Int := 2 * (n : Int) - 62

def scoreIfBothPass (n : Nat) : Int :=
  let base := scoreBase n
  if base > 0 then base else base - 2

def solveCounts (n nf nfOpp : Nat) (alpha : Int) : Int :=
  let base := scoreBase n
  if nf ≠ 0 then base + nf
  else
    let sip := scoreIfBothPass n
    if sip > alpha then
      if nfOpp > 0 then base - 2 - nfOpp else sip
    else sip

def exactCounts (n nf nfOpp : Nat) : Int :=
  if nf ≠ 0 then scoreBase n + nf
  else if nfOpp > 0 then scoreBase n - 2 - nfOpp
  else scoreIfBothPass n

def NWSResult (alpha value result : Int) : Prop :=
  value ≤ result ∧ (result > alpha → result = value) ∧
    (result ≤ alpha ↔ value ≤ alpha)

theorem solveCounts_sound (n nf nfOpp : Nat) (alpha : Int) :
    NWSResult alpha (exactCounts n nf nfOpp) (solveCounts n nf nfOpp alpha) := by
  by_cases hf : nf = 0 <;> by_cases ho : nfOpp = 0 <;>
    by_cases hb : scoreBase n > 0 <;> by_cases ha : scoreIfBothPass n > alpha <;>
    simp [NWSResult, solveCounts, exactCounts, hf, ho, scoreIfBothPass, scoreBase] at * <;> omega


def natWord (n : Nat) : ScoreWord := BitVec.ofNat 32 n

def baseWord (n : Nat) : ScoreWord := 2 * natWord n - 64 + 2

def passWord (base : ScoreWord) : ScoreWord :=
  if base.toInt > 0 then base else base - 2

def scalarWord (n nf no : Nat) (alpha : ScoreWord) : ScoreWord :=
  let score := baseWord n + natWord nf
  if natWord nf = 0#32 then
    let sip := passWord score
    if sip.toInt > alpha.toInt then
      if (natWord no).toInt > 0 then score - 2 - natWord no else sip
    else sip
  else score

def bmi2Word (n nf no : Nat) (alpha : ScoreWord) : ScoreWord :=
  let base := baseWord n
  let score := base + natWord nf
  if natWord nf ≠ 0#32 then score
  else
    let sip := passWord base
    if sip.toInt ≤ alpha.toInt then sip
    else if (natWord no).toInt > 0 then base - 2 - natWord no else sip

def neonWord (n nf no : Nat) (alpha : ScoreWord) : ScoreWord :=
  let base := baseWord n
  let sip := passWord base
  let zeroPath := if (decide (sip.toInt > alpha.toInt) && decide ((natWord no).toInt > 0))
    then base - 2 - natWord no else sip
  if natWord nf ≠ 0#32 then base + natWord nf else zeroPath

theorem natWord_value (n : Nat) (small : n ≤ 96) : (natWord n).toInt = (n : Int) := by
  have nat : (natWord n).toNat = n := Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by omega), nat]

theorem natWord_zero (n : Nat) (small : n ≤ 96) : natWord n = 0#32 ↔ n = 0 := by
  constructor
  · intro h
    have := natWord_value n small
    rw [h] at this
    change (0 : Int) = (n : Int) at this
    omega
  · intro h; subst n; rfl

theorem add_bridge (a b : ScoreWord)
    (lo : -2147483648 ≤ a.toInt + b.toInt) (hi : a.toInt + b.toInt < 2147483648) :
    (a + b).toInt = a.toInt + b.toInt ∧ BitVec.saddOverflow a b = false := by
  constructor
  · rw [BitVec.toInt_add]
    exact Int.bmod_eq_of_le lo hi
  · simp only [BitVec.saddOverflow, Bool.or_eq_false_iff, decide_eq_false_iff_not]
    omega

theorem sub_bridge (a b : ScoreWord)
    (lo : -2147483648 ≤ a.toInt - b.toInt) (hi : a.toInt - b.toInt < 2147483648) :
    (a - b).toInt = a.toInt - b.toInt ∧ BitVec.ssubOverflow a b = false := by
  constructor
  · rw [BitVec.toInt_sub]
    exact Int.bmod_eq_of_le lo hi
  · simp only [BitVec.ssubOverflow, Bool.or_eq_false_iff, decide_eq_false_iff_not]
    omega

theorem base_arithmetic (n : Nat) (bound : n ≤ 64) :
    (baseWord n).toInt = scoreBase n ∧
    BitVec.smulOverflow 2 (natWord n) = false ∧
    BitVec.ssubOverflow (2 * natWord n) 64 = false ∧
    BitVec.saddOverflow (2 * natWord n - 64) 2 = false := by
  have two : (2 : ScoreWord).toInt = 2 := rfl
  have sixtyFour : (64 : ScoreWord).toInt = 64 := rfl
  have nv := natWord_value n (by omega)
  have product : (2 * natWord n).toInt = 2 * (n : Int) := by
    rw [BitVec.toInt_mul, nv]
    change (2 * (n : Int)).bmod 4294967296 = _
    exact Int.bmod_eq_of_le (by omega) (by omega)
  have sub := sub_bridge (2 * natWord n) 64 (by change -2147483648 ≤ _ - 64; omega)
    (by change _ - 64 < 2147483648; omega)
  have add := add_bridge (2 * natWord n - 64) 2 (by change -2147483648 ≤ _ + 2; omega)
    (by change _ + 2 < 2147483648; omega)
  refine ⟨?_, ?_, sub.2, add.2⟩
  · change (2 * natWord n - 64 + 2).toInt = _
    rw [add.1, sub.1, product]
    change 2 * (n : Int) - 64 + 2 = 2 * (n : Int) - 62
    omega
  · simp only [BitVec.smulOverflow, nv]
    change (decide (2 * (n : Int) ≥ 2147483648) || decide (2 * (n : Int) < -2147483648)) = false
    simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not]
    omega

theorem score_arithmetic (n nf no : Nat) (hn : n ≤ 64) (hf : nf ≤ 96) (ho : no ≤ 96) :
    (baseWord n + natWord nf).toInt = scoreBase n + nf ∧
    (baseWord n - 2).toInt = scoreBase n - 2 ∧
    (baseWord n - 2 - natWord no).toInt = scoreBase n - 2 - no ∧
    BitVec.saddOverflow (baseWord n) (natWord nf) = false ∧
    BitVec.ssubOverflow (baseWord n) 2 = false ∧
    BitVec.ssubOverflow (baseWord n - 2) (natWord no) = false := by
  have two : (2 : ScoreWord).toInt = 2 := rfl
  have base := (base_arithmetic n hn).1
  have fv := natWord_value nf hf
  have ov := natWord_value no ho
  have range : -62 ≤ (baseWord n).toInt ∧ (baseWord n).toInt ≤ 66 := by
    rw [base]; unfold scoreBase; omega
  have add := add_bridge (baseWord n) (natWord nf) (by omega) (by omega)
  have sub := sub_bridge (baseWord n) 2 (by change -2147483648 ≤ _ - 2; omega)
    (by change _ - 2 < 2147483648; omega)
  have sub2 := sub_bridge (baseWord n - 2) (natWord no) (by change -2147483648 ≤ _ - _; omega)
    (by change _ - _ < 2147483648; omega)
  refine ⟨?_, ?_, ?_, add.2, sub.2, sub2.2⟩
  · rw [add.1, base, fv]
  · rw [sub.1, base]; rfl
  · rw [sub2.1, sub.1, base, ov]; rfl

theorem passWord_value (n : Nat) (hn : n ≤ 64) :
    (passWord (baseWord n)).toInt = scoreIfBothPass n := by
  have base := (base_arithmetic n hn).1
  have sub := (score_arithmetic n 0 0 hn (by omega) (by omega)).2.1
  by_cases h : scoreBase n > 0
  · simp only [passWord, base, h, ↓reduceIte, scoreIfBothPass]
  · simp only [passWord, base, h, ↓reduceIte, scoreIfBothPass]
    exact sub

theorem backend_word_equivalence (n nf no : Nat) (alpha : ScoreWord) :
    scalarWord n nf no alpha = bmi2Word n nf no alpha ∧
    neonWord n nf no alpha = bmi2Word n nf no alpha := by
  by_cases z : natWord nf = 0#32
  · have score : baseWord n + natWord nf = baseWord n := by rw [z, BitVec.add_zero]
    by_cases hp : (passWord (baseWord n)).toInt > alpha.toInt
    · have notle : ¬ (passWord (baseWord n)).toInt ≤ alpha.toInt := by omega
      by_cases ho : (natWord no).toInt > 0 <;>
        simp only [scalarWord, bmi2Word, neonWord, z, BitVec.add_zero, hp, notle, ho,
          ne_eq, not_true_eq_false, ↓reduceIte, decide_true, decide_false, Bool.true_and, Bool.false_eq_true, and_self]
    · have le : (passWord (baseWord n)).toInt ≤ alpha.toInt := by omega
      simp only [scalarWord, bmi2Word, neonWord, z, BitVec.add_zero, hp, le,
        ne_eq, not_true_eq_false, ↓reduceIte, decide_false, Bool.false_and, Bool.false_eq_true, and_self]
  · simp only [scalarWord, bmi2Word, neonWord, z, ne_eq, not_false_eq_true, ↓reduceIte, and_self]

theorem bmi2_bridge (n nf no : Nat) (alpha : ScoreWord)
    (hn : n ≤ 64) (hf : nf ≤ 96) (ho : no ≤ 96) :
    (bmi2Word n nf no alpha).toInt = solveCounts n nf no alpha.toInt := by
  have arithmetic := score_arithmetic n nf no hn hf ho
  have pass := passWord_value n hn
  have ov := natWord_value no ho
  have zero := natWord_zero nf hf
  by_cases z : nf = 0
  · have wz := zero.mpr z
    by_cases hp : scoreIfBothPass n > alpha.toInt
    · have hp' : ¬ (passWord (baseWord n)).toInt ≤ alpha.toInt := by omega
      by_cases ho' : no > 0
      · have positive : (natWord no).toInt > 0 := by omega
        simp only [bmi2Word, wz, ne_eq, not_true_eq_false, ↓reduceIte, hp', positive]
        simp only [solveCounts, z, ne_eq, not_true_eq_false, ↓reduceIte, hp, ho']
        exact arithmetic.2.2.1
      · have nonpositive : ¬ (natWord no).toInt > 0 := by omega
        simp only [bmi2Word, wz, ne_eq, not_true_eq_false, ↓reduceIte, hp', nonpositive]
        simp only [solveCounts, z, ne_eq, not_true_eq_false, ↓reduceIte, hp, ho']
        exact pass
    · have hp' : (passWord (baseWord n)).toInt ≤ alpha.toInt := by omega
      simp only [bmi2Word, wz, ne_eq, not_true_eq_false, ↓reduceIte, hp']
      simp only [solveCounts, z, ne_eq, not_true_eq_false, ↓reduceIte, hp]
      exact pass
  · have wz : natWord nf ≠ 0#32 := mt zero.mp z
    simp only [bmi2Word, wz, ↓reduceIte, solveCounts, z, ne_eq, not_false_eq_true]
    exact arithmetic.1

theorem scalar_bridge (n nf no : Nat) (alpha : ScoreWord)
    (hn : n ≤ 64) (hf : nf ≤ 96) (ho : no ≤ 96) :
    (scalarWord n nf no alpha).toInt = solveCounts n nf no alpha.toInt := by
  rw [(backend_word_equivalence n nf no alpha).1]
  exact bmi2_bridge n nf no alpha hn hf ho

theorem scalar_nws (n nf no : Nat) (alpha : ScoreWord)
    (hn : n ≤ 64) (hf : nf ≤ 96) (ho : no ≤ 96) :
    NWSResult alpha.toInt (exactCounts n nf no) (scalarWord n nf no alpha).toInt := by
  rw [scalar_bridge n nf no alpha hn hf ho]
  exact solveCounts_sound n nf no alpha.toInt

theorem backend_nws (n nf no : Nat) (alpha : ScoreWord)
    (hn : n ≤ 64) (hf : nf ≤ 96) (ho : no ≤ 96) :
    NWSResult alpha.toInt (exactCounts n nf no) (bmi2Word n nf no alpha).toInt ∧
    NWSResult alpha.toInt (exactCounts n nf no) (neonWord n nf no alpha).toInt := by
  have equal := backend_word_equivalence n nf no alpha
  rw [← equal.1, equal.2, ← equal.1]
  exact ⟨scalar_nws n nf no alpha hn hf ho, scalar_nws n nf no alpha hn hf ho⟩

theorem exactCounts_terminal (n f fo : Nat) :
    exactCounts n (2*f) (2*fo) =
      (if (f : Int) > 0 then ((n : Int)+1+f)-(63-n-f)
       else if (fo : Int) > 0 then ((n : Int)-fo)-(64-n+fo)
       else if (n : Int) > 63-n then ((n : Int)+1)-(63-n) else (n : Int)-(64-n)) := by
  by_cases hf : (f : Int) > 0
  · have hfn : 2*f ≠ 0 := by omega
    simp [exactCounts, hfn, scoreBase]
    omega
  · have hf0 : f = 0 := by omega
    subst f
    by_cases hfo : (fo : Int) > 0
    · have hfon : 2*fo > 0 := by omega
      simp [exactCounts, scoreBase]
      omega
    · have hfo0 : fo = 0 := by omega
      subst fo
      by_cases hn : n > 31
      · have hb : scoreBase n > 0 := by unfold scoreBase; omega
        have hcmp : (n : Int) > 63-n := by omega
        simp [exactCounts, scoreIfBothPass, scoreBase]
        omega
      · have hb : ¬scoreBase n > 0 := by unfold scoreBase; omega
        have hcmp : ¬(n : Int) > 63-n := by omega
        simp [exactCounts, scoreIfBothPass, scoreBase]
        omega


theorem solveCounts_coarse_bound (n nf no : Nat) (alpha : Int)
    (hn : n ≤ 64) (hf : nf ≤ 96) (ho : no ≤ 96) :
    -160 ≤ solveCounts n nf no alpha ∧ solveCounts n nf no alpha ≤ 162 := by
  by_cases hz : nf = 0 <;> by_cases hop : no > 0 <;>
    by_cases hb : scoreBase n > 0 <;>
    by_cases ha : scoreIfBothPass n > alpha <;>
    simp [solveCounts, scoreIfBothPass, scoreBase, hz, hop] at * <;> omega

theorem word_result_bound (n nf no : Nat) (alpha : ScoreWord)
    (hn : n ≤ 64) (hf : nf ≤ 96) (ho : no ≤ 96) :
    -160 ≤ (scalarWord n nf no alpha).toInt ∧
      (scalarWord n nf no alpha).toInt ≤ 162 := by
  rw [scalar_bridge n nf no alpha hn hf ho]
  exact solveCounts_coarse_bound n nf no alpha.toInt hn hf ho

example : (scalarWord 31 0 2 0).toInt = -2 := rfl
example : exactCounts 31 0 2 = -4 := rfl
example : (scalarWord 31 0 2 0x80000000#32).toInt = -4 := rfl
example : (scalarWord 31 0 2 0x7FFFFFFF#32).toInt = -2 := rfl
example : (scalarWord 32 0 0 0).toInt = 2 := rfl
example : (scalarWord 31 0 0 0).toInt = -2 := rfl

end NeuralReversi.Solve1

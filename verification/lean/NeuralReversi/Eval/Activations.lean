import Lean

namespace NeuralReversi.Eval.Activations

private def satU8 (x : Int) : Int := min (max 0 x) 255
private def satI16 (x : Int) : Int := min (max (-32768) x) 32767

def reluSpec (x : Int) : Int := satU8 (x / 64)

def squareSpec (x : Int) : Int := satU8 (x * x / 1048576)

def scalarSquare (x : Int) : Int :=
  let s := satI16 x
  min (s * s / 1048576) 255

private def x86Square (x : Int) : Int :=
  let s := satI16 x
  satU8 (s * s / 65536 / 16)

private def neonSquare (x : Int) : Int :=
  let s := satI16 x
  -- SQDMULH doubles the product and saturates its high half.
  satU8 (satI16 (2 * (s * s) / 65536) / 32)

private theorem satI16_bounds (x : Int) : -32768 ≤ satI16 x ∧ satI16 x ≤ 32767 := by
  unfold satI16
  omega

private theorem satI16_eq_self {x : Int} (lo : -32768 ≤ x) (hi : x ≤ 32767) :
    satI16 x = x := by
  unfold satI16
  omega

private theorem square_le_square {a b : Int} (hn : 0 ≤ a) (hle : a ≤ b) : a * a ≤ b * b :=
  Int.le_trans (Int.mul_le_mul_of_nonneg_left hle hn)
    (Int.mul_le_mul_of_nonneg_right hle (Int.le_trans hn hle))

private theorem square_nonneg (x : Int) : 0 ≤ x * x := by
  by_cases h : 0 ≤ x
  · exact Int.mul_nonneg h h
  · exact Int.mul_nonneg_of_nonpos_of_nonpos (by omega) (by omega)

private theorem satI16_square_bounds (x : Int) :
    0 ≤ satI16 x * satI16 x ∧ satI16 x * satI16 x ≤ 1073741824 := by
  obtain ⟨lo, hi⟩ := satI16_bounds x
  refine ⟨square_nonneg _, ?_⟩
  by_cases hn : 0 ≤ satI16 x
  · exact square_le_square hn (by omega : satI16 x ≤ 32768)
  · have bound := square_le_square (a := -satI16 x) (b := 32768) (by omega) (by omega)
    simpa only [Int.neg_mul_neg, show (32768 : Int) * 32768 = 1073741824 from rfl] using bound

theorem squareIntermediateBounds (x : Int) :
    0 ≤ satI16 x * satI16 x ∧ satI16 x * satI16 x < 2^63 ∧
    0 ≤ satI16 x * satI16 x / 65536 ∧ satI16 x * satI16 x / 65536 ≤ 32767 := by
  obtain ⟨lo, hi⟩ := satI16_square_bounds x
  omega

private theorem square_ge_at_clip (x : Int) (h : x < -32768 ∨ 32767 < x) :
    255 ≤ x * x / 1048576 := by
  have hz : 32767 * 32767 ≤ x * x := by
    rcases h with low | high
    · have hmul := square_le_square (a := 32767) (b := -x) (by decide) (by omega)
      simpa only [Int.neg_mul_neg] using hmul
    · exact square_le_square (by decide) (by omega)
  omega

private theorem scalarSquare_sound (x : Int) : scalarSquare x = squareSpec x := by
  obtain low | high | mid : x < -32768 ∨ 32767 < x ∨ (-32768 ≤ x ∧ x ≤ 32767) := by omega
  · have saturated : satI16 x = -32768 := by unfold satI16; omega
    have big := square_ge_at_clip x (.inl low)
    simp only [scalarSquare, squareSpec, satU8, saturated]
    omega
  · have saturated : satI16 x = 32767 := by unfold satI16; omega
    have big := square_ge_at_clip x (.inr high)
    simp only [scalarSquare, squareSpec, satU8, saturated]
    omega
  · have nn := square_nonneg x
    simp only [scalarSquare, squareSpec, satI16_eq_self mid.1 mid.2, satU8]
    omega

private theorem saturatedRelu_sound (x : Int) : satU8 (satI16 x / 64) = reluSpec x := by
  unfold reluSpec satI16 satU8
  omega

private theorem x86Square_sound (x : Int) : x86Square x = squareSpec x := by
  rw [← scalarSquare_sound x]
  dsimp only [x86Square, scalarSquare, satU8]
  rw [Int.ediv_ediv_of_nonneg (by decide : 0 ≤ (65536 : Int))]
  have nn := square_nonneg (satI16 x)
  omega

private theorem neonSquare_sound (x : Int) : neonSquare x = squareSpec x := by
  rw [← scalarSquare_sound x]
  let s := satI16 x
  let n := s * s
  have nn : 0 ≤ n := square_nonneg s
  have mulHi : 2 * n / 65536 = n / 32768 := by omega
  change satU8 (satI16 (2 * n / 65536) / 32) = min (n / 1048576) 255
  rw [mulHi, ← show n / 32768 / 32 = n / 1048576 from
    Int.ediv_ediv_of_nonneg (by decide : 0 ≤ (32768 : Int))]
  unfold satI16 satU8
  omega

/-- The two 128-bit lanes of PACKSSDW each read four values from each input vector. -/
private def avx2WordsIndex (i : Fin 16) : Fin 16 :=
  ⟨i.val / 8 * 4 + i.val % 8 / 4 * 8 + i.val % 4, by omega⟩

/-- PACKUSWB packs eight square results, then eight ReLU results, in each lane. -/
private def avx2PackedOutput (input : Fin 16 → Int32) (i : Fin 32) : Int :=
  let x := (input (avx2WordsIndex ⟨i.val / 16 * 8 + i.val % 8, by omega⟩)).toInt
  if i.val % 16 < 8 then x86Square x else satU8 (satI16 x / 64)

/-- VPERMD selects eight 32-bit groups in the order given by _mm256_set_epi32. -/
private def avx2Shuffle (i : Fin 32) : Fin 32 :=
  ⟨(i.val / 4 % 2 * 4 + i.val / 8) * 4 + i.val % 4, by omega⟩

private theorem avx2Shuffle_selection : ∀ i : Fin 32,
    let j := avx2Shuffle i
    (j.val % 16 < 8 ↔ i.val < 16) ∧
      (avx2WordsIndex ⟨j.val / 16 * 8 + j.val % 8, by omega⟩).val = i.val % 16 := by
  decide

inductive Backend where
  | scalar | avx2 | avx512 | neon

def fusedOutput (backend : Backend) (input : Fin 16 → Int32) (i : Fin 32) : Int :=
  let x := (input ⟨i.val % 16, by omega⟩).toInt
  match backend with
  | .scalar => if i.val < 16 then scalarSquare x else reluSpec x
  | .avx2 => avx2PackedOutput input (avx2Shuffle i)
  | .avx512 =>
    if i.val < 16 then x86Square x else satU8 (satI16 x / 64)
  | .neon => if i.val < 16 then neonSquare x else satU8 (satI16 x / 64)

def FusedContract : Prop :=
  ∀ (backend : Backend) (input : Fin 16 → Int32) (i : Fin 32),
    let x := (input ⟨i.val % 16, by omega⟩).toInt
    fusedOutput backend input i = if i.val < 16 then squareSpec x else reluSpec x

theorem fusedContract_proven : FusedContract := by
  intro backend input i
  cases backend with
  | scalar => simp only [fusedOutput, scalarSquare_sound]
  | avx512 => simp only [fusedOutput, x86Square_sound, saturatedRelu_sound]
  | neon => simp only [fusedOutput, neonSquare_sound, saturatedRelu_sound]
  | avx2 =>
    obtain ⟨hs, hi⟩ := avx2Shuffle_selection i
    have indexEq : avx2WordsIndex ⟨(avx2Shuffle i).val / 16 * 8 + (avx2Shuffle i).val % 8,
        by omega⟩ = (⟨i.val % 16, by omega⟩ : Fin 16) := Fin.ext hi
    simp only [fusedOutput, avx2PackedOutput, hs, indexEq, x86Square_sound, saturatedRelu_sound]

private def satU16 (x : Int) : Int := min (max 0 x) 65535
private def activationCeiling : Int := 255 * 64

def scReluSpec (x : Int) : Int :=
  let clamped := min (max 0 x) activationCeiling
  clamped * clamped / 1048576

private def scReluX86 (wide : Bool) (x : Int) : Int :=
  let s := if wide then satU16 (min x activationCeiling)
    else min (satU16 x) activationCeiling
  satU8 (s * s / 65536 / 16)

private def scReluNeon (x : Int) : Int :=
  let s := min (satU16 x) activationCeiling
  satU8 (satI16 (2 * (s * s) / 65536) / 32)

private theorem scReluSpec_bounds (x : Int) : 0 ≤ scReluSpec x ∧ scReluSpec x ≤ 254 := by
  let s := min (max 0 x) activationCeiling
  have range : 0 ≤ s ∧ s ≤ 16320 := by unfold s activationCeiling; omega
  have hit := square_nonneg s
  have bound : s * s ≤ 16320 * 16320 := square_le_square range.1 range.2
  change 0 ≤ s * s / 1048576 ∧ s * s / 1048576 ≤ 254
  omega

private theorem scReluX86_sound (wide : Bool) (x : Int) :
    scReluX86 wide x = scReluSpec x := by
  have packAfter : satU16 (min x activationCeiling) = min (max 0 x) activationCeiling := by
    unfold satU16 activationCeiling
    omega
  have packBefore : min (satU16 x) activationCeiling = min (max 0 x) activationCeiling := by
    unfold satU16 activationCeiling
    omega
  cases wide <;>
    simp only [scReluX86, Bool.false_eq_true, ↓reduceIte, packAfter, packBefore] <;>
    rw [Int.ediv_ediv_of_nonneg (by decide : 0 ≤ (65536 : Int))] <;>
    change satU8 (scReluSpec x) = scReluSpec x <;>
    obtain ⟨lo, hi⟩ := scReluSpec_bounds x <;> unfold satU8 <;> omega

private theorem scReluNeon_sound (x : Int) : scReluNeon x = scReluSpec x := by
  let s := min (satU16 x) activationCeiling
  have range : 0 ≤ s ∧ s ≤ 16320 := by unfold s satU16 activationCeiling; omega
  have same : s = min (max 0 x) activationCeiling := by
    unfold s satU16 activationCeiling
    omega
  have nn := square_nonneg s
  have bound : s * s ≤ 16320 * 16320 := square_le_square range.1 range.2
  have mulHi : 2 * (s * s) / 65536 = s * s / 32768 := by omega
  change satU8 (satI16 (2 * (s * s) / 65536) / 32) = scReluSpec x
  rw [mulHi, satI16_eq_self (by omega) (by omega),
    Int.ediv_ediv_of_nonneg (by decide : 0 ≤ (32768 : Int)), same]
  change satU8 (scReluSpec x) = scReluSpec x
  obtain ⟨lo, hi⟩ := scReluSpec_bounds x
  unfold satU8
  omega

private def scReluStoreLanes (backend : Backend) (size : Nat) : Nat :=
  match backend with
  | .avx512 => if size % 64 = 0 then 16 else if size % 32 = 0 then 8 else 0
  | .avx2 => if size % 32 = 0 then 8 else 0
  | .scalar | .neon => 0

def scReluStoreOrder (backend : Backend) (size i : Nat) : Nat :=
  let lanes := scReluStoreLanes backend size
  if lanes = 0 then i
  else
    let p := i % (lanes * 4)
    let q := p / 4
    i - p + 4 * (q / 4) + lanes * (q % 4) + p % 4

/-- In each 128-bit lane, PACKUSWB stores four groups of four values in
    the order of the four original input vectors. -/
private def packedLaneIndex (lanes pos : Nat) : Nat :=
  pos % 16 / 4 * lanes + pos / 16 * 4 + pos % 4

private def packedSource (lanes i : Nat) : Nat :=
  i / (4 * lanes) * (4 * lanes) + packedLaneIndex lanes (i % (4 * lanes))

private theorem packedSource_eq_order (backend : Backend) (size i : Nat)
    (hw : scReluStoreLanes backend size = 8 ∨ scReluStoreLanes backend size = 16) :
    packedSource (scReluStoreLanes backend size) i = scReluStoreOrder backend size i := by
  unfold scReluStoreOrder
  rcases hw with h | h <;> simp [h, packedSource, packedLaneIndex] <;> omega

private theorem scReluStoreOrder_lt (backend : Backend) (size i : Nat) (h : i < size) :
    scReluStoreOrder backend size i < size := by
  cases backend with
  | scalar | neon => exact h
  | avx2 =>
    by_cases h32 : size % 32 = 0 <;> simp [scReluStoreOrder, scReluStoreLanes, h32] <;> omega
  | avx512 =>
    by_cases h64 : size % 64 = 0 <;> by_cases h32 : size % 32 = 0 <;>
      simp [scReluStoreOrder, scReluStoreLanes, h64, h32] <;> omega

private theorem packedSource_injective (lanes i j : Nat)
    (hw : lanes = 8 ∨ lanes = 16) (same : packedSource lanes i = packedSource lanes j) :
    i = j := by
  let block := 4 * lanes
  have pos : 0 < block := by rcases hw with rfl | rfl <;> decide
  have inBlock (k : Nat) : packedLaneIndex lanes (k % block) < block := by
    rcases hw with rfl | rfl <;> unfold packedLaneIndex block <;> omega
  have quot (k : Nat) : packedSource lanes k / block = k / block := by
    have bound := inBlock k
    rcases hw with rfl | rfl <;> unfold packedSource block at * <;> omega
  have rem (k : Nat) : packedSource lanes k % block = packedLaneIndex lanes (k % block) := by
    have bound := inBlock k
    rcases hw with rfl | rfl <;> unfold packedSource block at * <;> omega
  have localPerm : ∀ a b : Fin (4 * lanes),
      packedLaneIndex lanes a.val = packedLaneIndex lanes b.val → a = b := by
    rcases hw with rfl | rfl <;> decide
  have q : i / block = j / block := by rw [← quot i, ← quot j, same]
  have hrem : packedLaneIndex lanes (i % block) = packedLaneIndex lanes (j % block) := by
    rw [← rem i, ← rem j, same]
  have r : i % block = j % block :=
    congrArg Fin.val (localPerm ⟨_, Nat.mod_lt i pos⟩ ⟨_, Nat.mod_lt j pos⟩ hrem)
  rw [← Nat.mod_add_div i block, ← Nat.mod_add_div j block, q, r]

private theorem nonzeroStoreLanes (backend : Backend) (size : Nat)
    (h : scReluStoreLanes backend size ≠ 0) :
    scReluStoreLanes backend size = 8 ∨ scReluStoreLanes backend size = 16 := by
  cases backend <;> simp_all [scReluStoreLanes] <;> split <;> simp_all

theorem scReluStoreOrder_injective (backend : Backend) (size i j : Nat)
    (same : scReluStoreOrder backend size i = scReluStoreOrder backend size j) :
    i = j := by
  by_cases h : scReluStoreLanes backend size = 0
  · simpa only [scReluStoreOrder, h, ↓reduceIte] using same
  · have wide := nonzeroStoreLanes backend size h
    rw [← packedSource_eq_order backend size i wide, ← packedSource_eq_order backend size j wide]
      at same
    exact packedSource_injective _ i j wide same

def scReluOutput (backend : Backend) (size : Nat) (input : Nat → Int32) (i : Nat) : Int :=
  let lanes := scReluStoreLanes backend size
  if lanes ≠ 0 then scReluX86 true (input (packedSource lanes i)).toInt
  else match backend with
    | .scalar => scReluSpec (input i).toInt
    | .avx2 | .avx512 =>
      if i < size / 16 * 16 then scReluX86 false (input i).toInt
      else scReluSpec (input i).toInt
    | .neon =>
      if i < size / 16 * 16 then scReluNeon (input i).toInt
      else scReluSpec (input i).toInt

def ScReluContract : Prop :=
  ∀ (backend : Backend) (size : Nat) (input : Nat → Int32) (i : Nat), i < size →
    let source := scReluStoreOrder backend size i
    source < size ∧ scReluOutput backend size input i = scReluSpec (input source).toInt ∧
      0 ≤ scReluOutput backend size input i ∧ scReluOutput backend size input i ≤ 254

theorem scReluContract_proven : ScReluContract := by
  intro backend size input i within
  have equal : scReluOutput backend size input i =
      scReluSpec (input (scReluStoreOrder backend size i)).toInt := by
    unfold scReluOutput
    by_cases h : scReluStoreLanes backend size = 0
    · simp only [h, ne_eq, not_true_eq_false, ↓reduceIte, scReluStoreOrder]
      cases backend <;> simp only [scReluX86_sound, scReluNeon_sound, ite_self]
    · have wide := nonzeroStoreLanes backend size h
      simp only [h, ne_eq, not_false_eq_true, ↓reduceIte, scReluX86_sound]
      rw [packedSource_eq_order backend size i wide]
  exact ⟨scReluStoreOrder_lt backend size i within, equal, equal ▸
    scReluSpec_bounds (input (scReluStoreOrder backend size i)).toInt⟩

private theorem mainStoreOrder_perm (backend : Backend) :
    List.Perm ((List.range 64).map (scReluStoreOrder backend 64)) (List.range 64) := by
  cases backend <;> decide

private theorem wordSum_perm {xs ys : List (BitVec 32)} (h : xs.Perm ys) : xs.sum = ys.sum := by
  induction h with
  | nil => rfl
  | cons x _ ih => simp only [List.sum_cons, ih]
  | swap x y l =>
    simp only [List.sum_cons]
    rw [← BitVec.add_assoc, ← BitVec.add_assoc, BitVec.add_comm x y]
  | trans _ _ left right => exact left.trans right

def LeadingDotProductContract : Prop :=
  ∀ (backend : Backend) (inputs : Nat → Int32) (weights : Nat → BitVec 32),
    let indices := List.range 64
    let canonical := fun i => BitVec.ofInt 32 (scReluSpec (inputs i).toInt) * weights i
    let packed := fun i => BitVec.ofInt 32 (scReluOutput backend 64 inputs i) *
      weights (scReluStoreOrder backend 64 i)
    (indices.map packed).sum = (indices.map canonical).sum

theorem leadingDotProductContract_proven : LeadingDotProductContract := by
  intro backend inputs weights indices canonical packed
  have eq : indices.map packed = (indices.map (scReluStoreOrder backend 64)).map canonical := by
    rw [List.map_map]
    apply List.map_congr_left
    intro i mem
    have within : i < 64 := List.mem_range.mp mem
    have value := (scReluContract_proven backend 64 inputs i within).2.1
    simp only [packed, canonical, Function.comp_apply, value]
  rw [eq]
  exact wordSum_perm ((mainStoreOrder_perm backend).map canonical)

end NeuralReversi.Eval.Activations

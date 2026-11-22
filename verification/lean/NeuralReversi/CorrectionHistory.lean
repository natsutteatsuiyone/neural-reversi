import NeuralReversi.TranspositionTable.Proofs

namespace NeuralReversi.CorrectionHistory

open TranspositionTable (Bound Depth)

private def clamp (lo hi x : Int) : Int := min hi (max lo x)

def correct (raw entry : Int) : Int := clamp (-16383) 16383 (raw + entry)

private def gap (raw entry score : Int) : Int :=
  clamp (-2048) 2048 (score - correct raw entry)

private def weight (depth : Depth) : Int := min (depth.toNat : Int) 16

private def change (raw entry score : Int) (depth : Depth) : Int :=
  ((gap raw entry score) * weight depth).tdiv 64

private def showsGap (bound : Bound) (eval score : Int) : Bool :=
  match bound with
  | .lower => score > eval
  | .upper => score < eval
  | .exact => true
  | .none => false

def update (raw entry score : Int) (bound : Bound) (depth : Depth) : Int :=
  if showsGap bound (correct raw entry) score then
    clamp (-2048) 2048 (entry + change raw entry score depth)
  else entry

def Sound (bound : Bound) (score target : Int) : Prop :=
  match bound with
  | .lower => score ≤ target
  | .upper => target ≤ score
  | .exact => score = target
  | .none => True

private theorem weight_range (depth : Depth) : 0 ≤ weight depth ∧ weight depth ≤ 16 := by
  unfold weight
  omega

private theorem scaled_between (g w : Int) (hw : 0 ≤ w ∧ w ≤ 16) :
    min 0 g ≤ (g * w).tdiv 64 ∧ (g * w).tdiv 64 ≤ max 0 g := by
  by_cases hg : 0 ≤ g
  · have prod := Int.mul_nonneg hg hw.1
    have high := Int.mul_le_mul_of_nonneg_left hw.2 hg
    rw [Int.tdiv_eq_ediv_of_nonneg prod]
    omega
  · have prod := Int.mul_nonpos_of_nonpos_of_nonneg (Int.le_of_lt (Int.lt_of_not_ge hg)) hw.1
    have low := Int.mul_le_mul_of_nonpos_left (Int.le_of_lt (Int.lt_of_not_ge hg)) hw.2
    rw [Int.tdiv_cases]
    simp only [show (0 : Int) ≤ 64 by decide, ite_true]
    split <;> omega

private def FitsI32 (n : Int) : Prop := -2147483648 ≤ n ∧ n ≤ 2147483647

def BoundedUpdate : Prop :=
  ∀ (raw entry score target : Int) (bound : Bound) (depth : Depth),
    -32767 ≤ raw → raw ≤ 32767 → -32767 ≤ score → score ≤ 32767 →
    -2048 ≤ entry → entry ≤ 2048 → Sound bound score target →
    let next := update raw entry score bound depth
    let oldEval := correct raw entry
    let newEval := correct raw next
    (-2048 : Int) ≤ next ∧ next ≤ 2048 ∧
      min oldEval target ≤ newEval ∧ newEval ≤ max oldEval target ∧
      FitsI32 (raw + entry) ∧ FitsI32 (score - oldEval) ∧
      FitsI32 (gap raw entry score * weight depth) ∧
      FitsI32 (entry + change raw entry score depth)

private theorem correct_moves (raw entry next : Int) :
    min 0 (next - entry) ≤ correct raw next - correct raw entry ∧
    correct raw next - correct raw entry ≤ max 0 (next - entry) := by
  unfold correct clamp
  omega

private theorem boundedUpdate_proven : BoundedUpdate := by
  intro raw entry score target bound depth hr₁ hr₂ hs₁ hs₂ he₁ he₂ sound
  dsimp only
  let old := correct raw entry
  let g := gap raw entry score
  let step := change raw entry score depth
  have hw := weight_range depth
  have hg : -2048 ≤ g ∧ g ≤ 2048 := by dsimp [g, gap, clamp]; omega
  have hprod₁ := Int.mul_le_mul_of_nonneg_right hg.1 hw.1
  have hprod₂ := Int.mul_le_mul_of_nonneg_right hg.2 hw.1
  have hstep : min 0 g ≤ step ∧ step ≤ max 0 g := scaled_between g _ hw
  have evalRange : -16383 ≤ old ∧ old ≤ 16383 := by dsimp [old, correct, clamp]; omega
  have signed : FitsI32 (raw + entry) ∧ FitsI32 (score - old) ∧
      FitsI32 (g * weight depth) ∧ FitsI32 (entry + step) := by
    unfold FitsI32
    omega
  have gapDir : min 0 (score - old) ≤ g ∧ g ≤ max 0 (score - old) := by
    change min 0 (score - old) ≤ clamp _ _ (score - old) ∧ _
    dsimp [g, gap, old, clamp]
    omega
  unfold update
  split
  · rename_i takes
    change showsGap bound old score = true at takes
    have direction : min old target ≤ min old score ∧ max old score ≤ max old target := by
      cases bound <;> simp_all [showsGap, Sound] <;> omega
    let next := clamp (-2048) 2048 (entry + step)
    have bounded : -2048 ≤ next ∧ next ≤ 2048 := by dsimp [next, clamp]; omega
    have clipped : min 0 step ≤ next - entry ∧ next - entry ≤ max 0 step := by
      dsimp [next, clamp]
      omega
    have moved := correct_moves raw entry next
    change min 0 (next - entry) ≤ correct raw next - old ∧ _ at moved
    exact ⟨bounded.1, bounded.2, by change min old target ≤ correct raw next; omega,
      by change correct raw next ≤ max old target; omega, signed⟩
  · exact ⟨he₁, he₂, by omega, by omega, signed⟩

def FullStepProgress : Prop :=
  ∀ (raw entry score : Int) (bound : Bound) (depth : Depth),
    -2048 < entry → entry < 2048 → -16383 < raw + entry → raw + entry < 16383 →
    16 ≤ depth.toNat → showsGap bound (correct raw entry) score = true →
    let old := correct raw entry
    let next := correct raw (update raw entry score bound depth)
    (old + 4 ≤ score → old < next) ∧ (score + 4 ≤ old → next < old)

private theorem fullStepProgress_proven : FullStepProgress := by
  intro raw entry score bound depth _ _ _ _ hd eligible
  have full : weight depth = 16 := by unfold weight; omega
  dsimp only
  rw [update, ite_eq_left eligible]
  simp only [change, full, gap, correct, clamp, Int.tdiv_cases]
  constructor <;> intro h <;> split <;> omega

def UpdateContract : Prop := BoundedUpdate ∧ FullStepProgress

theorem updateContract_proven : UpdateContract :=
  ⟨boundedUpdate_proven, fullStepProgress_proven⟩

end NeuralReversi.CorrectionHistory

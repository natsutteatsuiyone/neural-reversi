module
public import NeuralReversi.EndgameProbCutModel
public import NeuralReversi.ShallowBoardTree

@[expose] public section

namespace NeuralReversi.EndgameProbCut

open Aspiration (SoundResult Window)
open PvsMoveLoop (scoreInf scoreMax)

def RoundingContract : Prop :=
  (∀ units : Int, 0 ≤ units → 0 ≤ roundedUnits true units) ∧
  (∀ screen verification : Int, screen ≤ verification →
    roundedUnits false screen ≤ roundedUnits true verification)

set_option exponentiation.threshold 4096 in
theorem roundingContract_proven : RoundingContract := by
  let q : Int := EndgameTiming.quantum
  have positive : 0 < q := by decide
  constructor
  · intro units nonneg
    have quotient : (-units) / q ≤ 0 := by
      simpa using Int.ediv_le_ediv positive (show -units ≤ 0 by omega)
    change 0 ≤ satI32 (-((-units) / q))
    unfold satI32 clamp
    omega
  · intro screen verification ordered
    let floor := screen / q
    have product := Int.ediv_mul_le screen (Int.ne_of_gt positive)
    have upperBound : (-verification) / q ≤ -floor := by
      apply (Int.ediv_le_iff_le_mul positive).2
      rw [Int.neg_mul]
      change floor * q ≤ screen at product
      omega
    change satI32 floor ≤ satI32 (-((-verification) / q))
    unfold satI32 clamp
    omega

def ThresholdContract : Prop :=
  ∀ (beta verification screen : Int), -scoreInf < beta → beta ≤ scoreInf →
    0 ≤ verification →
    let built := thresholds beta verification screen
    built.verification = min scoreMax (beta + verification) ∧
    built.screen = clamp (-scoreMax) scoreMax (beta + screen) ∧
    (-scoreMax ≤ built.screen ∧ built.screen ≤ scoreMax) ∧
    (built.verification < scoreMax →
      beta ≤ built.verification ∧ Window (built.verification - 1) built.verification) ∧
    (-scoreMax ≤ beta → screen ≤ verification → built.screen ≤ built.verification)

theorem thresholdContract_proven : ThresholdContract := by
  intro beta verification screen low high positive built
  dsimp [built, thresholds, satI32, clamp, scoreInf, scoreMax] at *
  refine ⟨by omega, by omega, ⟨by omega, by omega⟩, fun fits => ⟨by omega, ⟨by omega, by omega, by omega⟩⟩,
    fun betaRange ordered => by omega⟩

def verificationWindow (evaluation : Int) (threshold : Thresholds) : Option (Int × Int) :=
  if scoreMax ≤ threshold.verification ∨ evaluation < threshold.screen then none
  else some (threshold.verification - 1, threshold.verification)

def attempt (pv : Bool) (depth selectivity : Nat) (beta evaluation : Int)
    (threshold : Thresholds) (verify : Int → Int → Int) : Option Int :=
  if pv = true ∨ depth ≤ 12 ∨ selectivity = 5 then none
  else do
    let (a, b) ← verificationWindow evaluation threshold
    if b ≤ verify a b then some beta else none

def VerificationContract : Prop :=
  ∀ (verify : Int → Int → Int) (shallow : Int) (pv : Bool) (depth selectivity : Nat)
    (beta evaluation verification screen result : Int),
    selectivity ∈ EndgameProgress.ladder → -scoreInf < beta → beta ≤ scoreInf → 0 ≤ verification →
    (∀ a b, Window a b → SoundResult shallow a b (verify a b)) →
    let threshold := thresholds beta verification screen
    (attempt pv depth selectivity beta evaluation threshold verify = some result ↔
      pv = false ∧ 13 ≤ depth ∧ (1 ≤ selectivity ∧ selectivity ≤ 4) ∧
      threshold.verification < scoreMax ∧ threshold.screen ≤ evaluation ∧
      threshold.verification ≤ shallow ∧ result = beta)

theorem verificationContract_proven : VerificationContract := by
  intro verify shallow pv depth selectivity beta evaluation verification screen result listed low high nonnegative verified threshold
  have levels : 1 ≤ selectivity ∧ selectivity ≤ 5 := by
    simp [EndgameProgress.ladder] at listed
    omega
  have valid := (thresholdContract_proven beta verification screen low high nonnegative).2.2.2.1
  have enabled : selectivity ≠ 5 ↔ 1 ≤ selectivity ∧ selectivity ≤ 4 := by omega
  by_cases fits : threshold.verification < scoreMax
  · have sound := verified _ _ (valid fits).2
    have pass : threshold.verification ≤ verify (threshold.verification - 1) threshold.verification ↔
        threshold.verification ≤ shallow := ShallowSearch.reachesThreshold _ _ _ sound
    simp only [attempt, verificationWindow, Int.not_le_of_gt fits, false_or, fits, true_and]
    by_cases screened : threshold.screen ≤ evaluation
    · have ge := Int.not_lt_of_ge screened
      cases pv <;> simp [ge, screened, pass, enabled, show ¬ depth ≤ 12 ↔ 13 ≤ depth by omega,
        not_or, and_assoc, eq_comm]
    · simp [show evaluation < threshold.screen by omega, screened]
  · simp [attempt, verificationWindow, Int.le_of_not_gt fits, fits]

def BoardVerificationContract : Prop :=
  ∀ (mc : Bitboard.Moves.X86Config) (fc : Flip.Dispatch.X86Config)
    (evaluate : ShallowBoardTree.LeafEval) (board : PvsBoardTree.Position) (verify : Int → Int → Int)
    (pv : Bool) (depth selectivity : Nat) (beta evaluation verification screen result : Int),
    selectivity ∈ EndgameProgress.ladder → -scoreInf < beta → beta ≤ scoreInf → 0 ≤ verification →
    (∀ a b, Window a b → ShallowBoardTree.Completed mc fc evaluate 2 board a b (verify a b)) →
    let threshold := thresholds beta verification screen
    let shallow := ShallowBoardTree.value mc fc evaluate 2 board
    (attempt pv depth selectivity beta evaluation threshold verify = some result ↔
      pv = false ∧ 13 ≤ depth ∧ (1 ≤ selectivity ∧ selectivity ≤ 4) ∧
      threshold.verification < scoreMax ∧ threshold.screen ≤ evaluation ∧
      threshold.verification ≤ shallow ∧ result = beta) ∧
    (∀ full, attempt pv depth selectivity beta evaluation threshold verify = some result →
      shallow - verification ≤ full → result ≤ full)

theorem boardVerificationContract_proven : BoardVerificationContract := by
  intro mc fc evaluate board verify pv depth selectivity beta evaluation verification screen result
    listed low high nonnegative searched threshold shallow
  have sound a b (window : Window a b) : SoundResult shallow a b (verify a b) :=
    ShallowBoardTree.replayContract_proven mc fc evaluate 2 board a b (verify a b) (searched a b window) window
  have checked := verificationContract_proven verify shallow pv depth selectivity beta evaluation verification screen
    result listed low high nonnegative sound
  refine ⟨checked, fun full cut margin => ?_⟩
  obtain ⟨_, _, _, available, _, pass, returned⟩ := checked.mp cut
  have built : threshold.verification = min scoreMax (beta + verification) :=
    (thresholdContract_proven beta verification screen low high nonnegative).1
  change threshold.verification < scoreMax at available
  change threshold.verification ≤ shallow at pass
  omega

example : (thresholds 0 scoreMax 0).verification = scoreMax ∧
    verificationWindow 0 ⟨2, 0⟩ = some (1, 2) ∧
    attempt false 13 5 0 0 ⟨0, 0⟩ (fun _ _ => 0) = none := by decide

end NeuralReversi.EndgameProbCut

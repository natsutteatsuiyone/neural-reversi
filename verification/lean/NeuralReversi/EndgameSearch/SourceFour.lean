import NeuralReversi.EndgameSearch
import NeuralReversi.ScoreRefresh.Properties
import NeuralReversi.Solve34.Target

namespace NeuralReversi.EndgameSearch

open PvsBoardTree (Position legalSquares minimax)
open EndgameTreeCache (Store Table discValue soundHistory)
open Bitboard.Moves (X86Config)
open Solve34 (EmptyBoard)
open Flip.Dispatch (TargetConfig)
open ScoreRefresh (DiscScore Score fromDiscDiff toDiscDiff)

abbrev FourBoards := (b : Position) → (Board.emptyCount b.val).toNat = 4 →
  {ordered : EmptyBoard 4 // ordered.asBoard = b.val}

private theorem cast_asBoard {n m : Nat} (eq : n = m) (b : EmptyBoard n) :
    (eq ▸ b : EmptyBoard m).asBoard = b.asBoard := by
  cases eq
  rfl

def squareOrder : FourBoards := fun b count =>
  ⟨count ▸ EmptyBoard.ofBoard b.val b.property, cast_asBoard count _⟩

def withSourceFour (target : TargetConfig) (bmi2 : Bool) (opts : Options mc) (boards : FourBoards) :
    Options mc :=
  { opts with four := some fun b alpha =>
      if four : (Board.emptyCount b.val).toNat = 4 then
        (Solve34.target4Run target bmi2 (boards b four).val (BitVec.ofInt 32 alpha)).score.toInt
      else 0 }

theorem withSourceFour_correct (target : TargetConfig) (bmi2 : Bool)
    (mc : X86Config) (fc : FlipConfig) (opts : Options mc) (boards : FourBoards)
    (moves : ∀ b table sq, sq ∈ opts.order b table ↔ sq ∈ legalSquares mc b.val) :
    CorrectOptions mc fc (withSourceFour target bmi2 opts boards) := by
  refine ⟨moves, ?_⟩
  intro f selected b alpha four safe
  have word : (BitVec.ofInt 32 alpha).toInt = alpha :=
    BitVec.toInt_ofInt_eq_self (by decide) (by have := safe.1; omega) (by have := safe.2; omega)
  have supported : Solve34.WindowSafe (BitVec.ofInt 32 alpha) := by
    unfold Solve34.WindowSafe
    rw [word]
    exact safe
  have sound := (Solve34.targetTreeContract_proven target bmi2 mc fc
    (BitVec.ofInt 32 alpha) supported).2.2 (boards b four).val
  rw [(boards b four).property, word] at sound
  have eq := Option.some.inj selected
  rw [← eq]
  simpa only [four, ↓reduceDIte] using sound

def ThresholdArithmetic : Prop :=
  ∀ (checked : Bool) (alpha : DiscScore), SafeThreshold alpha.raw.toInt →
    let beta : DiscScore := ⟨alpha.raw + 1⟩
    let child : DiscScore := ⟨-beta.raw⟩
    DiscScore.addInt checked alpha 1 = .ok beta ∧ beta.raw.toInt = alpha.raw.toInt + 1 ∧
      DiscScore.negate checked beta = .ok child ∧ child.raw.toInt = -(alpha.raw.toInt + 1) ∧
      DiscScore.subInt checked beta 1 = .ok alpha ∧ SafeThreshold child.raw.toInt

theorem thresholdArithmetic_proven : ThresholdArithmetic := by
  intro checked alpha safe
  obtain ⟨lo, hi⟩ := safe
  have beta := Solve2.add_one_safe alpha.raw hi
  have child := Solve2.childAlpha_safe alpha.raw hi
  have addSafe : ¬ ScaledScore.outsideI32 (alpha.raw.toInt + (1#32).toInt) := by
    unfold ScaledScore.outsideI32
    change ¬ (alpha.raw.toInt + 1 < -2147483648 ∨ alpha.raw.toInt + 1 ≥ 2147483648)
    omega
  have negSafe : ¬ ScaledScore.outsideI32 (-(alpha.raw + 1).toInt) := by
    rw [beta]
    unfold ScaledScore.outsideI32
    omega
  have subSafe : ¬ ScaledScore.outsideI32 ((alpha.raw + 1).toInt - (1#32).toInt) := by
    rw [beta]
    unfold ScaledScore.outsideI32
    change ¬ (alpha.raw.toInt + 1 - 1 < -2147483648 ∨ alpha.raw.toInt + 1 - 1 ≥ 2147483648)
    omega
  have addOk : alpha.raw.saddOverflow (1 : ScoreRefresh.Word) = false := by
    rw [ScaledScore.add_overflow_spec]
    exact decide_eq_false addSafe
  have subOk : (alpha.raw + 1).ssubOverflow (1 : ScoreRefresh.Word) = false := by
    rw [ScaledScore.sub_overflow_spec]
    exact decide_eq_false subSafe
  refine ⟨?_, beta, ?_, child, ?_, ?_⟩
  · simp only [DiscScore.addInt, addOk, Bool.and_false, Bool.false_eq_true, ↓reduceIte]
  · simp only [DiscScore.negate, ScaledScore.neg_overflow_spec, decide_eq_false negSafe,
      Bool.and_false, Bool.false_eq_true, ↓reduceIte]
  · simp only [DiscScore.subInt, subOk, Bool.and_false, Bool.false_eq_true, ↓reduceIte]
    cases alpha
    simp [BitVec.add_sub_cancel]
  · change SafeThreshold (Solve2.childAlpha alpha.raw).toInt
    rw [child]
    exact ⟨by omega, by omega⟩

def SourceFourContract : Prop :=
  ∀ (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (opts : Options mc) (boards : FourBoards) (table : Table)
    (b : Position) (alpha beta : Score) (history : List Store) (checked : Bool),
    (∀ p tier sq, sq ∈ opts.order p tier ↔ sq ∈ legalSquares mc p.val) →
    soundHistory mc fc history →
    beta.raw.toInt = alpha.raw.toInt + 1 →
    let result := run mc fc (withSourceFour target bmi2 opts boards) table b (toDiscDiff alpha).raw.toInt history
    ∃ score, fromDiscDiff checked ⟨BitVec.ofInt 32 result.score⟩ = .ok score ∧
      Aspiration.SoundResult (minimax mc fc b) alpha.raw.toInt beta.raw.toInt score.raw.toInt ∧
      soundHistory mc fc result.history ∧ history <+: result.history

theorem sourceFourContract_proven : SourceFourContract := by
  intro target bmi2 mc fc opts boards table b alpha beta history checked moves sound window
  let settings := withSourceFour target bmi2 opts boards
  let threshold := (toDiscDiff alpha).raw.toInt
  let result := run mc fc settings table b threshold history
  have floored := ScoreRefresh.disc_floor alpha
  have thresholdSafe : SafeThreshold threshold := by
    have low := BitVec.le_toInt alpha.raw
    have high := BitVec.toInt_lt (x := alpha.raw)
    change SafeThreshold ((toDiscDiff alpha).raw.toInt)
    rw [floored]
    exact ⟨by simp at low; omega, by simp at high; omega⟩
  have bounds := contract_proven mc fc settings table b threshold history
    (withSourceFour_correct target bmi2 mc fc opts boards moves) sound (fun _ _ => thresholdSafe)
  change EndgameCache.SearchRange result.score ∧ Solve2.NWS threshold (discValue mc fc b.val) result.score ∧
    soundHistory mc fc result.history ∧ history <+: result.history at bounds
  obtain ⟨⟨lo, hi⟩, nws, after, grew⟩ := bounds
  let discs : DiscScore := ⟨BitVec.ofInt 32 result.score⟩
  have asInt : discs.raw.toInt = result.score :=
    BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  obtain ⟨score, cast, _, scaled, _⟩ := ScoreRefresh.bounded_roundtrip discs (by omega) (by omega)
  have converted : fromDiscDiff checked discs = .ok score := by
    have within : ¬ (discs.raw.toInt < -127 ∨ discs.raw.toInt > 127) := by omega
    simpa [fromDiscDiff, within] using cast
  have raw := EndgameTreeCache.rawWindow_sound mc fc b alpha.raw.toInt result.score ⟨lo, hi⟩
    (floored ▸ nws)
  rw [← window] at raw
  rw [asInt] at scaled
  exact ⟨score, converted, (scaled.trans (Int.mul_comm _ _)) ▸ raw, after, grew⟩

end NeuralReversi.EndgameSearch

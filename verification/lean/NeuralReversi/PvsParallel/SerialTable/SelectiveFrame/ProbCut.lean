module
public import NeuralReversi.EndgameProbCut
public import NeuralReversi.PvsParallel.SerialCalls.ScoreWords

@[expose] public section

namespace NeuralReversi.PvsParallel.SerialTable.SelectiveFrame

open TranspositionTable (Depth Selectivity)
open EndgameProbCut (roundedMargin nativeThresholds attempt)

structure Estimates where
  evaluation : BitVec 32
  verifyDepth2 : Int → Int → BitVec 32

def tailCoefficients : Array (Float × Float × Float) :=
  #[(0.3786,0.1317,0.3825),(-1.3215,0.2985,0.6360),(0.0170,0.3131,0.7175),(-0.4242,0.4874,0.9121)]

def mean : List Float :=
  [3.2996552433,-0.0583166555,-0.2489634608,-0.3459216837,0.2330285133,0.3175907433,0.5569823145,-0.6444854153]

def logSigma : List Float :=
  [0.9479918545,-0.0175477071,-0.0846528866,-0.0265623046,-0.3431613432,0.3472355880,-0.0296237147,-0.1353929169]

def dot (coeff features : List Float) : Float :=
  (coeff.zipWith (· * ·) features).foldl (· + ·) 0

/-- The fitted floating-point model of EndgameWindow::scaled_margins.
    The native nonnegative verification margin is encoded as a magnitude. -/
def margins (depth : Depth) (level : Selectivity) (cut : Bool) (beta eval : BitVec 32) : BitVec 64 × BitVec 64 :=
  let n := depth.toNat
  let bounded := min 30 (max 13 n)
  let betaDiscs := beta.toInt.toFloat / 256
  let evalDiscs := eval.toInt.toFloat / 256
  let d := min 2 (max (-2) ((evalDiscs-betaDiscs)/8))
  let u := betaDiscs/32
  let features := [1,Float.ofNat bounded,Float.ofNat (bounded%2),Float.ofNat cut.toNat,d,(d+0.375).abs,u,u*u]
  let (a,b,c) := tailCoefficients[level.val-1]!
  let tail := a + b*Float.ofNat n + c*Float.ofNat (n%2)
  let raw := tail*(dot logSigma features).exp - dot mean features
  let v := if 0 < raw then raw else 0
  let screen := v - (if cut then 17.730469 else 4.90625)
  ((v*256).toBits.toBitVec &&& 0x7fffffffffffffff#64,(screen*256).toBits.toBitVec)

def probCut (d : Depth) (level : Selectivity) (cut : Bool) (beta : ScoreRefresh.Score) (scores : Estimates) : Option Int :=
  if 13 ≤ d.toNat ∧ 1 ≤ level.val ∧ level.val < 5 then
    let bits := margins d level cut beta.raw scores.evaluation
    attempt false d.toNat level.val beta.raw.toInt scores.evaluation.toInt
      (nativeThresholds beta.raw.toInt bits.1 bits.2) (fun a z => (scores.verifyDepth2 a z).toInt)
  else none

private theorem margin_nonnegative (d : Depth) (s : Selectivity) (cut : Bool) (z eval : BitVec 32) :
    0 ≤ roundedMargin true (margins d s cut z eval).1 := by
  have msb : (margins d s cut z eval).1.msb = false := by
    dsimp only [margins]
    rw [BitVec.msb_and]
    exact Bool.and_false _
  unfold roundedMargin
  dsimp only
  cases h : EndgameTiming.decode ((margins d s cut z eval).1 &&& 0x7fffffffffffffff#64) with
  | none => exact Int.le_refl 0
  | some r =>
    cases r <;> simp only [msb, Bool.false_eq_true, ite_false]
    · exact EndgameProbCut.roundingContract_proven.1 _ (Int.natCast_nonneg _)
    all_goals decide

theorem probCut_finite {d : Depth} {level : Selectivity} {beta : ScoreRefresh.Score} {scores : Estimates}
    (win : Aspiration.Window a beta.raw.toInt) (hit : probCut d level cut beta scores = some v) :
    v = beta.raw.toInt ∧ SerialCalls.ScoreWords.Finite v := by
  unfold probCut at hit
  split at hit <;> try contradiction
  let bits := margins d level cut beta.raw scores.evaluation
  let threshold := nativeThresholds beta.raw.toInt bits.1 bits.2
  change attempt _ _ _ _ _ threshold _ = some v at hit
  unfold attempt at hit
  split at hit <;> try contradiction
  obtain ⟨⟨a',z'⟩,screen,result⟩ := Option.bind_eq_some_iff.mp hit
  simp only [EndgameProbCut.verificationWindow] at screen
  split at screen <;> try contradiction
  rename_i screened
  dsimp only at result
  split at result <;> try contradiction
  cases result
  obtain ⟨lo,ord,hi⟩ := win
  have nonneg := margin_nonnegative d level cut beta.raw scores.evaluation
  have property := EndgameProbCut.thresholdContract_proven beta.raw.toInt (roundedMargin true bits.1)
    (roundedMargin false bits.2) (by change -32767 < _; omega) hi nonneg
  have within : threshold.verification < PvsMoveLoop.scoreMax := by omega
  have lower := (property.2.2.2.1 within).1
  change beta.raw.toInt ≤ threshold.verification at lower
  change threshold.verification < 16384 at within
  exact ⟨rfl,by unfold SerialCalls.ScoreWords.Finite; omega⟩

end NeuralReversi.PvsParallel.SerialTable.SelectiveFrame

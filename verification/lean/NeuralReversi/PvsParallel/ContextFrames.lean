import NeuralReversi.PvsParallel.ContextPathMir
import NeuralReversi.PvsParallel.NativeFrames

namespace NeuralReversi.PvsParallel.ContextPathMir

open PvsBoardTree (Position)
open EndgameSearch.CacheHistory (Input)
open ScoreRefresh (Score)

def args (b : Position) (d : ExactDepth.Depth) (alpha beta : Score) (pv cut : Bool)
    (line : Option Nat) : NativeFrames.Args :=
  ⟨b.val, d, alpha, beta, true, Cutoffs.exactSelectivity, pv, cut, line⟩

def At.cacheInput (path : At mc fc root b d s) (fits : (Board.emptyCount root.val).toNat ≤ 60)
    (alpha beta : Score) (attempts : List TranspositionTable.ProbeRetry.RawAttempt)
    (polls : Threading.SplitCutoff.Polling) (parallel active aborted checked : Bool)
    (target : Flip.Dispatch.TargetConfig) (bmi2 paired : Bool) : Input :=
  ⟨b, leafState s, (path.frameInputs fits).1, alpha, beta, d, Cutoffs.exactSelectivity, attempts,
    polls, parallel, active, aborted, checked, target, bmi2, paired⟩

theorem At.rowOpened (path : At mc fc root b d s) (fits : (Board.emptyCount root.val).toNat ≤ 60)
    (alpha beta value : Score) (pv cut : Bool) (line : Option Nat) (size worker opening parent step : Nat)
    (window : Aspiration.Window alpha.raw.toInt beta.raw.toInt)
    (narrow : pv = false → beta.raw.toInt = alpha.raw.toInt + 1) :
    ∃ row, (args b d alpha beta pv cut line).row size worker opening parent step value = some row ∧
      row.frame.position = b ∧ row.depth = d := by
  have full := (path.frameInputs fits).2.1
  have wf : b.val.player &&& b.val.opponent = 0 := b.property
  have good : d = Board.emptyCount b.val ∧ -32767 ≤ alpha.raw.toInt ∧ alpha.raw.toInt < beta.raw.toInt ∧
      beta.raw.toInt ≤ 32767 ∧ (pv = false → beta.raw.toInt = alpha.raw.toInt + 1) :=
    ⟨full, window.1, window.2, window.3, narrow⟩
  unfold args NativeFrames.Args.row
  dsimp only
  rw [dite_eq_left wf, dite_eq_left good]
  refine ⟨_, rfl, ?_, rfl⟩
  apply Subtype.ext
  cases b with
  | mk board valid =>
    cases board
    simp [FrameReturns.Frame.position, TraceCheck.FrameReturn.board]

end NeuralReversi.PvsParallel.ContextPathMir

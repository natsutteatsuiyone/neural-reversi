module
public import NeuralReversi.PvsParallel.RootUpdateMir.Program
public import NeuralReversi.PvsParallel.RootSuffix.Writes

@[expose] public section

namespace NeuralReversi.PvsParallel.RootUpdateMir

open RootReplay (Snapshot)
open Threading.SplitWindow (marksCutoff)
open RootOrder (Report)
open Bitboard (Square)

def decision (first : Bool) (alpha beta best score : Int) : Changes :=
  changes first (decide (alpha < score)) (decide (best < score)) (decide (score < beta))
    (decide (PvsMoveLoop.scoreMax ≤ score))

def after (before : Snapshot) (square : Square) (score : Int) (d : Changes) : Snapshot :=
  ⟨⟨if d.raisesAlpha then score else before.state.alpha,
     if d.raisesBest then score else before.state.best⟩,
   square :: before.seen,
   RootOrder.writeScore before.scores square (if d.visible then score else -PvsMoveLoop.scoreInf)⟩

theorem agrees (before : Snapshot) (square : Square) (beta score : Int) :
    let d := decision before.seen.isEmpty before.state.alpha beta before.state.best score
    after before square score d = before.record beta ⟨square, before.state.alpha, score⟩ ∧
      d.cuts = marksCutoff true beta before.state score ∧
      d.takesMove = decide (before.state.best < score ∧ before.state.alpha < score) ∧
      d.visible = (before.seen.isEmpty || decide (before.state.alpha < score)) := by
  intro d
  by_cases hAlpha : before.state.alpha < score <;> by_cases hBest : before.state.best < score <;>
    by_cases hBeta : score < beta <;> by_cases hMax : PvsMoveLoop.scoreMax ≤ score <;>
    cases f : before.seen.isEmpty <;>
    simp [d, decision, changes, after, RootReplay.Snapshot.record, RootOrder.visibleScore,
      Threading.SplitWindow.publish, Threading.SplitWindow.raisesAlpha, marksCutoff,
      Int.max_def, hAlpha, hBest, hBeta, hMax, f] <;> omega

theorem recordWrites (writes : RootSuffix.Writes initial idx alpha beta rows before)
    (sq : Square) (score : Int) (pv : List Square) (candidate : sq ∈ (rows.drop idx).map RootMoveStorage.Row.square) :
    let d := decision before.seen.isEmpty before.state.alpha beta before.state.best score
    RootSuffix.Writes initial idx alpha beta
      (RootSuffix.writeFirst sq (fun row => RootSuffix.updateRow row score d.visible pv) rows)
      (after before sq score d) := by
  intro d
  obtain ⟨same, _, _, visible⟩ := agrees before sq beta score
  rw [same]
  exact .record writes ⟨sq, before.state.alpha, score⟩ d.visible pv candidate visible

end NeuralReversi.PvsParallel.RootUpdateMir

module
public import NeuralReversi.Board.MovesProperties
public import NeuralReversi.ScoreRefresh.BoardBridge
public import NeuralReversi.ScoreRefresh.Properties
public import NeuralReversi.Solve34.Control

@[expose] public section

namespace NeuralReversi.EndgamePass
open Board (Board)
open ScoreRefresh (DiscScore)
open Bitboard.Moves (LegalMove X86Config)

def noMove (mc : X86Config) (checked : Bool) (b : Board) (alpha : DiscScore)
    (search : Board → DiscScore → DiscScore) : Option (Except String DiscScore) :=
  if Board.getMoves mc b ≠ 0 then none else some do
    let beta ← DiscScore.addInt checked alpha 1
    let next := Board.switchPlayers b
    if Board.getMoves mc next = 0 then
      ScoreRefresh.BoardBridge.solveDiscDiff checked b (Board.emptyCount b)
    else
      let childAlpha ← DiscScore.negate checked beta
      DiscScore.negate checked (search next childAlpha)

def NoMoveContract : Prop :=
  ∀ mc checked (b : Board) (alpha : DiscScore) (search : Board → DiscScore → DiscScore) (childValue : Int),
    Board.wellFormed b → (∀ i, ¬ LegalMove b.player b.opponent i) →
    -65 ≤ alpha.raw.toInt → alpha.raw.toInt ≤ 64 →
    (∀ (window : DiscScore), -65 ≤ window.raw.toInt → window.raw.toInt ≤ 64 →
      (∃ i, LegalMove b.opponent b.player i) →
      let child := search (Board.switchPlayers b) window
      (-64 ≤ child.raw.toInt ∧ child.raw.toInt ≤ 64 ∧ Solve2.NWS window.raw.toInt childValue child.raw.toInt)) →
    ∃ score, noMove mc checked b alpha search = some (.ok score) ∧
      -64 ≤ score.raw.toInt ∧ score.raw.toInt ≤ 64 ∧
      ((∀ i, ¬ LegalMove b.opponent b.player i) → score.raw.toInt = Board.AwardedScore b) ∧
      ((∃ i, LegalMove b.opponent b.player i) → Solve2.NWS alpha.raw.toInt (-childValue) score.raw.toInt)

private theorem negate_ok (checked : Bool) (s : DiscScore)
    (lo : -65 ≤ s.raw.toInt) (hi : s.raw.toInt ≤ 65) :
    DiscScore.negate checked s = .ok ⟨-s.raw⟩ := by
  have safe : ¬ ScaledScore.outsideI32 (-s.raw.toInt) := by unfold ScaledScore.outsideI32; omega
  simp [DiscScore.negate, ScaledScore.neg_overflow_spec, safe]

theorem noMoveContract_proven : NoMoveContract := by
  intro mc checked b alpha search childValue wf stuck lo hi childSound
  let beta : DiscScore := ⟨alpha.raw + 1⟩
  have betaValue : beta.raw.toInt = alpha.raw.toInt + 1 := Solve2.add_one_safe _ (by omega)
  have addSafe : ¬ ScaledScore.outsideI32 (alpha.raw.toInt + 1) := by
    change ¬ (alpha.raw.toInt + 1 < -2147483648 ∨ 2147483647 < alpha.raw.toInt + 1)
    omega
  have add : DiscScore.addInt checked alpha 1 = .ok beta := by
    simp [DiscScore.addInt, ScaledScore.add_overflow_spec, addSafe, beta]
  have cur := (Board.moves_zero_iff mc b).mpr stuck
  let next := Board.switchPlayers b
  have run : noMove mc checked b alpha search = some
      (if Board.getMoves mc next = 0 then ScoreRefresh.BoardBridge.solveDiscDiff checked b (Board.emptyCount b)
       else do
         let window ← DiscScore.negate checked beta
         DiscScore.negate checked (search next window)) := by
    simp only [noMove, cur, ne_eq, not_true_eq_false, ↓reduceIte, add]
    rfl
  by_cases terminal : Board.getMoves mc next = 0
  · have opponentStuck := (Board.moves_zero_iff mc next).mp terminal
    obtain ⟨score, release, debug, awarded, lower, upper⟩ := ScoreRefresh.BoardBridge.solveDiscDiff_awarded b wf
    refine ⟨score, ?_, lower, upper, fun _ => awarded, ?_⟩
    · rw [run, ite_eq_left terminal]
      exact congrArg some (by cases checked <;> assumption)
    · rintro ⟨i, legal⟩
      exact (opponentStuck i legal).elim
  · have mobile : ∃ i, LegalMove b.opponent b.player i := by
      apply Classical.byContradiction
      intro absent
      exact terminal ((Board.moves_zero_iff mc next).mpr (fun i move => absent ⟨i, move⟩))
    let window : DiscScore := ⟨-beta.raw⟩
    have windowValue : window.raw.toInt = -(alpha.raw.toInt + 1) := by
      exact (Solve2.neg_safe _ (by omega)).trans (congrArg Neg.neg betaValue)
    let child := search next window
    obtain ⟨childLo, childHi, sound⟩ := childSound window (by omega) (by omega) mobile
    change -64 ≤ child.raw.toInt at childLo
    change child.raw.toInt ≤ 64 at childHi
    have call : DiscScore.negate checked beta = .ok window := negate_ok checked beta (by omega) (by omega)
    have ret := negate_ok checked child (by omega) (by omega)
    have negValue : (-child.raw).toInt = -child.raw.toInt := Solve2.neg_safe _ (by omega)
    refine ⟨⟨-child.raw⟩, ?_, by dsimp; rw [negValue]; omega, by dsimp; rw [negValue]; omega, ?_, ?_⟩
    · rw [run, ite_eq_right terminal, call]
      exact congrArg some ret
    · intro noOpponent
      obtain ⟨i, legal⟩ := mobile
      exact (noOpponent i legal).elim
    · intro _
      change Solve2.NWS _ _ (-child.raw).toInt
      rw [negValue]
      apply Solve34.negate_nws
      rwa [← windowValue]

example : Solve2.NWS 0 (-2) 0 ∧ ¬ Solve2.NWS (-1) (-2) 0 ∧ ¬ Solve2.NWS 0 2 0 := by
  unfold Solve2.NWS
  decide

end NeuralReversi.EndgamePass

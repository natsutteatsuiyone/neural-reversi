import NeuralReversi.PvsBoardTree
import NeuralReversi.Stability.BoardRules
import NeuralReversi.Stability.Cutoff

namespace NeuralReversi.EndgameStability

open Stability.Semantics
open Stability.Cutoff (countColor countEmpty terminalScore popcount stabilityCutoff DiscScore)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position pass play legalSquares minimax)
private abbrev FlipConfig := Flip.Dispatch.X86Config

private theorem count_bits (bits : Bitboard.Bits) :
    (Bitboard.count bits).toNat = popcount bits := by
  have indices : (List.finRange 64).map Fin.val = List.range 64 := by decide
  rw [(Bitboard.populationCountContract_proven bits).1, ← indices, List.countP_map]
  rfl

private theorem awarded_eq_terminal (b : Position) (c : Color) :
    Board.AwardedScore b.val = terminalScore (stateAt b.val c).board c := by
  have repr := stateAt_represents b.val c b.property
  have own : (Board.playerCount b.val).toNat = countColor (stateAt b.val c).board c := by
    unfold Board.playerCount countColor
    rw [count_bits]
    apply List.countP_congr
    intro sq _
    simpa only [decide_eq_true_eq, Marks, boardIndex_square] using repr.1 (boardSquare sq)
  have opp : (Board.opponentCount b.val).toNat = countColor (stateAt b.val c).board (Opposite c) := by
    unfold Board.opponentCount countColor
    rw [count_bits]
    apply List.countP_congr
    intro sq _
    simpa only [decide_eq_true_eq, Marks, boardIndex_square] using repr.2 (boardSquare sq)
  have empty : (Board.emptyCount b.val).toNat = countEmpty (stateAt b.val c).board := by
    unfold Board.emptyCount countEmpty
    rw [count_bits]
    apply List.countP_congr
    intro sq _
    simp only [decide_eq_true_eq, Board.emptySquares, BitVec.getLsbD_not,
      BitVec.getLsbD_or, sq.isLt, decide_true, Bool.true_and, stateAt, decodeBoard, boardIndex_square]
    unfold decodeColor
    cases b.val.player.getLsbD sq.val <;> cases b.val.opponent.getLsbD sq.val <;> simp
  unfold Board.AwardedScore terminalScore
  rw [own, opp, empty]
  dsimp only
  repeat' split
  all_goals omega

private theorem terminal_opposite (b : Stability.Semantics.Board) (c : Color) :
    terminalScore b (Opposite c) = -terminalScore b c := by
  have twice : Opposite (Opposite c) = c := by cases c <;> rfl
  unfold terminalScore
  simp only [twice]
  repeat' split
  all_goals omega

private theorem play_step (mc : X86Config) (fc : FlipConfig) (b : Position) (c : Color)
    (sq : PvsBoardTree.GeneratedSquare mc b.val) :
    Step (stateAt b.val c) (stateAt (play mc fc b sq).val.val (Opposite c)) := by
  have legal := (Board.getMoves_exact mc b.val sq.val.val).mp sq.property
  change Step (stateAt b.val c) (stateAt
    (Board.place b.val (Flip.Dispatch.resultValue fc sq.val b.val.player b.val.opponent) sq.val)
    (Opposite c))
  rw [Flip.Dispatch.value_disjoint_scalar fc sq.val _ _ b.property,
    stateAt_place _ _ _ b.property legal]
  exact .move _ _ ((legal_iff_relative b.val c sq.val b.property).mpr legal)

private theorem pass_step (mc : X86Config) (b : Position) (c : Color)
    (forced : Board.ForcedPass mc b.val) :
    Step (stateAt b.val c) (stateAt (pass b).val (Opposite c)) := by
  have switch := stateAt_switch b.val c b.property
  change Step (stateAt b.val c) (stateAt (Board.switchPlayers b.val) (Opposite c))
  rw [switch]
  obtain ⟨stuck, available⟩ := (Board.forcedPass_exact mc b.val).mp forced
  apply Step.pass
  · intro a legal
    have atSquare : boardSquare (boardIndex a) = a := boardSquare_index a
    exact stuck _ ((legal_iff_relative b.val c (boardIndex a) b.property).mp (by rwa [atSquare]))
  · obtain ⟨i, legal⟩ := available
    let sq : Bitboard.Square := ⟨i, legal.1⟩
    have relative := (legal_iff_relative (pass b).val (Opposite c) sq (pass b).property).mpr legal
    refine ⟨boardSquare sq, ?_⟩
    change LegalMove (stateAt b.val c).board (Opposite c) (boardSquare sq)
    simpa only [pass, switch] using relative

private def inPerspective (root turn : Color) (v : Int) : Int :=
  if turn = root then v else -v

private theorem inPerspective_negate (root turn : Color) (v : Int) :
    inPerspective root turn (-v) = inPerspective root (Opposite turn) v := by
  cases root <;> cases turn <;> simp [inPerspective, Opposite]

private theorem bound_of_reachable_payoffs (mc : X86Config) (fc : FlipConfig)
    (initial : State) (root : Color) (limit : Int) (lo : -PvsMoveLoop.scoreInf ≤ limit)
    (bounded : ∀ s, Reach initial s → 256 * terminalScore s.board root ≤ limit)
    (b : Position) (c : Color) (reached : Reach initial (stateAt b.val c)) :
    inPerspective root c (minimax mc fc b) ≤ limit := by
  cases moves : legalSquares mc b.val with
  | nil =>
    cases other : Board.hasLegalMoves mc (pass b).val
    · rw [PvsBoardTree.terminal_value mc fc b moves other, awarded_eq_terminal b c]
      have payoff := bounded (stateAt b.val c) reached
      by_cases same : c = root
      · simpa [inPerspective, same] using payoff
      · have opposite : root = Opposite c := by cases c <;> cases root <;> simp_all [Opposite]
        rw [inPerspective, ite_eq_right same]
        rw [opposite, terminal_opposite] at payoff
        omega
    · have forced : Board.ForcedPass mc b.val := ⟨PvsBoardTree.stuck_of_nil moves, other⟩
      rw [PvsBoardTree.minimax_pass mc fc b moves other, inPerspective_negate]
      exact bound_of_reachable_payoffs mc fc initial root limit lo bounded (pass b) (Opposite c)
        (.tail reached (pass_step mc b c forced))
  | cons first rest =>
    have childBound (sq : PvsBoardTree.GeneratedSquare mc b.val) :
        inPerspective root c (-minimax mc fc (play mc fc b sq).val) ≤ limit := by
      rw [inPerspective_negate]
      exact bound_of_reachable_payoffs mc fc initial root limit lo bounded
        (play mc fc b sq).val (Opposite c) (.tail reached (play_step mc fc b c sq))
    rw [PvsBoardTree.minimax_moves mc fc b first rest moves]
    by_cases same : c = root
    · simp only [inPerspective, same, ↓reduceIte] at childBound ⊢
      exact PvsMoveLoop.bestValue_le _ _ _ _ lo (fun sq _ => childBound sq)
    · simp only [inPerspective, same, ↓reduceIte] at childBound ⊢
      have firstLe := PvsMoveLoop.bestValue_ge_move (fun sq => -minimax mc fc (play mc fc b sq).val)
        (first :: rest) (-PvsMoveLoop.scoreInf) (move := first) (by simp)
      have := childBound first
      omega
termination_by Board.playRank mc b.val
decreasing_by
  · exact Board.legalCheckedStep_strict mc (.pass forced) b.property
  · exact (play mc fc b sq).property

def tryCutoff (b : Board.Board) (n : Nat) (alpha : DiscScore) : Option DiscScore :=
  if valid : n < 64 then stabilityCutoff b.player b.opponent ⟨n, valid⟩ alpha else none

def CutoffContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (b : Position) (n : Nat) (alpha : DiscScore),
    (64 ≤ n → tryCutoff b.val n alpha = none) ∧
    ∀ result, tryCutoff b.val n alpha = some result →
      -64 ≤ result.value ∧ result.value ≤ 64 ∧
      minimax mc fc b ≤ 256 * result.value ∧ result.value ≤ alpha.value

theorem cutoffContract_proven : CutoffContract := by
  intro mc fc b n alpha
  refine ⟨fun outside => by simp [tryCutoff, show ¬ n < 64 by omega], ?_⟩
  intro result hit
  by_cases valid : n < 64
  · simp only [tryCutoff, valid, dite_true] at hit
    obtain ⟨score, threshold, _⟩ := Stability.Cutoff.stabilityCutoff_some_spec hit
    let s := stateAt b.val true
    have repr := stateAt_represents b.val true b.property
    have stableCount : popcount (getStableDiscs b.val.opponent b.val.player) ≤ 64 := by
      unfold popcount
      simpa using (List.countP_le_length (p := fun sq : Fin 64 =>
        (getStableDiscs b.val.opponent b.val.player).getLsbD sq.val) (l := List.finRange 64))
    refine ⟨by omega, by omega, ?_, threshold⟩
    have lo : -PvsMoveLoop.scoreInf ≤ 256 * result.value := by
      unfold PvsMoveLoop.scoreInf
      omega
    have bounded (t : State) (reached : Reach s t) : 256 * terminalScore t.board true ≤ 256 * result.value := by
      have le := Stability.Cutoff.terminalScore_le_of_stable repr reached
      omega
    simpa [inPerspective] using bound_of_reachable_payoffs mc fc s true
      (256 * result.value) lo bounded b true (.refl s)
  · simp [tryCutoff, valid] at hit

end NeuralReversi.EndgameStability

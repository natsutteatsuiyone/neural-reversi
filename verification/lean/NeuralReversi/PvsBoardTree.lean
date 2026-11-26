import NeuralReversi.PvsMoveLoop
import NeuralReversi.Board.Move.Documentation
import NeuralReversi.Bitboard.NeonMoves
import NeuralReversi.ScoreRefresh.BoardBridge

namespace NeuralReversi.PvsBoardTree

open Bitboard.Moves (X86Config LegalMove)
open Board (Board)
open PvsMoveLoop (Tree)
open Aspiration (Window SoundResult)

abbrev Position := {b : Board // Board.wellFormed b}

abbrev GeneratedSquare (mc : X86Config) (b : Board) :=
  {sq : Bitboard.Square // (Board.getMoves mc b).getLsbD sq.val = true}

def legalSquares (mc : X86Config) (b : Board) : List (GeneratedSquare mc b) :=
  (List.finRange 64).filterMap fun sq =>
    if h : (Board.getMoves mc b).getLsbD sq.val = true then some ⟨sq, h⟩ else none

theorem legalSquares_values (mc : X86Config) (b : Board) :
    (legalSquares mc b).map Subtype.val =
      (List.finRange 64).filter (fun sq => (Board.getMoves mc b).getLsbD sq.val) := by
  rw [legalSquares, List.map_filterMap, ← List.filterMap_eq_filter]
  congr 1
  funext sq
  by_cases h : (Board.getMoves mc b).getLsbD sq.val = true <;> simp [Option.guard, h]

theorem legalSquares_complete (mc : X86Config) (b : Board) (sq : Bitboard.Square) :
    (∃ move ∈ legalSquares mc b, move.val = sq) ↔ LegalMove b.player b.opponent sq.val := by
  constructor
  · rintro ⟨move, _, rfl⟩
    exact (Board.getMoves_exact mc b _).mp move.property
  · intro legal
    have h := (Board.getMoves_exact mc b _).mpr legal
    refine ⟨⟨sq, h⟩, ?_, rfl⟩
    apply List.mem_filterMap.mpr
    exact ⟨sq, by simp, by simp [h]⟩

theorem stuck_of_nil {mc : X86Config} {b : Board} (h : legalSquares mc b = []) :
    Board.hasLegalMoves mc b = false := by
  cases avail : Board.hasLegalMoves mc b
  · rfl
  · obtain ⟨i, legal⟩ := (Board.hasLegalMoves_exact mc b).mp avail
    obtain ⟨move, found, _⟩ := (legalSquares_complete mc b ⟨i, legal.1⟩).mpr legal
    rw [h] at found
    exact False.elim (List.not_mem_nil found)

def play (mc : X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    (sq : GeneratedSquare mc b.val) :
    {next : Position // Board.playRank mc next.val < Board.playRank mc b.val} := by
  let flipped := Flip.Dispatch.resultValue fc sq.val b.val.player b.val.opponent
  let next := Board.place b.val flipped sq.val
  have edge : Board.LegalCheckedStep mc b.val next :=
    .place ((Board.getMoves_exact mc b.val _).mp sq.property)
      (Board.precomputedMoveContract_proven mc fc true true b.val sq.val b.property sq.property).1
  exact ⟨⟨next, (Board.legalCheckedStep_progress mc edge b.property).1⟩,
    Board.legalCheckedStep_strict mc edge b.property⟩

theorem play_occupancy (mc : X86Config) (fc : Flip.Dispatch.X86Config) (board : Position)
    (square : GeneratedSquare mc board.val) :
    let next := (play mc fc board square).val
    next.val.player ||| next.val.opponent = (board.val.player ||| board.val.opponent) ||| Bitboard.squareMask square.val := by
  have accepted := (Board.precomputedMoveContract_proven mc fc true true
    board.val square.val board.property square.property).1
  rw [Board.makeMoveWithFlippedAny_valid] at accepted
  exact (Board.playableCheckedPlacementInvariant_proven _ _ _ _ board.property accepted).2.2.2

/-- The newly occupied square distinguishes children of the same parent even
    when searching them flips different sets of opposing discs. -/
theorem play_injective (mc : X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    {m n : GeneratedSquare mc b.val} (same : (play mc fc b m).val.val = (play mc fc b n).val.val) : m = n := by
  have occupied := congrArg (fun child : Board => (child.player ||| child.opponent).getLsbD m.val.val) same
  change ((play mc fc b m).val.val.player ||| _).getLsbD _ = ((play mc fc b n).val.val.player ||| _).getLsbD _ at occupied
  rw [play_occupancy,play_occupancy] at occupied
  have legal := (Board.getMoves_exact mc b.val _).mp m.property
  simp only [BitVec.getLsbD_or,Bitboard.squareMask_bit,legal.2.1,legal.2.2.1] at occupied
  apply Subtype.ext
  apply Fin.ext
  simpa using occupied

def TargetChildContract : Prop :=
  ∀ (target : Flip.Dispatch.TargetConfig) (mc : X86Config) (fc : Flip.Dispatch.X86Config)
    (b : Position) (sq : Bitboard.Square) (debugAssertions overflowChecks : Bool),
    let bits := Bitboard.Moves.legalMovesTarget target b.val.player b.val.opponent
    (bits.getLsbD sq.val = true ↔ ∃ choice ∈ legalSquares mc b.val, choice.val = sq) ∧
    ∀ choice ∈ legalSquares mc b.val,
      (do
        let flipped ← Flip.Dispatch.flipTarget target overflowChecks choice.val
          b.val.player b.val.opponent
        Board.makeMoveWithFlippedAny debugAssertions overflowChecks b.val flipped choice.val) =
      .ok (play mc fc b choice).val.val

theorem targetChildContract_proven : TargetChildContract := by
  intro target mc fc b sq debugAssertions overflowChecks
  have coordinates := (Bitboard.Moves.targetMoveGenerationContract_proven target
    b.val.player b.val.opponent sq.val b.property).1
  refine ⟨coordinates.trans (legalSquares_complete mc b.val sq).symm, ?_⟩
  intro choice _
  have result := Flip.Dispatch.flipTarget_eq_value_disjoint target overflowChecks choice.val
    b.val.player b.val.opponent b.property
  have same := Flip.Dispatch.value_disjoint_scalar fc choice.val b.val.player b.val.opponent b.property
  rw [← same] at result
  rw [result]
  exact (Board.precomputedMoveContract_proven mc fc debugAssertions overflowChecks
    b.val choice.val b.property choice.property).1

def pass (b : Position) : Position :=
  ⟨Board.switchPlayers b.val, Board.switchPlayers_wellFormed b.val b.property⟩

def terminalScore (b : Position) :
    {s : Int // -PvsMoveLoop.scoreMax ≤ s ∧ s ≤ PvsMoveLoop.scoreMax} := by
  refine ⟨256 * Board.AwardedScore b.val, ?_⟩
  obtain ⟨score, _, _, equal, lo, hi⟩ :=
    ScoreRefresh.BoardBridge.solveDiscDiff_awarded b.val b.property
  unfold PvsMoveLoop.scoreMax
  omega

def build (mc : X86Config) (fc : Flip.Dispatch.X86Config) (b : Position) : Tree :=
  match _hm : legalSquares mc b.val with
  | first :: rest =>
    .turn (build mc fc (play mc fc b first).val)
      (rest.map fun sq => build mc fc (play mc fc b sq).val)
  | [] =>
    let next := pass b
    if _mobile : Board.hasLegalMoves mc next.val = true then
      .pass (build mc fc next)
    else .leaf (terminalScore b)
termination_by Board.playRank mc b.val
decreasing_by
  · exact (play mc fc b first).property
  · exact (play mc fc b sq).property
  · exact Board.legalCheckedStep_strict mc (.pass ⟨stuck_of_nil _hm, _mobile⟩) b.property

theorem nil_of_stuck {mc : X86Config} {b : Board} (stuck : Board.hasLegalMoves mc b = false) :
    legalSquares mc b = [] := by
  cases moves : legalSquares mc b with
  | nil => rfl
  | cons first rest =>
    have legal := (Board.getMoves_exact mc b first.val.val).mp first.property
    have mobile := (Board.hasLegalMoves_exact mc b).mpr ⟨first.val.val, legal⟩
    rw [stuck] at mobile
    cases mobile

def minimax (mc : X86Config) (fc : Flip.Dispatch.X86Config) (b : Position) : Int :=
  PvsMoveLoop.minimax (build mc fc b)

theorem minimax_moves (mc : X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    (first : GeneratedSquare mc b.val) (rest : List (GeneratedSquare mc b.val))
    (moves : legalSquares mc b.val = first :: rest) :
    minimax mc fc b = PvsMoveLoop.bestValue (fun sq => -minimax mc fc (play mc fc b sq).val)
      (first :: rest) (-PvsMoveLoop.scoreInf) := by
  have turn : build mc fc b =
      .turn (build mc fc (play mc fc b first).val)
        (rest.map fun sq => build mc fc (play mc fc b sq).val) := by
    rw [build.eq_def]
    split
    next eq =>
      rw [moves] at eq
      cases eq
      rfl
    next eq =>
      rw [moves] at eq
      cases eq
  unfold minimax
  rw [turn, PvsMoveLoop.minimax_turn]
  exact PvsMoveLoop.bestValue_map (fun child => -PvsMoveLoop.minimax child)
    (fun sq => build mc fc (play mc fc b sq).val) (first :: rest) _

theorem minimax_pass (mc : X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    (stuck : legalSquares mc b.val = [])
    (other : Board.hasLegalMoves mc (pass b).val = true) :
    minimax mc fc b = -minimax mc fc (pass b) := by
  have switched : build mc fc b = .pass (build mc fc (pass b)) := by
    rw [build.eq_def]
    split
    next eq =>
      rw [stuck] at eq
      cases eq
    next => simp [other]
  unfold minimax
  rw [switched, PvsMoveLoop.minimax.eq_def]

def Contract : Prop :=
  ∀ (mc : X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    (pv : Bool) (alpha beta : Int), Window alpha beta →
    (pv = false → beta = alpha + 1) →
    SoundResult (minimax mc fc b) alpha beta (PvsMoveLoop.run (build mc fc b) pv alpha beta)

theorem contract_proven : Contract := by
  intro mc fc b pv alpha beta window null
  exact PvsMoveLoop.run_sound (build mc fc b) pv alpha beta window null

theorem terminal_tree (mc : X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    (stuck : legalSquares mc b.val = [])
    (other : Board.hasLegalMoves mc (Board.switchPlayers b.val) = false) :
    build mc fc b = .leaf (terminalScore b) := by
  rw [build.eq_def]
  split
  next eq => rw [stuck] at eq; cases eq
  next => simp [pass, other]

theorem terminal_value (mc : X86Config) (fc : Flip.Dispatch.X86Config) (b : Position)
    (stuck : legalSquares mc b.val = [])
    (other : Board.hasLegalMoves mc (Board.switchPlayers b.val) = false) :
    minimax mc fc b = 256 * Board.AwardedScore b.val := by
  unfold minimax
  rw [terminal_tree mc fc b stuck other, PvsMoveLoop.minimax.eq_def]
  rfl

theorem fullWindow_exact (mc : X86Config) (fc : Flip.Dispatch.X86Config) (b : Position) :
    PvsMoveLoop.run (build mc fc b) true (-PvsMoveLoop.scoreInf) PvsMoveLoop.scoreInf =
      minimax mc fc b :=
  PvsMoveLoop.fullWindow_exact _

end NeuralReversi.PvsBoardTree

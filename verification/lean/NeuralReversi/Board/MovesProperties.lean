module
public import NeuralReversi.Board.MovesModel
public import NeuralReversi.Board.Counting
public import NeuralReversi.Bitboard.DispatchProperties
public import NeuralReversi.Bitboard.SquareProperties

@[expose] public section

namespace NeuralReversi.Board
open Bitboard.Moves (X86Config LegalMove)

theorem getMoves_exact (config : X86Config) (b : Board) (i : Nat) :
    (getMoves config b).getLsbD i = true ↔ LegalMove b.player b.opponent i :=
  Bitboard.getMoves_exact config b.player b.opponent i

def X86CombinedMovesContract : Prop := ∀ config b i,
  Bitboard.Moves.BothCoordinateOutputs b.player b.opponent i (getMovesAndPotential config b)

theorem x86CombinedMovesContract_proven : X86CombinedMovesContract := by
  intro config b i
  exact Bitboard.getMovesAndPotential_exact config b.player b.opponent i

def X86IsLegalMoveContract : Prop :=
  ∀ config debugAssertions overflowChecks b (sq : Bitboard.Square),
    isLegalMove config debugAssertions overflowChecks b sq =
      .ok ((getMoves config b).getLsbD sq.val) ∧
    ((getMoves config b).getLsbD sq.val = true ↔ LegalMove b.player b.opponent sq.val)

theorem x86IsLegalMoveContract_proven : X86IsLegalMoveContract := by
  intro config debugAssertions overflowChecks b sq
  exact ⟨Bitboard.squareMembershipContract_proven _ sq debugAssertions overflowChecks,
    getMoves_exact config b sq.val⟩

theorem moves_zero_iff (config : X86Config) (b : Board) :
    getMoves config b = 0 ↔ ∀ i, ¬ LegalMove b.player b.opponent i := by
  rw [Bitboard.word_zero_iff_no_bits]
  apply forall_congr'
  intro i
  rw [← getMoves_exact]
  cases (getMoves config b).getLsbD i <;> decide

theorem hasLegalMoves_exact (config : X86Config) (b : Board) :
    hasLegalMoves config b = true ↔ ∃ i, LegalMove b.player b.opponent i := by
  classical
  have nonzero : (hasLegalMoves config b = true) ↔ getMoves config b ≠ 0 := by
    simp [hasLegalMoves,Bitboard.isEmpty]
  rw [nonzero]
  constructor
  · intro notzero
    by_cases witness : ∃ i, LegalMove b.player b.opponent i
    · exact witness
    · exfalso
      apply notzero
      exact (moves_zero_iff config b).mpr (fun i legal => witness ⟨i,legal⟩)
  · rintro ⟨i,legal⟩ zero
    exact (moves_zero_iff config b).mp zero i legal

theorem isGameOver_exact (config : X86Config) (b : Board) :
    isGameOver config b = true ↔
      (∀ i, ¬ LegalMove b.player b.opponent i) ∧
      (∀ i, ¬ LegalMove b.opponent b.player i) := by
  have current := hasLegalMoves_exact config b
  have other : hasLegalMoves config (switchPlayers b) = true ↔
      ∃ i, LegalMove b.opponent b.player i := hasLegalMoves_exact config (switchPlayers b)
  unfold isGameOver
  cases hc : hasLegalMoves config b <;> cases ho : hasLegalMoves config (switchPlayers b) <;>
    simp_all

def X86BoardMovesContract : Prop := ∀ config b i,
  (getMoves config b).getLsbD i = true ↔ LegalMove b.player b.opponent i

def X86HasLegalMovesContract : Prop := ∀ config b,
  hasLegalMoves config b = true ↔ ∃ i, LegalMove b.player b.opponent i

def X86GameOverContract : Prop := ∀ config b,
  isGameOver config b = true ↔
    (∀ i, ¬ LegalMove b.player b.opponent i) ∧
    (∀ i, ¬ LegalMove b.opponent b.player i)

theorem x86BoardMovesContract_proven : X86BoardMovesContract := getMoves_exact

theorem x86HasLegalMovesContract_proven : X86HasLegalMovesContract := hasLegalMoves_exact

theorem x86GameOverContract_proven : X86GameOverContract := isGameOver_exact

def ForcedPass (config : X86Config) (b : Board) : Prop :=
  hasLegalMoves config b = false ∧ hasLegalMoves config (switchPlayers b) = true

theorem forcedPass_exact (config : X86Config) (b : Board) :
    ForcedPass config b ↔
      (∀ i, ¬ LegalMove b.player b.opponent i) ∧
      (∃ i, LegalMove b.opponent b.player i) := by
  have current := hasLegalMoves_exact config b
  have other : hasLegalMoves config (switchPlayers b) = true ↔
      ∃ i, LegalMove b.opponent b.player i := hasLegalMoves_exact config (switchPlayers b)
  unfold ForcedPass
  cases hc : hasLegalMoves config b <;> cases ho : hasLegalMoves config (switchPlayers b) <;>
    simp_all

theorem forcedPass_not_gameOver (config : X86Config) (b : Board)
    (forced : ForcedPass config b) : isGameOver config b = false := by
  simp [isGameOver, forced.1, forced.2]

theorem forcedPass_not_repeat (config : X86Config) (b : Board)
    (forced : ForcedPass config b) : ¬ ForcedPass config (switchPlayers b) := by
  intro again
  have impossible := again.1
  rw [forced.2] at impossible
  cases impossible

def playRank (config : X86Config) (b : Board) : Nat :=
  2 * (emptyCount b).toNat + if hasLegalMoves config b then 0 else 1

inductive LegalCheckedStep (config : X86Config) : Board → Board → Prop where
  | place {b result : Board} {overflowChecks : Bool} {flipped : Bitboard.Bits}
      {sq : Bitboard.AnySquare}
      (legal : LegalMove b.player b.opponent sq.val)
      (accepted : makeMoveWithFlippedAny true overflowChecks b flipped sq = .ok result) :
      LegalCheckedStep config b result
  | pass {b : Board} (forced : ForcedPass config b) :
      LegalCheckedStep config b (switchPlayers b)

theorem legalCheckedStep_progress (config : X86Config) {before after : Board}
    (edge : LegalCheckedStep config before after) (separated : wellFormed before) :
    wellFormed after ∧ playRank config after + 1 ≤ playRank config before := by
  cases edge with
  | place legal accepted =>
    have step := checkedPlacementProgress_proven _ _ _ _ _ separated accepted
    have decrement := step.2
    have available : hasLegalMoves config before = true :=
      (hasLegalMoves_exact config before).mpr ⟨_, legal⟩
    refine ⟨step.1, ?_⟩
    cases nextAvailable : hasLegalMoves config after <;>
      simp [playRank, available, nextAvailable] <;> omega
  | pass forced =>
    refine ⟨switchPlayers_wellFormed _ separated, ?_⟩
    simp [playRank, forced.1, forced.2, switchPlayers_emptyCount]

theorem legalCheckedStep_strict (config : X86Config) {before after : Board}
    (edge : LegalCheckedStep config before after) (separated : wellFormed before) :
    playRank config after < playRank config before := by
  have bound := (legalCheckedStep_progress config edge separated).2
  omega

theorem terminal_no_legalCheckedStep (config : X86Config) {b result : Board}
    (terminal : isGameOver config b = true) : ¬ LegalCheckedStep config b result := by
  intro edge
  cases edge with
  | place legal accepted =>
    have available : hasLegalMoves config b = true :=
      (hasLegalMoves_exact config b).mpr ⟨_, legal⟩
    simp [isGameOver, available] at terminal
  | pass forced =>
    have notTerminal := forcedPass_not_gameOver config b forced
    rw [notTerminal] at terminal
    cases terminal

inductive LegalCheckedTrace (config : X86Config) : Board → Nat → Board → Prop where
  | nil (b : Board) : LegalCheckedTrace config b 0 b
  | step {start current result : Board} {steps : Nat}
      (history : LegalCheckedTrace config start steps current)
      (edge : LegalCheckedStep config current result) :
      LegalCheckedTrace config start (steps + 1) result

theorem legalCheckedTrace_rank_bound (config : X86Config) {start result : Board} {steps : Nat}
    (trace : LegalCheckedTrace config start steps result) (separated : wellFormed start) :
    wellFormed result ∧ playRank config result + steps ≤ playRank config start := by
  revert separated
  induction trace with
  | nil =>
    intro separated
    exact ⟨separated, by omega⟩
  | step history edge ih =>
    intro separated
    obtain ⟨currentWf, total⟩ := ih separated
    have progress := legalCheckedStep_progress config edge currentWf
    exact ⟨progress.1, by have := progress.2; omega⟩

theorem legalCheckedTrace_length_bound (config : X86Config) {start result : Board} {steps : Nat}
    (trace : LegalCheckedTrace config start steps result) (separated : wellFormed start) :
    steps ≤ playRank config start := by
  have bound := (legalCheckedTrace_rank_bound config trace separated).2
  omega

theorem legalCheckedTrace_empty_bound (config : X86Config) {start result : Board} {steps : Nat}
    (trace : LegalCheckedTrace config start steps result) (separated : wellFormed start) :
    steps ≤ 2 * (emptyCount start).toNat + 1 ∧
    (hasLegalMoves config start = true → steps ≤ 2 * (emptyCount start).toNat) := by
  have bound := legalCheckedTrace_length_bound config trace separated
  constructor
  · cases mobility : hasLegalMoves config start <;> simp [playRank, mobility] at bound <;> omega
  · intro available
    simpa [playRank, available] using bound

set_option maxRecDepth 4096 in
theorem initial_playRank (config : X86Config) : playRank config newBoard = 120 := by
  have legal : LegalMove newBoard.player newBoard.opponent 19 :=
    (getMoves_exact ⟨false, false⟩ newBoard 19).mp (by decide)
  have available : hasLegalMoves config newBoard = true :=
    (hasLegalMoves_exact config newBoard).mpr ⟨19, legal⟩
  have empties : (emptyCount newBoard).toNat = 60 := by decide
  simp [playRank, available, empties]

set_option maxRecDepth 4096 in
theorem legalCheckedTrace_initial_bound (config : X86Config) {result : Board} {steps : Nat}
    (trace : LegalCheckedTrace config newBoard steps result) :
    wellFormed result ∧ playRank config result + steps ≤ 120 ∧ steps ≤ 120 := by
  have initialWf : wellFormed newBoard := by
    unfold wellFormed
    decide
  obtain ⟨separated, bound⟩ := legalCheckedTrace_rank_bound config trace initialWf
  rw [initial_playRank] at bound
  exact ⟨separated, bound, by omega⟩

theorem legalCheckedStep_wellFounded (config : X86Config) :
    WellFounded (fun (next current : { b : Board // wellFormed b }) =>
      LegalCheckedStep config current.val next.val) := by
  apply Subrelation.wf (r := fun (next current : { b : Board // wellFormed b }) =>
    playRank config next.val < playRank config current.val)
  · intro next current edge
    exact legalCheckedStep_strict config edge current.property
  · exact InvImage.wf (fun b : { b : Board // wellFormed b } => playRank config b.val)
      Nat.lt_wfRel.wf

theorem legalCheckedStep_no_infinite (config : X86Config) (boards : Nat → Board)
    (initialWf : wellFormed (boards 0))
    (edges : ∀ n, LegalCheckedStep config (boards n) (boards (n + 1))) : False := by
  have traces : ∀ n, LegalCheckedTrace config (boards 0) n (boards n) := by
    intro n
    induction n with
    | zero => exact .nil _
    | succ n ih => exact .step ih (edges n)
  have bound := legalCheckedTrace_length_bound config
    (traces (playRank config (boards 0) + 1)) initialWf
  omega

theorem legalCheckedTrace_strict (config : X86Config) {start result : Board} {steps : Nat}
    (trace : LegalCheckedTrace config start steps result) (separated : wellFormed start)
    (nonempty : 0 < steps) : playRank config result < playRank config start := by
  have bound := (legalCheckedTrace_rank_bound config trace separated).2
  omega

theorem legalCheckedTrace_no_cycle (config : X86Config) {b : Board} {steps : Nat}
    (trace : LegalCheckedTrace config b steps b) (separated : wellFormed b) : steps = 0 := by
  have bound := (legalCheckedTrace_rank_bound config trace separated).2
  omega

theorem legalCheckedTrace_no_return (config : X86Config) {start result : Board} {steps : Nat}
    (trace : LegalCheckedTrace config start steps result) (separated : wellFormed start)
    (nonempty : 0 < steps) : ∀ backSteps, ¬ LegalCheckedTrace config result backSteps start := by
  intro backSteps back
  obtain ⟨resultWf, forwardBound⟩ := legalCheckedTrace_rank_bound config trace separated
  have backBound := (legalCheckedTrace_rank_bound config back resultWf).2
  omega

example : ForcedPass ⟨false, false⟩ ⟨2, 1⟩ := by
  unfold ForcedPass
  decide

example : LegalCheckedStep ⟨false, false⟩ ⟨2, 1⟩ ⟨1, 2⟩ := by
  apply LegalCheckedStep.pass
  unfold ForcedPass
  decide

example : LegalCheckedStep ⟨false, false⟩ ⟨1, 2⟩ ⟨0, 7⟩ := by
  apply LegalCheckedStep.place (overflowChecks := true) (flipped := 2) (sq := ⟨2, by decide⟩)
  · exact (getMoves_exact ⟨false, false⟩ ⟨1, 2⟩ 2).mp (by decide)
  · rfl

example : isGameOver ⟨false, false⟩ ⟨0, 7⟩ = true := by decide

example : ¬ ForcedPass ⟨false, false⟩ ⟨0, 0⟩ := by
  unfold ForcedPass
  decide

set_option maxRecDepth 4096 in
example : playRank ⟨false, false⟩ ⟨1, 2⟩ + 1 = playRank ⟨false, false⟩ ⟨2, 1⟩ := by
  decide

end NeuralReversi.Board

import NeuralReversi.Board.Model
import NeuralReversi.Board.Properties
import NeuralReversi.Board.InitializationModel
import NeuralReversi.Bitboard.Counting

namespace NeuralReversi.Board
open Bitboard

def SquareCountsContract : Prop :=
  ∀ b : Board,
    (playerCount b).toNat = (List.range 64).countP b.player.getLsbD ∧
    (opponentCount b).toNat = (List.range 64).countP b.opponent.getLsbD ∧
    (emptyCount b).toNat = (List.range 64).countP
      (fun i => !(b.player.getLsbD i || b.opponent.getLsbD i))

theorem squareCountsContract_proven : SquareCountsContract := by
  intro b
  refine ⟨(populationCountContract_proven b.player).1,
    (populationCountContract_proven b.opponent).1, ?_⟩
  change (count (emptySquares b)).toNat = _
  rw [(populationCountContract_proven (emptySquares b)).1]
  apply List.countP_congr
  intro i member
  have within := List.mem_range.mp member
  simp [emptySquares, within]

def CountConservation : Prop :=
  ∀ b : Board, wellFormed b →
    (playerCount b).toNat + (opponentCount b).toNat + (emptyCount b).toNat = 64

theorem countConservation_proven : CountConservation := by
  intro b disjoint
  have unionCount := count_or_of_disjoint b.player b.opponent disjoint
  have complement := (populationCountContract_proven (b.player ||| b.opponent)).2.2.2
  change (count b.player).toNat + (count b.opponent).toNat +
    (count (~~~(b.player ||| b.opponent))).toNat = 64
  omega

private theorem range_countP_equal (n k : Nat) :
    (List.range n).countP (fun i => decide (i = k)) = if k < n then 1 else 0 := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.countP_append, List.countP_singleton, ih]
    by_cases prior : k < n
    · have next : k < n + 1 := by omega
      have different : n ≠ k := by omega
      simp [prior, next, different]
    · by_cases same : n = k
      · subst k
        simp
      · have next : ¬ k < n + 1 := by omega
        simp [prior, next, same]

theorem squareMask_count_one (sq : Square) : (count (squareMask sq)).toNat = 1 := by
  rw [(populationCountContract_proven (squareMask sq)).1]
  have cells : (squareMask sq).getLsbD = fun i => decide (i = sq.val) :=
    funext (fun i => squareMask_bit sq i)
  rw [cells, range_countP_equal]
  simp only [sq.isLt, ↓reduceIte]

theorem place_emptyCount_step (b : Board) (flipped : Bits) (sq : Square)
    (disjoint : wellFormed b) (subset : flipped &&& ~~~b.opponent = 0)
    (empty : squareMask sq &&& (b.player ||| b.opponent) = 0) :
    (emptyCount (place b flipped sq)).toNat + 1 = (emptyCount b).toNat := by
  have invariant := placementInvariant_proven b flipped sq disjoint subset empty
  have unionCount := count_or_of_disjoint (b.player ||| b.opponent) (squareMask sq)
    (by rw [BitVec.and_comm]; exact empty)
  have singleton := squareMask_count_one sq
  have before : (emptyCount b).toNat + (count (b.player ||| b.opponent)).toNat = 64 :=
    (populationCountContract_proven (b.player ||| b.opponent)).2.2.2
  have after : (emptyCount (place b flipped sq)).toNat +
      (count ((place b flipped sq).player ||| (place b flipped sq).opponent)).toNat = 64 :=
    (populationCountContract_proven
      ((place b flipped sq).player ||| (place b flipped sq).opponent)).2.2.2
  rw [invariant.2.2.2] at after
  omega

def CheckedPlacementProgress : Prop :=
  ∀ overflowChecks (b : Board) (flipped : Bits) (sq : AnySquare) (result : Board),
    wellFormed b → makeMoveWithFlippedAny true overflowChecks b flipped sq = .ok result →
    wellFormed result ∧ (emptyCount result).toNat + 1 = (emptyCount b).toNat

theorem checkedPlacementProgress_proven : CheckedPlacementProgress := by
  intro overflowChecks b flipped sq result disjoint accepted
  obtain ⟨_, _, separated, _⟩ :=
    fullCheckedPlacementInvariant_proven overflowChecks b flipped sq result disjoint accepted
  have bridge := makeMoveWithFlippedAny_valid true overflowChecks b flipped sq
  have acceptedOld : makeMoveWithFlipped true b flipped sq = .ok result := by
    rw [← bridge]
    exact accepted
  have resultEq : result = place b flipped sq := by
    by_cases hf : flipped = 0 <;>
      by_cases hs : emptySquares b &&& squareMask sq = 0 <;>
      by_cases ho : flipped &&& ~~~b.opponent = 0 <;>
      simp_all [makeMoveWithFlipped]
  have success : makeMoveWithFlipped true b flipped sq = .ok (place b flipped sq) := by
    simpa only [resultEq] using acceptedOld
  have pre := ((playableGuardedMoveContract_proven b flipped sq).2.1).mp success
  have empty : squareMask sq &&& (b.player ||| b.opponent) = 0 :=
    (empty_squareMask (b.player ||| b.opponent) sq).mp
      (by simpa [emptySquares] using pre.2.1)
  refine ⟨separated, ?_⟩
  rw [resultEq]
  exact place_emptyCount_step b flipped sq disjoint pre.2.2 empty

theorem checkedPlacement_strict_progress (overflowChecks : Bool) (b : Board)
    (flipped : Bits) (sq : AnySquare) (result : Board) (disjoint : wellFormed b)
    (accepted : makeMoveWithFlippedAny true overflowChecks b flipped sq = .ok result) :
    (emptyCount result).toNat < (emptyCount b).toNat := by
  have step := (checkedPlacementProgress_proven overflowChecks b flipped sq result disjoint accepted).2
  omega

theorem switchPlayers_emptyCount (b : Board) :
    emptyCount (switchPlayers b) = emptyCount b := by
  unfold emptyCount
  rw [(perspectiveExchange_proven b).2.2.2]

theorem switchPlayers_wellFormed (b : Board) (disjoint : wellFormed b) :
    wellFormed (switchPlayers b) := by
  change b.opponent &&& b.player = 0
  rw [BitVec.and_comm]
  exact disjoint

inductive CheckedPlacementTrace : Board → Nat → Board → Prop where
  | nil (b : Board) : CheckedPlacementTrace b 0 b
  | place {start current result : Board} {placements : Nat}
      {overflowChecks : Bool} {flipped : Bits} {sq : AnySquare}
      (history : CheckedPlacementTrace start placements current)
      (accepted : makeMoveWithFlippedAny true overflowChecks current flipped sq = .ok result) :
      CheckedPlacementTrace start (placements + 1) result
  | switch {start current : Board} {placements : Nat}
      (history : CheckedPlacementTrace start placements current) :
      CheckedPlacementTrace start placements (switchPlayers current)

theorem checkedPlacementTrace_progress {start result : Board} {placements : Nat}
    (trace : CheckedPlacementTrace start placements result) (disjoint : wellFormed start) :
    wellFormed result ∧ (emptyCount result).toNat + placements = (emptyCount start).toNat := by
  revert disjoint
  induction trace with
  | nil =>
    intro disjoint
    exact ⟨disjoint, by omega⟩
  | place history accepted ih =>
    intro disjoint
    obtain ⟨currentWf, total⟩ := ih disjoint
    have step := checkedPlacementProgress_proven _ _ _ _ _ currentWf accepted
    exact ⟨step.1, by have := step.2; omega⟩
  | switch history ih =>
    intro disjoint
    obtain ⟨currentWf, total⟩ := ih disjoint
    refine ⟨switchPlayers_wellFormed _ currentWf, ?_⟩
    rw [switchPlayers_emptyCount]
    exact total

theorem checkedPlacementTrace_bound {start result : Board} {placements : Nat}
    (trace : CheckedPlacementTrace start placements result) (disjoint : wellFormed start) :
    placements ≤ (emptyCount start).toNat := by
  have total := (checkedPlacementTrace_progress trace disjoint).2
  omega

set_option maxRecDepth 4096 in
theorem checkedPlacementTrace_initial_bound {result : Board} {placements : Nat}
    (trace : CheckedPlacementTrace newBoard placements result) :
    wellFormed result ∧ (emptyCount result).toNat + placements = 60 ∧ placements ≤ 60 := by
  have initialWf : wellFormed newBoard := by
    unfold wellFormed
    decide
  have initialEmpties : (emptyCount newBoard).toNat = 60 := by decide
  obtain ⟨separated, total⟩ := checkedPlacementTrace_progress trace initialWf
  rw [initialEmpties] at total
  exact ⟨separated, total, by omega⟩

example : makeMoveWithFlippedAny true true newBoard (squareMask ⟨27, by decide⟩)
    ⟨19, by decide⟩ = .ok (place newBoard (squareMask ⟨27, by decide⟩) ⟨19, by decide⟩) := by
  rfl

set_option maxRecDepth 4096 in
example : (emptyCount (place newBoard (squareMask ⟨27, by decide⟩) ⟨19, by decide⟩)).toNat = 59 := by
  decide

example : makeMoveWithFlippedAny true true ⟨1, 1⟩ 1 ⟨1, by decide⟩ = .ok ⟨0, 3⟩ := by
  rfl

example : makeMoveWithFlippedAny false false ⟨1, 0⟩ 0 ⟨0, by decide⟩ = .ok ⟨0, 1⟩ := by
  rfl

end NeuralReversi.Board

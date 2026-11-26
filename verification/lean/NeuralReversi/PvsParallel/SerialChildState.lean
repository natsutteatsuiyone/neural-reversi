import NeuralReversi.PvsParallel.ExactDepth
import NeuralReversi.EndgameSearch.Cancellation

namespace NeuralReversi.PvsParallel.SerialChildState

open Bitboard.Moves (X86Config)
open ExactDepth (Depth FlipConfig RecursiveCall)
open PvsBoardTree (Position GeneratedSquare play pass legalSquares)
open EndgameSearch.Cancellation (EmptyState)

def nextState (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (state : EmptyState) (next : Position) (childDepth : Depth) : Option EmptyState :=
  if Board.hasLegalMoves mc b.val then
    if childDepth = depth - 1 then
      ((legalSquares mc b.val).find? fun m => decide ((play mc fc b m).val.val = next.val)).map
        fun m => state.remove m.val
    else none
  else if Board.hasLegalMoves mc (pass b).val && decide (childDepth = depth ∧ next.val = (pass b).val) then
    some state
  else none

theorem nextState_play (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (state : EmptyState) (m : GeneratedSquare mc b.val) :
    nextState mc fc b depth state (play mc fc b m).val (depth - 1) = some (state.remove m.val) := by
  have legal := (Board.getMoves_exact mc b.val m.val.val).mp m.property
  have mobile := (Board.hasLegalMoves_exact mc b.val).mpr ⟨m.val.val, legal⟩
  obtain ⟨sq, member, value⟩ := (PvsBoardTree.legalSquares_complete mc b.val m.val).mpr legal
  have present : m ∈ legalSquares mc b.val := Subtype.ext value ▸ member
  simp only [nextState, mobile, ite_true]
  let isChild : GeneratedSquare mc b.val → Bool :=
    fun move => decide ((play mc fc b move).val.val = (play mc fc b m).val.val)
  change ((legalSquares mc b.val).find? isChild).map _ = _
  cases found : (legalSquares mc b.val).find? isChild with
  | none =>
    have missed := List.find?_eq_none.mp found m present
    simp [isChild] at missed
  | some candidate =>
    have same : (play mc fc b candidate).val.val = (play mc fc b m).val.val := by
      simpa only [isChild, decide_eq_true_eq] using List.find?_some found
    have selected := PvsBoardTree.play_injective mc fc b same
    subst candidate
    rfl

theorem nextState_pass (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (state : EmptyState) (forced : Board.ForcedPass mc b.val) :
    nextState mc fc b depth state (pass b) depth = some state := by
  have available : Board.hasLegalMoves mc (pass b).val = true := forced.2
  simp [nextState, forced.1, available]

/-- The child ABI identifies the exact empty-list update for every legal
    recursive edge. A forced pass preserves the prior list and parity. -/
theorem nextState_matches (edge : RecursiveCall mc fc b depth next childDepth)
    (aligned : state.matches b) :
    ∃ childState, nextState mc fc b depth state next childDepth = some childState ∧
      childState.matches next := by
  cases edge with
  | ply move depth =>
    exact ⟨state.remove move.val, nextState_play mc fc _ depth state move,
      (EndgameSearch.emptyStatePlayContract_proven mc fc _ move _ _ aligned).2.1⟩
  | forcedPass depth forced =>
    exact ⟨state, nextState_pass mc fc b depth state forced,
      EndgameSearch.emptyState_pass b _ _ aligned⟩

theorem edge_of_nextState {mc : X86Config} {fc : FlipConfig} {b next : Position} {d cd : Depth}
    (accepted : nextState mc fc b d s next cd = some t) : RecursiveCall mc fc b d next cd := by
  unfold nextState at accepted
  split at accepted
  · split at accepted
    · rename_i same
      obtain ⟨m, found, _⟩ := Option.map_eq_some_iff.mp accepted
      have board : (play mc fc b m).val = next := Subtype.ext (by
        simpa only [decide_eq_true_eq] using List.find?_some found)
      rw [← board, same]
      exact .ply b m d
    · contradiction
  · rename_i stuck
    split at accepted
    · rename_i test
      obtain ⟨mobile, same, pos⟩ := (show _ ∧ _ ∧ _ from by simpa only [Bool.and_eq_true, decide_eq_true_eq] using test)
      have board : next = pass b := Subtype.ext pos
      rw [board, same]
      exact .forcedPass b d ⟨Bool.eq_false_iff.mpr stuck, mobile⟩
    · contradiction

end NeuralReversi.PvsParallel.SerialChildState

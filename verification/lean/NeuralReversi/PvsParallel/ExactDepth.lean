import NeuralReversi.EndgameSearch.EmptyState
import NeuralReversi.LateMoveReduction
import NeuralReversi.RootDispatch

namespace NeuralReversi.PvsParallel.ExactDepth

open Bitboard.Moves (X86Config)
open PvsBoardTree (Position GeneratedSquare play pass minimax)
open ScoreRefresh.BoardBridge (finalScore)

abbrev Depth := BitVec 32
abbrev FlipConfig := Flip.Dispatch.X86Config

def Full (b : Position) (depth : Depth) : Prop := depth = Board.emptyCount b.val

def scoutDepth (depth : Depth) (node : LateMoveReduction.Node)
    (cut ttPv : Bool) (alpha moveValue : Int) : Depth :=
  depth - 1 - BitVec.ofNat 32 (LateMoveReduction.scheduled node cut ttPv alpha moveValue)

theorem move_depth (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (move : GeneratedSquare mc b.val) (full : Full b depth) :
    0 < depth.toNat ∧ Full (play mc fc b move).val (depth - 1) ∧
    ((depth - 1) + 1).toNat = depth.toNat := by
  obtain ⟨pos, child, _, _, restored⟩ :=
    EndgameSearch.emptyStatePlayContract_proven mc fc b move _ _
      (EndgameSearch.emptyState_initial b)
  unfold Full at full ⊢
  rw [full]
  exact ⟨pos, child.1, congrArg BitVec.toNat restored⟩

theorem pass_depth (b : Position) (depth : Depth) (full : Full b depth) :
    Full (pass b) depth :=
  full.trans (Board.switchPlayers_emptyCount b.val).symm

theorem endgame_scout_depth (mc : X86Config) (fc : FlipConfig) (b : Position) (depth : Depth)
    (move : GeneratedSquare mc b.val) (node : LateMoveReduction.Node) (cut ttPv : Bool)
    (alpha moveValue : Int) (full : Full b depth) (endgame : node.endgame = true) :
    scoutDepth depth node cut ttPv alpha moveValue = depth - 1 ∧
    Full (play mc fc b move).val (scoutDepth depth node cut ttPv alpha moveValue) := by
  have eq : scoutDepth depth node cut ttPv alpha moveValue = depth - 1 := by
    simp [scoutDepth, LateMoveReduction.scheduled, LateMoveReduction.reduction,
      LateMoveReduction.disabled, endgame]
  exact ⟨eq, eq ▸ (move_depth mc fc b depth move full).2.1⟩

def iir (endgame root allNode selective : Bool) (depth : Nat) (ttMove : Bool) : Bool :=
  !endgame && !root && !allNode && selective && decide (6 ≤ depth) && !ttMove

def ttProbcut (endgame pv selective : Bool) : Bool := !endgame && !pv && selective

def endgameProbcut (selectivity : Fin 6) (decision : Unit → Option Int) : Option Int :=
  if selectivity = 5 then none else decision ()

def HeuristicGateContract : Prop :=
  (∀ root allNode selective depth ttMove, iir true root allNode selective depth ttMove = false) ∧
  (∀ pv selective, ttProbcut true pv selective = false) ∧
  (∀ decision, endgameProbcut 5 decision = none) ∧
  ∀ (node : LateMoveReduction.Node) cut ttPv alpha ordering,
    node.endgame = true → LateMoveReduction.scheduled node cut ttPv alpha ordering = 0

theorem heuristicGateContract_proven : HeuristicGateContract := by
  refine ⟨fun _ _ _ _ _ => rfl, fun _ _ => rfl, fun _ => rfl, ?_⟩
  intro node cut ttPv alpha ordering endgame
  simp [LateMoveReduction.scheduled, LateMoveReduction.reduction, LateMoveReduction.disabled, endgame]

inductive RecursiveCall (mc : X86Config) (fc : FlipConfig) :
    Position → Depth → Position → Depth → Prop
  | ply (b : Position) (move : GeneratedSquare mc b.val) (depth : Depth) :
      RecursiveCall mc fc b depth (play mc fc b move).val (depth - 1)
  | forcedPass (b : Position) (depth : Depth) (forced : Board.ForcedPass mc b.val) :
      RecursiveCall mc fc b depth (pass b) depth

theorem RecursiveCall.full
    (call : RecursiveCall mc fc before depth after childDepth) (initial : Full before depth) :
    Full after childDepth := by
  cases call with
  | ply move _ => exact (move_depth mc fc before depth move initial).2.1
  | forcedPass _ _ => exact pass_depth before depth initial

theorem RecursiveCall.earlier (call : RecursiveCall mc fc before depth after childDepth) :
    Board.playRank mc after.val < Board.playRank mc before.val := by
  cases call with
  | ply move _ => exact (play mc fc before move).property
  | forcedPass _ forced => exact Board.legalCheckedStep_strict mc (.pass forced) before.property

inductive RecursiveCalls (mc : X86Config) (fc : FlipConfig) :
    Position → Depth → Position → Depth → Nat → Prop
  | done : RecursiveCalls mc fc b d b d 0
  | next : RecursiveCall mc fc b d child cd → RecursiveCalls mc fc child cd result rd calls →
      RecursiveCalls mc fc b d result rd (calls + 1)

def DepthContract : Prop :=
  ∀ mc fc (b result : Position) depth childDepth calls,
    Full b depth → RecursiveCalls mc fc b depth result childDepth calls →
    Full result childDepth ∧ Board.playRank mc result.val + calls ≤ Board.playRank mc b.val

theorem depthContract_proven : DepthContract := by
  intro mc fc b result depth childDepth calls initial trace
  induction trace with
  | done => exact ⟨initial, by simp⟩
  | next call _ ih =>
    obtain ⟨terminal, rank⟩ := ih (call.full initial)
    have decreases := call.earlier
    exact ⟨terminal, by omega⟩

private theorem unoccupied_none (b : Position) (full : Board.emptySquares b.val = 0#64)
    (mc : X86Config) : Board.hasLegalMoves mc b.val = false := by
  cases present : Board.hasLegalMoves mc b.val
  · rfl
  · obtain ⟨i, legal⟩ := (Board.hasLegalMoves_exact mc b.val).mp present
    have h : (Board.emptySquares b.val).getLsbD i = true := by
      simp [Board.emptySquares, legal.1, legal.2.1, legal.2.2.1, -BitVec.getLsbD_eq_getElem]
    simp [full] at h

def ZeroDepthContract : Prop :=
  ∀ mc fc (b : Position) (checked : Bool),
    Full b 0 → Board.hasLegalMoves mc b.val = false ∧
      Board.hasLegalMoves mc (pass b).val = false ∧
      (finalScore checked b.val).map (fun score => score.raw.toInt) = .ok (minimax mc fc b)

theorem zeroDepthContract_proven : ZeroDepthContract := by
  intro mc fc b checked zero
  have full : Board.emptySquares b.val = 0#64 :=
    ((Bitboard.populationCountContract_proven (Board.emptySquares b.val)).2.2.1).mp zero.symm
  have alsoFull : Board.emptySquares (pass b).val = 0#64 := by
    simpa only [PvsBoardTree.pass, Board.perspectiveExchange_proven b.val |>.2.2.2] using full
  have stuck := unoccupied_none b full mc
  have other := unoccupied_none (pass b) alsoFull mc
  refine ⟨stuck, other, ?_⟩
  rw [ScoreRefresh.BoardBridge.finalScore_scaled, PvsBoardTree.terminal_value mc fc b
    (PvsBoardTree.nil_of_stuck stuck) other]
  have countEq : Board.emptyCount b.val = 0 := zero.symm
  have conservation : (Board.playerCount b.val).toNat + (Board.opponentCount b.val).toNat = 64 := by
    simpa [countEq] using Board.countConservation_proven b.val b.property
  simp [full, Board.AwardedScore, countEq]
  split <;> omega

def PerfectRootContract : Prop :=
  ∀ (empties : Nat), empties ≤ 60 → RootDispatch.fixedPhase 60 5 (fun _ => 60) empties = .endgame 5

theorem perfectRootContract_proven : PerfectRootContract := by
  intro empties fits
  simp [RootDispatch.fixedPhase, EndgameProgress.endSelectivity, EndgameProgress.ladder,
    EndgameProgress.fixedStepAllowed, fits]

end NeuralReversi.PvsParallel.ExactDepth

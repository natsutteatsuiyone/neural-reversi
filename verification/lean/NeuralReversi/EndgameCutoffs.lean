import NeuralReversi.EndgameStability
import NeuralReversi.EndgameTreeCache
import NeuralReversi.Board.Move.Wipeout
import NeuralReversi.EndgameSearch.Thresholds

namespace NeuralReversi.EndgameCutoffs

open PvsBoardTree (Position GeneratedSquare legalSquares play minimax)
open Bitboard.Moves (X86Config)
open EndgameCache (CacheRange)
open EndgameTreeCache (Table Store Address discValue soundHistory entryFor)

private abbrev FlipConfig := Flip.Dispatch.X86Config

def wipesOut (fc : FlipConfig) (b : Board.Board) (sq : Bitboard.Square) : Bool :=
  Flip.Dispatch.resultValue fc sq b.player b.opponent == b.opponent

theorem wipeout_child_value (mc : X86Config) (fc : FlipConfig) (b : Position)
    (sq : GeneratedSquare mc b.val) (wipe : wipesOut fc b.val sq.val = true) :
    minimax mc fc (play mc fc b sq).val = -16384 := by
  have flipAll : Flip.Dispatch.resultValue fc sq.val b.val.player b.val.opponent = b.val.opponent :=
    beq_iff_eq.mp wipe
  obtain ⟨child, movesTo, terminal, score⟩ := Board.wipeoutScoreContract_proven
    mc fc true true b.val sq.val b.property sq.property flipAll
  have accepts := (Board.checkedAcceptsGeneratedMoves_proven mc fc b.val sq.val
    b.property sq.property).2
  have eq : child = (play mc fc b sq).val.val :=
    Except.ok.inj (movesTo.symm.trans accepts)
  let next := (play mc fc b sq).val
  change child = next.val at eq
  subst child
  obtain ⟨stuck, otherStuck⟩ := (Board.isGameOver_exact mc next.val).mp terminal
  have stuckFalse : Board.hasLegalMoves mc next.val = false := by
    cases available : Board.hasLegalMoves mc next.val
    · rfl
    · obtain ⟨i, legal⟩ := (Board.hasLegalMoves_exact mc next.val).mp available
      exact False.elim (stuck i legal)
  have otherFalse : Board.hasLegalMoves mc (Board.switchPlayers next.val) = false := by
    cases available : Board.hasLegalMoves mc (Board.switchPlayers next.val)
    · rfl
    · obtain ⟨i, legal⟩ := (Board.hasLegalMoves_exact mc (Board.switchPlayers next.val)).mp available
      exact False.elim (otherStuck i legal)
  have terminalValue := PvsBoardTree.terminal_value mc fc next (PvsBoardTree.nil_of_stuck stuckFalse) otherFalse
  have award := ScoreRefresh.BoardBridge.solve_scaled_awarded next.val next.property
  rw [score] at award
  have valueEq : ScoreRefresh.scoreMin.raw.toInt = 256 * Board.AwardedScore next.val :=
    Except.ok.inj award
  have minValue : ScoreRefresh.scoreMin.raw.toInt = -16384 := by decide
  change minimax mc fc next = -16384
  omega

theorem wipeout_value (mc : X86Config) (fc : FlipConfig) (b : Position)
    (sq : GeneratedSquare mc b.val) (member : sq ∈ legalSquares mc b.val)
    (wipe : wipesOut fc b.val sq.val = true) :
    minimax mc fc b = 16384 := by
  have result := wipeout_child_value mc fc b sq wipe
  have treeRange := (PvsMoveLoop.minimax_range (PvsBoardTree.build mc fc b)).2
  change minimax mc fc b ≤ 16384 at treeRange
  cases h : legalSquares mc b.val with
  | nil => rw [h] at member; exact False.elim (List.not_mem_nil member)
  | cons first rest =>
    rw [h] at member
    have low := PvsMoveLoop.bestValue_ge_move (fun m => -minimax mc fc (play mc fc b m).val)
      (first :: rest) (-PvsMoveLoop.scoreInf) member
    rw [← PvsBoardTree.minimax_moves mc fc b first rest h] at low
    omega

def beforeMoves (fc : FlipConfig) (b : Board.Board) (history : List Store)
    (address : Address) (table : Table) (alpha : Int) (moves : List Bitboard.Square) : Option Int :=
  if moves.isEmpty then none
  else
    match EndgameCache.probe (entryFor address history table b) b alpha with
    | some result => some result
    | none =>
      match EndgameStability.tryCutoff b (Board.emptyCount b).toNat ⟨alpha⟩ with
      | some result => some result.value
      | none =>
        if table = .deep then
          if moves.any (wipesOut fc b) then some 64
          else if EndgameSearch.Thresholds.minEcEtcEmpties ≤ (Board.emptyCount b).toNat ∧ 2 ≤ moves.length then
            EndgameCache.enhancedCutoff (entryFor address history table) alpha
              (moves.map fun sq => Board.place b (Flip.Dispatch.resultValue fc sq b.player b.opponent) sq)
          else none
        else none

theorem no_wipeout_after_cutoffs (fc : FlipConfig) (b : Board.Board) (history : List Store)
    (address : Address) (alpha : Int) (moves : List Bitboard.Square)
    (ready : beforeMoves fc b history address .deep alpha moves = none)
    (sq : Bitboard.Square) (member : sq ∈ moves) :
    Flip.Dispatch.resultValue fc sq b.player b.opponent ≠ b.opponent := by
  intro allFlipped
  have found : moves.any (wipesOut fc b) = true :=
    List.any_eq_true.mpr ⟨sq, member, beq_iff_eq.mpr allFlipped⟩
  have nonempty : moves.isEmpty = false := by
    cases moves with
    | nil => cases member
    | cons => rfl
  simp only [beforeMoves, nonempty, Bool.false_eq_true, ↓reduceIte, found] at ready
  split at ready
  · contradiction
  · split at ready <;> simp_all

def Contract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (b : Position) (history : List Store)
    (address : Address) (table : Table) (alpha result : Int)
    (moves : List (GeneratedSquare mc b.val)),
    soundHistory mc fc history →
    (∀ sq ∈ moves, sq ∈ legalSquares mc b.val) →
    beforeMoves fc b.val history address table alpha (moves.map Subtype.val) = some result →
    CacheRange result ∧ Solve2.NWS alpha (discValue mc fc b.val) result

theorem contract_proven : Contract := by
  intro mc fc b history address table alpha result moves sound legal hit
  unfold beforeMoves at hit
  split at hit
  · contradiction
  · split at hit
    next score cached =>
      cases hit
      exact EndgameTreeCache.probe_sound sound address table b.val alpha result cached
    next =>
      split at hit
      next score stable =>
        cases hit
        obtain ⟨lo, hi, cutoff, threshold⟩ :=
          (EndgameStability.cutoffContract_proven mc fc b (Board.emptyCount b.val).toNat ⟨alpha⟩).2 score stable
        change score.value ≤ alpha at threshold
        have scaled := EndgameTreeCache.scaled_discValue mc fc b
        unfold CacheRange Solve2.NWS
        exact ⟨⟨by omega, by omega⟩, fun _ => by omega, fun _ => by omega⟩
      next =>
        split at hit
        next deep =>
          subst table
          split at hit
          next found =>
            cases hit
            obtain ⟨target, member, wipe⟩ := List.any_eq_true.mp found
            obtain ⟨sq, inList, rfl⟩ := List.mem_map.mp member
            have value := wipeout_value mc fc b sq (legal sq inList) wipe
            have scaled := EndgameTreeCache.scaled_discValue mc fc b
            unfold CacheRange Solve2.NWS
            exact ⟨by decide, fun _ => by omega, fun _ => by omega⟩
          next =>
            split at hit
            next =>
              have childCut : EndgameCache.enhancedCutoff (entryFor address history .deep) alpha
                    (moves.map fun sq => (play mc fc b sq).val.val) = some result := by
                rw [List.map_map] at hit
                exact hit
              obtain ⟨range, high, bound⟩ :=
                EndgameTreeCache.enhancedContract_proven
                  mc fc b history address alpha result moves sound legal childCut
              have scaled := EndgameTreeCache.scaled_discValue mc fc b
              exact ⟨range, fun _ => by omega, fun _ => by omega⟩
            next => contradiction
        next => contradiction

theorem rawWindow_sound (mc : X86Config) (fc : FlipConfig) (b : Position) (history : List Store)
    (address : Address) (table : Table) (alpha result : Int)
    (moves : List (GeneratedSquare mc b.val)) (sound : soundHistory mc fc history)
    (legal : ∀ sq ∈ moves, sq ∈ legalSquares mc b.val)
    (hit : beforeMoves fc b.val history address table (alpha / 256) (moves.map Subtype.val) =
      some result) :
    Aspiration.SoundResult (minimax mc fc b) alpha (alpha + 1) (256 * result) := by
  obtain ⟨range, bounds⟩ :=
    contract_proven mc fc b history address table (alpha / 256) result moves sound legal hit
  exact EndgameTreeCache.rawWindow_sound mc fc b alpha result (EndgameCache.cacheRange_search range) bounds

end NeuralReversi.EndgameCutoffs

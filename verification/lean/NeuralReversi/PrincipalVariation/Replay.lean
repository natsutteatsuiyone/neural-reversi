import NeuralReversi.PvsBoardTree
import NeuralReversi.PrincipalVariation.Buffer

namespace NeuralReversi.PrincipalVariation

open Bitboard (Square squareMask)
open Bitboard.Moves (X86Config)
open Board (emptyCount hasLegalMoves)
open PvsBoardTree (Position GeneratedSquare pass play)

abbrev FlipConfig := Flip.Dispatch.X86Config

def playable (mc : X86Config) (board : Position) : Position :=
  if hasLegalMoves mc board.val then board else pass board

def moveAt (mc : X86Config) (fc : FlipConfig) (board : Position) (square : Square) : Option Position :=
  if legal : (Board.getMoves mc board.val).getLsbD square.val = true then
    some (play mc fc board ⟨square, legal⟩).val
  else none

def replay (mc : X86Config) (fc : FlipConfig) (board : Position) : List Square → Option Position
  | [] => some board
  | square :: rest =>
    (moveAt mc fc (playable mc board) square).bind (fun next => replay mc fc next rest)

inductive Legal (mc : X86Config) (fc : FlipConfig) : Position → List Square → Position → Prop where
  | empty {board : Position} : Legal mc fc board [] board
  | move {board after : Position} {rest : List Square} (square : GeneratedSquare mc board.val) :
      Legal mc fc (play mc fc board square).val rest after → Legal mc fc board (square.val :: rest) after
  | forcedPass {board after : Position} {square : Square} {rest : List Square} :
      hasLegalMoves mc board.val = false → hasLegalMoves mc (pass board).val = true →
      Legal mc fc (pass board) (square :: rest) after → Legal mc fc board (square :: rest) after

private theorem generated_mobile (mc : X86Config) (board : Position) (square : GeneratedSquare mc board.val) :
    hasLegalMoves mc board.val = true :=
  (Board.hasLegalMoves_exact mc board.val).mpr
    ⟨square.val.val, (Board.getMoves_exact mc board.val _).mp square.property⟩

theorem replay_complete (legal : Legal mc fc board line after) : replay mc fc board line = some after := by
  induction legal with
  | empty => rfl
  | @move board after rest square _ ih =>
    have ready : playable mc board = board := by unfold playable; rw [ite_eq_left (generated_mobile mc board square)]
    rw [replay, ready]
    have atSquare : moveAt mc fc board square.val = some (play mc fc board square).val := by
      unfold moveAt
      rw [dite_eq_left square.property]
    rwa [atSquare]
  | @forcedPass board after square rest stuck other _ ih =>
    have passed : playable mc board = pass board := by simp [playable, stuck]
    have stays : playable mc (pass board) = pass board := by simp [playable, other]
    rw [replay, passed]
    rwa [replay, stays] at ih

theorem replay_sound (mc : X86Config) (fc : FlipConfig) (board : Position) (line : List Square)
    (after : Position) (ok : replay mc fc board line = some after) : Legal mc fc board line after := by
  induction line generalizing board with
  | nil =>
    cases Option.some.inj ok
    exact .empty
  | cons square rest ih =>
    rw [replay] at ok
    unfold moveAt at ok
    split at ok
    next legal =>
      let mover := playable mc board
      let generated : GeneratedSquare mc mover.val := ⟨square, legal⟩
      have stepped : Legal mc fc mover (square :: rest) after := .move generated (ih _ ok)
      cases mobile : hasLegalMoves mc board.val
      · have passed : mover = pass board := by simp [mover, playable, mobile]
        rw [passed] at stepped
        have next : hasLegalMoves mc (pass board).val = true := passed ▸ generated_mobile mc mover generated
        exact .forcedPass mobile next stepped
      · have same : mover = board := by simp [mover, playable, mobile]
        exact same ▸ stepped
    next => cases ok

def vacant (board : Position) (square : Square) : Prop :=
  (board.val.player ||| board.val.opponent).getLsbD square.val = false

private theorem move_occupancy (mc : X86Config) (fc : FlipConfig) (board : Position)
    (square : GeneratedSquare mc board.val) :
    let next := (play mc fc board square).val
    next.val.player ||| next.val.opponent = (board.val.player ||| board.val.opponent) ||| squareMask square.val := by
  have accepted := (Board.precomputedMoveContract_proven mc fc true true
    board.val square.val board.property square.property).1
  rw [Board.makeMoveWithFlippedAny_valid] at accepted
  exact (Board.playableCheckedPlacementInvariant_proven _ _ _ _ board.property accepted).2.2.2

private theorem move_count (mc : X86Config) (fc : FlipConfig) (board : Position)
    (square : GeneratedSquare mc board.val) :
    (emptyCount (play mc fc board square).val.val).toNat + 1 = (emptyCount board.val).toNat :=
  (Board.checkedPlacementProgress_proven true _ _ _ _ board.property
    (Board.precomputedMoveContract_proven mc fc true true board.val square.val board.property square.property).1).2

private theorem line_stats (legal : Legal mc fc board line after) :
    line.Nodup ∧ (emptyCount after.val).toNat + line.length = (emptyCount board.val).toNat ∧
      ∀ square ∈ line, vacant board square := by
  induction legal with
  | empty => simp
  | @move board after rest square child ih =>
    obtain ⟨nodup, count, later⟩ := ih
    have delta := move_count mc fc board square
    have occupied := move_occupancy mc fc board square
    have target := (Board.getMoves_exact mc board.val square.val.val).mp square.property
    refine ⟨List.nodup_cons.mpr ⟨?_, nodup⟩, by simp only [List.length_cons]; omega, ?_⟩
    · intro repeated
      have childEmpty := later square.val repeated
      unfold vacant at childEmpty
      rw [occupied, BitVec.getLsbD_or, Bitboard.squareMask_bit] at childEmpty
      simp at childEmpty
    · intro candidate member
      rcases List.mem_cons.mp member with same | tail
      · subst candidate
        unfold vacant
        rw [BitVec.getLsbD_or, target.2.1, target.2.2.1]
        rfl
      · have childEmpty := later candidate tail
        unfold vacant at childEmpty ⊢
        rw [occupied, BitVec.getLsbD_or] at childEmpty
        exact (Bool.or_eq_false_iff.mp childEmpty).1
  | @forcedPass board after square rest stuck other child ih =>
    obtain ⟨nodup, count, empty⟩ := ih
    exact ⟨nodup, by simpa [pass, Board.switchPlayers_emptyCount] using count, fun sq h => by
      simpa [vacant, pass, Board.switchPlayers, BitVec.or_comm] using empty sq h⟩

def ReplayContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (board : Position) (line : List Square) (after : Position),
    (replay mc fc board line = some after ↔ Legal mc fc board line after) ∧
    (Legal mc fc board line after →
      line.Nodup ∧ (emptyCount after.val).toNat + line.length = (emptyCount board.val).toNat ∧
      ∀ square ∈ line, vacant board square)

theorem replayContract_proven : ReplayContract := by
  intro mc fc board line after
  exact ⟨⟨replay_sound mc fc board line after, replay_complete⟩, line_stats⟩

def InstallContract : Prop :=
  ∀ (mc : X86Config) (fc : FlipConfig) (board after : Position)
    (square : GeneratedSquare mc board.val) (child current : Buffer),
    (emptyCount board.val).toNat ≤ 60 → current.length = capacity →
    Legal mc fc (play mc fc board square).val (visible child) after →
    let next := prepend square.val child current
    60 - (emptyCount board.val).toNat + 1 < capacity ∧
    visible next = square.val :: visible child ∧ Legal mc fc board (visible next) after

theorem installContract_proven : InstallContract := by
  intro mc fc board after square child current boardLimit capacityFits checked next
  have delta := move_count mc fc board square
  have childCount := (line_stats checked).2.1
  have bounded : (visible child).length ≤ capacity - 2 := by unfold capacity; omega
  have observed : visible next = square.val :: visible child :=
    (bufferContract_proven square.val child current capacityFits).2.2.2.2 bounded
  exact ⟨by unfold capacity; omega, observed, observed.symm ▸ .move square checked⟩

private def passFixture : Position :=
  ⟨⟨0x0100000000000001#64, 0x0200000000000002#64⟩, by unfold Board.wellFormed; decide⟩

example :
    let mc : X86Config := ⟨false, false⟩
    let fc : FlipConfig := ⟨false, false, false⟩
    let first : Square := ⟨2, by decide⟩
    let second : Square := ⟨58, by decide⟩
    (replay mc fc passFixture [first, second]).isSome = true ∧
    (replay mc fc passFixture [first, first]).isSome = false ∧
    (replay mc fc passFixture [first, second, ⟨8, by decide⟩]).isSome = false := by decide +kernel

end NeuralReversi.PrincipalVariation

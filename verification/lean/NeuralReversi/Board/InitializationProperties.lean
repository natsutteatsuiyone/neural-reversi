import NeuralReversi.Board.InitializationModel
import NeuralReversi.Board.CellProperties
import NeuralReversi.Board.Counting
import NeuralReversi.Bitboard.RepresentationProperties

namespace NeuralReversi.Board
open Bitboard

def startingCell (row file : Nat) : Disc.Disc :=
  if (row = 3 ∧ file = 4) ∨ (row = 4 ∧ file = 3) then .black
  else if (row = 3 ∧ file = 3) ∨ (row = 4 ∧ file = 4) then .white
  else .empty

theorem defaultBoard_cells : ∀ i : Fin 64,
    cellDisc defaultBoard i.val .black = startingCell (i.val / 8) (i.val % 8) := by
  decide

def InitialBoardContract : Prop :=
  ∀ b ∈ [defaultBoard, newBoard],
    wellFormed b ∧ (playerCount b).toNat = 2 ∧ (opponentCount b).toNat = 2 ∧
    (emptyCount b).toNat = 60 ∧
    ∀ (sq : Bitboard.Square) (checked : Bool),
      discAt checked b ⟨sq.val, by omega⟩ .black =
        .ok (startingCell (sq.val / 8) (sq.val % 8))

theorem initialBoardContract_proven : InitialBoardContract := by
  intro b member
  have equal : b = defaultBoard := by simpa only [List.mem_cons, List.not_mem_nil,
    or_false, newBoard, or_self] using member
  subst b
  refine ⟨?_, by decide, by decide, by decide, ?_⟩
  · unfold wellFormed
    decide
  · intro sq checked
    rw [discAt_playable, defaultBoard_cells]

def BoardObservationContract : Prop :=
  ∀ b : Board,
    bits (getPlayer b) = b.player ∧ intoU64 (getOpponent b) = b.opponent ∧
    (⟨intoU64 (getPlayer b), intoU64 (getOpponent b)⟩ : Board) = b ∧
    (valueCount (getPlayer b)).toNat = (playerCount b).toNat ∧
    (valueCount (getOpponent b)).toNat = (opponentCount b).toNat ∧
    ∀ (sq : Bitboard.Square) (checked : Bool),
      valueContains checked (getPlayer b) ⟨sq.val, by omega⟩ = .ok (b.player.toNat.testBit sq.val) ∧
      valueContains checked (getOpponent b) ⟨sq.val, by omega⟩ = .ok (b.opponent.toNat.testBit sq.val)

theorem boardObservationContract_proven : BoardObservationContract := by
  intro b
  refine ⟨rfl, rfl, ?_, rfl, rfl, ?_⟩
  · cases b; rfl
  · intro sq checked
    exact ⟨squareMembershipContract_proven b.player sq checked checked,
      squareMembershipContract_proven b.opponent sq checked checked⟩

end NeuralReversi.Board

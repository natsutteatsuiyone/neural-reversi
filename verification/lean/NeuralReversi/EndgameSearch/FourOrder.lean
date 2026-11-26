import NeuralReversi.EndgameSearch.MoveOrder
import NeuralReversi.Empties

namespace NeuralReversi.EndgameSearch

open PvsBoardTree (Position minimax)
open Bitboard.Moves (X86Config)
open EndgameTreeCache (Address Store Table soundHistory)
open ScoreRefresh (Score fromDiscDiff toDiscDiff)
open Flip.Dispatch (TargetConfig)

private def presortedEmpties (b : Board.Board) : List Bitboard.Square :=
  Empties.presorted.filter fun sq => (Board.emptySquares b).getLsbD sq.val

private theorem presorted_perm (b : Position) :
    (presortedEmpties b.val).Perm (Solve34.EmptyBoard.ofBoard b.val b.property).squares :=
  Empties.presorted_perm.filter _

private theorem exactlyFour {α : Type} (xs : List α) (h : xs.length = 4) :
    ∃ a b c d, xs = [a, b, c, d] := by
  match xs with
  | [] | [_] | [_, _] | [_, _, _] => simp at h
  | [a, b, c, d] => exact ⟨a, b, c, d, rfl⟩
  | _ :: _ :: _ :: _ :: _ :: _ => simp at h

/-- The empty squares in odd quadrants first, each group in presorted order. -/
private def sortedEmpties (b : Board.Board) : List Bitboard.Square :=
  let xs := presortedEmpties b
  xs.filter (Empties.oddIn xs) ++ xs.filter (fun x => !Empties.oddIn xs x)

def presortedFourOrder : FourBoards := fun b four => by
  let board := Solve34.EmptyBoard.ofBoard b.val b.property
  let xs := sortedEmpties b.val
  have perm : xs.Perm board.squares := (List.filter_append_perm _ _).trans (presorted_perm b)
  refine ⟨⟨board.player, board.opponent, xs, perm.length_eq.trans (board.length_eq.trans four),
    perm.symm.nodup board.nodup, board.disjoint, ?_⟩, rfl⟩
  rw [board.empty_exact]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  apply Bool.eq_iff_iff.mpr
  rw [Solve34.emptyMask_bit _ ⟨i, hi⟩, Solve34.emptyMask_bit _ ⟨i, hi⟩]
  exact perm.mem_iff.symm

/-- `sort_last4` on the presorted empty squares of a four-empty board. -/
theorem presortedFourOrder_reads (b : Position) (four : (Board.emptyCount b.val).toNat = 4)
    (s : Empties.State) (mask : Empties.Encodes s.mask (Board.emptySquares b.val)) :
    Empties.sortLastFour s (Empties.oddQuadrants (Board.emptySquares b.val)) =
      (presortedFourOrder b four).val.squares := by
  have perm := presorted_perm b
  have len := perm.length_eq.trans (Solve34.EmptyBoard.ofBoard b.val b.property).length_eq
  obtain ⟨a, c, d, e, shape⟩ := exactlyFour _ (len.trans four)
  have exact := Empties.exactly_presorted (Board.emptySquares b.val)
  have unique := Empties.presorted_filter_nodup (Board.emptySquares b.val)
  change presortedEmpties b.val = _ at shape
  unfold presortedEmpties at shape
  rw [shape] at exact unique
  rw [Empties.sortLastFour_partition s _ a c d e mask exact unique shape]
  change _ = sortedEmpties b.val
  unfold sortedEmpties presortedEmpties
  rw [shape]

def orderedOptions (target : TargetConfig) (bmi2 : Bool) (mc : X86Config)
    (fc : FlipConfig) (address : Address) : Options mc :=
  withSourceFour target bmi2 ⟨address, orderedMoves mc fc, none⟩ presortedFourOrder

theorem orderedOptions_correct (target : TargetConfig) (bmi2 : Bool) (mc : X86Config)
    (fc : FlipConfig) (address : Address) :
    CorrectOptions mc fc (orderedOptions target bmi2 mc fc address) :=
  withSourceFour_correct target bmi2 mc fc _ presortedFourOrder (orderedMoves_members mc fc)

def OrderedContract : Prop :=
  ∀ (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : FlipConfig)
    (address : Address) (table : Table)
    (b : Position) (alpha beta : Score) (history : List Store) (checked : Bool),
    soundHistory mc fc history → beta.raw.toInt = alpha.raw.toInt + 1 →
    let result := run mc fc (orderedOptions target bmi2 mc fc address) table b (toDiscDiff alpha).raw.toInt history
    ∃ score, fromDiscDiff checked ⟨BitVec.ofInt 32 result.score⟩ = .ok score ∧
      Aspiration.SoundResult (minimax mc fc b) alpha.raw.toInt beta.raw.toInt score.raw.toInt ∧
      soundHistory mc fc result.history ∧ history <+: result.history

theorem orderedContract_proven : OrderedContract := by
  intro target bmi2 mc fc address table b alpha beta history checked sound window
  exact sourceFourContract_proven target bmi2 mc fc ⟨address, orderedMoves mc fc, none⟩ presortedFourOrder
    table b alpha beta history checked (orderedMoves_members mc fc) sound window

end NeuralReversi.EndgameSearch

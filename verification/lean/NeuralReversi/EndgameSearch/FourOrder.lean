import NeuralReversi.EndgameSearch.MoveOrder
import NeuralReversi.SortLastFour

namespace NeuralReversi.EndgameSearch

open PvsBoardTree (Position minimax)
open Bitboard.Moves (X86Config)
open EndgameTreeCache (Address Store Table soundHistory)
open ScoreRefresh (Score fromDiscDiff toDiscDiff)
open Flip.Dispatch (TargetConfig)
open EmptyListLinks (Links head slot)

private def presortedEmpties (b : Board.Board) : List Bitboard.Square :=
  EmptyListParity.presorted.filter fun sq => (Board.emptySquares b).getLsbD sq.val

private theorem presorted_perm (b : Position) :
    (presortedEmpties b.val).Perm (Solve34.EmptyBoard.ofBoard b.val b.property).squares :=
  EmptyListParity.scan_perm.filter _

private theorem exactlyFour {α : Type} (xs : List α) (h : xs.length = 4) :
    ∃ a b c d, xs = [a, b, c, d] := by
  match xs with
  | [] | [_] | [_, _] | [_, _, _] => simp at h
  | [a, b, c, d] => exact ⟨a, b, c, d, rfl⟩
  | _ :: _ :: _ :: _ :: _ :: _ => simp at h

private def sortOnBoard (b : Board.Board) : Option (List Bitboard.Square) :=
  SortLastFour.sortLastFour (EmptyListInit.newLinks (Board.emptySquares b))
    (EmptyListParity.fromEmpties (Board.emptySquares b))

private theorem sortOnBoard_perm (b : Position) (four : (Board.emptyCount b.val).toNat = 4) :
    ∃ xs, sortOnBoard b.val = some xs ∧ xs.Perm (Solve34.EmptyBoard.ofBoard b.val b.property).squares := by
  have same := presorted_perm b
  have len := same.length_eq.trans (Solve34.EmptyBoard.ofBoard b.val b.property).length_eq
  obtain ⟨a, c, d, e, shape⟩ := exactlyFour _ (len.trans four)
  have decoded := SortLastFour.sort_initialized (Board.emptySquares b.val) a c d e shape
  refine ⟨_, decoded, ?_⟩
  exact (List.filter_append_perm _ _).trans (shape ▸ same)

private def sortedEmpties (b : Position) (four : (Board.emptyCount b.val).toNat = 4) :
    List Bitboard.Square :=
  (sortOnBoard b.val).get (by obtain ⟨_, found, _⟩ := sortOnBoard_perm b four; simp [found])

private theorem sortOnBoard_reads (b : Position) (four : (Board.emptyCount b.val).toNat = 4) :
    sortOnBoard b.val = some (sortedEmpties b four) := by
  obtain ⟨xs, found, _⟩ := sortOnBoard_perm b four
  simp [sortedEmpties, found]

def linkedFourOrder : FourBoards := fun b four => by
  let board := Solve34.EmptyBoard.ofBoard b.val b.property
  let xs := sortedEmpties b four
  have perm : xs.Perm board.squares := by
    obtain ⟨_, found, same⟩ := sortOnBoard_perm b four
    simpa [xs, sortedEmpties, found, board] using same
  refine ⟨⟨board.player, board.opponent, xs, perm.length_eq.trans (board.length_eq.trans four),
    perm.symm.nodup board.nodup, board.disjoint, ?_⟩, rfl⟩
  rw [board.empty_exact]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  apply Bool.eq_iff_iff.mpr
  rw [Solve34.emptyMask_bit _ ⟨i, hi⟩, Solve34.emptyMask_bit _ ⟨i, hi⟩]
  exact perm.mem_iff.symm

theorem linkedFourOrder_reads (b : Position) (four : (Board.emptyCount b.val).toNat = 4)
    (nodes : Links) (byte : BitVec 8) (cells : List Bitboard.Square)
    (links : EmptyListInit.chain nodes head (presortedEmpties b.val |>.map slot))
    (state : EmptyListParity.Reachable cells byte)
    (same : cells.Perm (Solve34.EmptyBoard.ofBoard b.val b.property).squares) :
    SortLastFour.sortLastFour nodes byte = some (linkedFourOrder b four).val.squares := by
  have perm := presorted_perm b
  have len := perm.length_eq.trans (Solve34.EmptyBoard.ofBoard b.val b.property).length_eq
  obtain ⟨a, c, d, e, shape⟩ := exactlyFour _ (len.trans four)
  rw [shape] at links
  have equivalent : cells.Perm [a, c, d, e] := shape ▸ same.trans perm.symm
  have actual := SortLastFour.orderContract_proven nodes byte cells a c d e links state equivalent
  have rebuilt := SortLastFour.sort_initialized (Board.emptySquares b.val) a c d e shape
  exact actual.trans (rebuilt.symm.trans (sortOnBoard_reads b four))

def orderedOptions (target : TargetConfig) (bmi2 : Bool) (mc : X86Config)
    (fc : FlipConfig) (address : Address) : Options mc :=
  withSourceFour target bmi2 ⟨address, orderedMoves mc fc, none⟩ linkedFourOrder

theorem orderedOptions_correct (target : TargetConfig) (bmi2 : Bool) (mc : X86Config)
    (fc : FlipConfig) (address : Address) :
    CorrectOptions mc fc (orderedOptions target bmi2 mc fc address) :=
  withSourceFour_correct target bmi2 mc fc _ linkedFourOrder (orderedMoves_members mc fc)

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
  exact sourceFourContract_proven target bmi2 mc fc ⟨address, orderedMoves mc fc, none⟩ linkedFourOrder
    table b alpha beta history checked (orderedMoves_members mc fc) sound window

end NeuralReversi.EndgameSearch

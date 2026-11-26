import NeuralReversi.EndgameSearch.Direct
import NeuralReversi.EndgameSearch.EmptyState
import NeuralReversi.EmptyListTraversal

namespace NeuralReversi.EndgameSearch.Direct

open EmptyListLinks (Links Index slot head)
open EmptyListInit (chain)
open PvsBoardTree (Position)
open Bitboard (Square)
open EmptyListTraversal (squareAfter)

def readSquares (nodes : Links) (atNode : Index) : Nat → Except String (List Square)
  | 0 => .ok []
  | n + 1 =>
    match squareAfter nodes atNode with
    | .error err => .error err
    | .ok square => (readSquares nodes (slot square) n).map (square :: ·)

private theorem readChain (nodes : Links) (atNode : Index) (squares : List Square)
    (path : chain nodes atNode (squares.map slot)) :
    readSquares nodes atNode squares.length = .ok squares := by
  induction squares generalizing atNode with
  | nil => rfl
  | cons sq rest ih =>
    have next := ((EmptyListTraversal.successorContract_proven.1 nodes atNode).2 sq).mpr path.1.1
    simp [readSquares, next, ih (slot sq) path.2, Except.map]

theorem linkedArgs (b : Position) (aligned : EmptyStateMatches b nodes parity count)
    (size : count.toNat = n) :
    ∃ args : Solve34.EmptyBoard n, args.asBoard = b.val ∧
      readSquares nodes head n = .ok args.squares := by
  let board := Solve34.EmptyBoard.ofBoard b.val b.property
  let squares := EmptyListParity.presorted.filter (fun sq => (Board.emptySquares b.val).getLsbD sq.val)
  have perm : squares.Perm board.squares := EmptyListParity.scan_perm.filter _
  have length := perm.length_eq.trans (board.length_eq.trans (aligned.1 ▸ size))
  have covered : ~~~(board.player ||| board.opponent) = Solve34.emptyMask squares := by
    rw [board.empty_exact]
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    apply Bool.eq_iff_iff.mpr
    rw [Solve34.emptyMask_bit _ ⟨i, hi⟩, Solve34.emptyMask_bit _ ⟨i, hi⟩]
    exact perm.mem_iff.symm
  refine ⟨⟨board.player, board.opponent, squares, length, perm.symm.nodup board.nodup,
    board.disjoint, covered⟩, rfl, ?_⟩
  exact length ▸ readChain nodes head squares aligned.2.1

theorem oneArg (b : Position) (aligned : EmptyStateMatches b nodes parity count) (size : count.toNat = 1) :
    ∃ leaf : Solve1.Bmi2.Board, leaf.asBoard = b.val ∧ squareAfter nodes head = .ok leaf.square := by
  obtain ⟨args, same, reads⟩ := linkedArgs b aligned size
  have len := args.length_eq
  obtain ⟨sq, shape⟩ : ∃ sq, args.squares = [sq] := by
    match e : args.squares with
    | [] | _ :: _ :: _ => simp [e] at len
    | [sq] => exact ⟨sq, rfl⟩
  have empty := (Solve34.empty_bits args sq).mpr (by simp [shape])
  let leaf : Solve1.Bmi2.Board := ⟨args.player, sq, empty.1⟩
  have occupied : ~~~(args.player ||| args.opponent) = Bitboard.squareMask sq := by
    simpa [shape, Solve34.emptyMask] using args.empty_exact
  have opposite : ~~~(args.player ||| Bitboard.squareMask sq) = args.opponent := by
    rw [← occupied]
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    have disjoint := congrArg (fun bits : BitVec 64 => bits.getLsbD i) args.disjoint
    simp only [BitVec.getLsbD_and] at disjoint
    simp only [BitVec.getLsbD_not, BitVec.getLsbD_or, hi, decide_true, Bool.true_and]
    cases hp : args.player.getLsbD i <;> cases ho : args.opponent.getLsbD i <;>
      simp_all [-BitVec.getLsbD_eq_getElem]
  refine ⟨leaf, ?_, ?_⟩
  · rw [← same]
    exact (Board.Board.mk.injEq ..).mpr ⟨rfl, opposite⟩
  · rw [shape] at reads
    cases first : squareAfter nodes head with
    | error err => simp [readSquares, first] at reads
    | ok i =>
      have equal : i = sq := by simpa [readSquares, first, Except.map] using reads
      rw [equal]

theorem fourArgs (b : Position) (aligned : EmptyStateMatches b nodes parity count) (size : count.toNat = 4) :
    ∃ args : Solve34.EmptyBoard 4, args.asBoard = b.val ∧
      SortLastFour.sortLastFour nodes parity = some args.squares := by
  have four : (Board.emptyCount b.val).toNat = 4 := aligned.1 ▸ size
  let args := linkedFourOrder b four
  exact ⟨args.val, args.property, linkedFourOrder_reads b four nodes parity _ aligned.2.1 aligned.2.2 .rfl⟩

theorem fixedOutputs (target : Flip.Dispatch.TargetConfig) (bmi2 : Bool)
    (b : Position) (alpha : BitVec 32) (aligned : EmptyStateMatches b nodes parity count) :
    (count.toNat = 1 → ∃ arg : Solve1.Bmi2.Board,
      squareAfter nodes head = .ok arg.square ∧ arg.player = b.val.player ∧
      FixedDisc target bmi2 b alpha (Solve2.solve1On target bmi2 arg alpha)) ∧
    (count.toNat = 2 → ∃ arg : Solve34.EmptyBoard 2,
      readSquares nodes head 2 = .ok arg.squares ∧ arg.asBoard = b.val ∧
      FixedDisc target bmi2 b alpha (Solve2.targetRun target bmi2 arg.toTwoEmpty alpha).score) ∧
    (count.toNat = 3 → ∃ arg : Solve34.EmptyBoard 3,
      readSquares nodes head 3 = .ok arg.squares ∧ arg.asBoard = b.val ∧
      FixedDisc target bmi2 b alpha (Solve34.target3Run target bmi2 arg alpha).score) ∧
    (count.toNat = 4 → ∃ arg : Solve34.EmptyBoard 4,
      SortLastFour.sortLastFour nodes parity = some arg.squares ∧ arg.asBoard = b.val ∧
      FixedDisc target bmi2 b alpha (Solve34.target4Run target bmi2 arg alpha).score) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> intro size
  · obtain ⟨arg, board, read⟩ := oneArg b aligned size
    exact ⟨arg, read, congrArg Board.Board.player board, .one arg board⟩
  · obtain ⟨arg, board, read⟩ := linkedArgs b aligned size
    exact ⟨arg, read, board, .two arg board⟩
  · obtain ⟨arg, board, read⟩ := linkedArgs b aligned size
    exact ⟨arg, read, board, .three arg board⟩
  · obtain ⟨arg, board, read⟩ := fourArgs b aligned size
    exact ⟨arg, read, board, .four arg board⟩

end NeuralReversi.EndgameSearch.Direct


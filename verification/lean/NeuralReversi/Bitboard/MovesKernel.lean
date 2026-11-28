module
public import NeuralReversi.Bitboard.MovesDirectional

@[expose] public section

namespace NeuralReversi.Bitboard.Moves

theorem shifted_preMask (h : BitVec w) (d : Nat) :
    (h &&& (h <<< d)) >>> d = h &&& (h >>> d) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_ushiftRight, BitVec.getLsbD_and, BitVec.getLsbD_shiftLeft]
  by_cases bound : d + i < w
  · have sub : d + i - d = i := by omega
    have lower : ¬ d + i < d := by omega
    simp only [bound, decide_true, lower, decide_false, Bool.not_false,
      Bool.true_and, sub, Bool.and_comm]
  · have zero : h.getLsbD (d + i) = false := BitVec.getLsbD_of_ge h (d + i) (by omega)
    simp only [zero, Bool.false_and, Bool.and_false]

def portableDirections (p o : Bits) : Bits :=
  let h := o &&& horizontalMask
  (h + (h &&& (p <<< (1 : Nat)))) |||
  ((expandedLeft h p 7) <<< (7 : Nat)) |||
  ((expandedLeft h p 9) <<< (9 : Nat)) |||
  ((expandedLeft o p 8) <<< (8 : Nat)) |||
  ((expandedRight h p 7) >>> (7 : Nat)) |||
  ((expandedRight h p 9) >>> (9 : Nat)) |||
  ((expandedRight o p 8) >>> (8 : Nat)) |||
  ((expandedRight h p 1) >>> (1 : Nat))

theorem portable_decomposition (p o : Bits) :
    getMovesPortable p o = portableDirections p o &&& ~~~(p ||| o) := by
  simp only [getMovesPortable, portableDirections, expandedLeft, expandedRight,
    shifted_preMask, BitVec.or_assoc]

end NeuralReversi.Bitboard.Moves

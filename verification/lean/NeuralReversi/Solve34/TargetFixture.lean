import NeuralReversi.Solve34.Target

namespace NeuralReversi.Solve34

private def board (p o : Word) (wf : p &&& o = 0)
    (h : (Board.emptyCount ⟨p, o⟩).toNat = n) : EmptyBoard n :=
  h ▸ EmptyBoard.ofBoard ⟨p, o⟩ wf

private def two : EmptyBoard 2 :=
  board 0xf9c1a59185859fbc 0x063e5a6e7a7a6040 (by decide) (by decide)

private def three : EmptyBoard 3 :=
  board 0xfe547864441e0c00 0x01ab879bbbe1f1f9 (by decide) (by decide)

private def four : EmptyBoard 4 :=
  board 0x1c1c0422242a1c20 0xe0e1fbdddbd5e3cf (by decide) (by decide)

private def fourPass : EmptyBoard 4 :=
  board 0x005f61595fdfdf7f 0xfca09ea620202000 (by decide) (by decide)

private def doublePass : EmptyBoard 4 :=
  board 0xfffffffffffffff0 0 (by decide) (by decide)

private def observe (run : Score → Dispatch.Outcome) (alpha : Score) : Int × Nat :=
  ((run alpha).score.toInt, (run alpha).nodes)

set_option maxRecDepth 8192 in
set_option maxHeartbeats 4000000 in
example :
    ((Solve2.sourceFallback two.toTwoEmpty (-65)).score.toInt,
      (Solve2.sourceFallback two.toTwoEmpty (-65)).nodeDelta) = (-12, 4) ∧
    ((Solve2.sourceFallback two.toTwoEmpty 64).score.toInt,
      (Solve2.sourceFallback two.toTwoEmpty 64).nodeDelta) = (-2, 3) ∧
    observe (source3Run three) (-65) = (-26, 8) ∧
    observe (source3Run three) 64 = (-22, 5) ∧
    observe (source4Run four) (-65) = (-40, 9) ∧
    observe (source4Run four) 64 = (-30, 4) ∧
    observe (source4Run fourPass) (-65) = (2, 28) ∧
    observe (source4Run fourPass) 64 = (6, 5) ∧
    observe (source4Run doublePass) (-65) = (64, 0) ∧
    observe (source4Run doublePass) 64 = (64, 0) := by decide +kernel

end NeuralReversi.Solve34

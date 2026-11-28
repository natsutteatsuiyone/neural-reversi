module
public import NeuralReversi.Bitboard.Model

@[expose] public section

/-! Depth-one visit tiers extracted from the rustc MIR of `search::midgame::DEPTH1_MOVE_ORDER_MASKS` by
`verification/lean/scripts/check_midgame_children_mir.py`. Regenerate with `check_endgame_mir.py --write`
instead of editing. -/

namespace NeuralReversi.PvsParallel.SerialTable.MidgameShallow

def depthOneMasks : List Bitboard.Bits :=
  [0x8100_0000_0000_0081, 0x2400_8100_0081_0024, 0x0000_2400_0024_0000, 0x1800_0081_8100_0018, 0x0000_1824_2418_0000,
   0x0018_0042_4200_1800, 0x0024_4200_0042_2400, 0x4281_0000_0000_8142, 0x0042_0000_0000_4200, 0x0000_0018_1800_0000]

end NeuralReversi.PvsParallel.SerialTable.MidgameShallow

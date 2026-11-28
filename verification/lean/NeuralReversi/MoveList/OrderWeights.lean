module

@[expose] public section

/-! Move-ordering table and weights extracted from the rustc MIR of `move_list::ordering` by
`verification/lean/scripts/extract_endgame_order_mir.py`. Regenerate with
`check_endgame_mir.py --write` instead of editing. -/

namespace NeuralReversi.MoveList.OrderWeights

def squareValue : Vector Int 64 := #v[18, 4, 16, 12, 12, 16, 4, 18, 4, 2, 6, 8, 8, 6, 2, 4, 16, 6, 14, 10, 10, 14, 6, 16, 12, 8, 10, 0, 0, 10, 8, 12, 12, 8, 10, 0, 0, 10, 8, 12, 16, 6, 14, 10, 10, 14, 6, 16, 4, 2, 6, 8, 8, 6, 2, 4, 18, 4, 16, 12, 12, 16, 4, 18]
def squareMax : Int := 18
def squareWeight : Int := 128
def stabilityWeight : Int := 2048
def mobilityWeight : Int := 16384
def searchStabilityWeight : Int := 256
def searchMobilityWeight : Int := 480
def searchPotentialWeight : Int := 256

end NeuralReversi.MoveList.OrderWeights

module

@[expose] public section

/-! Endgame search thresholds extracted from the rustc MIR of `search::endgame` by
`verification/lean/scripts/extract_deep_recursion_mir.py`. Regenerate with
`check_endgame_mir.py --write` instead of editing. -/

namespace NeuralReversi.EndgameSearch.Thresholds

/-- `MIN_EC_ETC_EMPTIES`: the fewest empties at which the deep search tries enhanced cutoffs. -/
def minEcEtcEmpties : Nat := 12

end NeuralReversi.EndgameSearch.Thresholds

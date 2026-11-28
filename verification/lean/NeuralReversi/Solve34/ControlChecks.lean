module
public import NeuralReversi.Solve34.Control

@[expose] public section

namespace NeuralReversi.Solve34

theorem solve3_early_bound_not_exact :
    sourceMax 0 true (-127) [some 2,some 10,some 12] 0 = 2 ∧
    (foldMax none [some 2,some 10,some 12]).getD 0 = 12 := by decide

theorem solve4_pass_cut_not_exact :
    sourceMin 0 true 127 [some (-2),some (-10),none,some (-12)] 0 = -2 ∧
    (foldMin none [some (-2),some (-10),none,some (-12)]).getD 0 = -12 := by decide

theorem sentinel_collision_control :
    sourceMax 0 true (-127) [some (-127),none,none] 7 = 7 ∧
    (maxRun 0 none [some (-127),none,none]).getD 7 = -127 := by decide

theorem final_slot_sentinel_control :
    sourceMax 0 true (-127) [none,none,some (-127)] 7 = -127 ∧
    sourceMin 0 true 127 [none,none,none,some 127] 7 = 127 := by decide

theorem general_child_need_not_be_oneEmpty :
    Solve2.NWS 0 10 2 ∧ ¬Solve2.OneEmpty 0 10 2 := by
  simp [Solve2.NWS,Solve2.OneEmpty]

end NeuralReversi.Solve34

module
public import NeuralReversi.ScoreRefresh.Properties

@[expose] public section

namespace NeuralReversi.ScoreRefresh

theorem addScore_bridge (checked : Bool) (a b : Score) :
    (addScore checked a b).map legacy = ScaledScore.addScore checked (legacy a) (legacy b) := by
  unfold addScore ScaledScore.addScore addWords ScaledScore.addWords legacy
  split <;> rfl

theorem subScore_bridge (checked : Bool) (a b : Score) :
    (subScore checked a b).map legacy = ScaledScore.subScore checked (legacy a) (legacy b) := by
  unfold subScore ScaledScore.subScore subWords ScaledScore.subWords legacy
  split <;> rfl

theorem negate_bridge (checked : Bool) (a : Score) :
    (negate checked a).map legacy = ScaledScore.negate checked (legacy a) := by
  unfold negate ScaledScore.negate  legacy
  split <;> rfl

theorem addInt_bridge (checked : Bool) (a : Score) (b : Word) :
    (addInt checked a b).map legacy = ScaledScore.addInt checked (legacy a) b := by
  unfold addInt ScaledScore.addInt addWords ScaledScore.addWords legacy
  split <;> rfl

theorem subInt_bridge (checked : Bool) (a : Score) (b : Word) :
    (subInt checked a b).map legacy = ScaledScore.subInt checked (legacy a) b := by
  unfold subInt ScaledScore.subInt subWords ScaledScore.subWords legacy
  split <;> rfl

def legacyAssignment (out : Except String Unit × Score) :
    Except String Unit × ScaledScore.ScaledScore := (out.1, legacy out.2)

theorem assignResult_bridge (before : Score) (result : Except String Score) :
    legacyAssignment (assignResult before result) =
      ScaledScore.assignResult (legacy before) (result.map legacy) := by
  cases result <;> rfl

theorem addAssign_bridge (checked : Bool) (a b : Score) :
    legacyAssignment (addAssign checked a b) = ScaledScore.addAssign checked (legacy a) (legacy b) := by
  unfold addAssign ScaledScore.addAssign
  rw [assignResult_bridge, addScore_bridge]

theorem subAssign_bridge (checked : Bool) (a b : Score) :
    legacyAssignment (subAssign checked a b) = ScaledScore.subAssign checked (legacy a) (legacy b) := by
  unfold subAssign ScaledScore.subAssign
  rw [assignResult_bridge, subScore_bridge]

theorem constants_contract :
    discMax.raw.toInt = 64 ∧ discMin.raw.toInt = -64 ∧ discInf.raw.toInt = 127 ∧
    scoreMax.raw.toInt = 16384 ∧ scoreMin.raw.toInt = -16384 ∧
    scoreInf.raw.toInt = 32767 ∧
    -scoreInf.raw.toInt < scoreMin.raw.toInt ∧ scoreMax.raw.toInt < scoreInf.raw.toInt := by decide

theorem disc_display_forward {Result : Type} (score : DiscScore)
    (options : ScaledScore.FormatOptions)
    (formatI32 : Word → ScaledScore.FormatOptions → Result) :
    DiscScore.display score options formatI32 = formatI32 score.raw options := rfl

end NeuralReversi.ScoreRefresh

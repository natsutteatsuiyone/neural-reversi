import NeuralReversi.Stability.Horizontal
import NeuralReversi.Stability.HorizontalProofs.Group3
namespace NeuralReversi.Stability
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
theorem full_0_32 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 32 = lineBits disc 0 32 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_33 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 33 = lineBits disc 0 33 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_34 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 34 = lineBits disc 0 34 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_35 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 35 = lineBits disc 0 35 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_36 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 36 = lineBits disc 0 36 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_37 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 37 = lineBits disc 0 37 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_38 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 38 = lineBits disc 0 38 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_39 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 39 = lineBits disc 0 39 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

end NeuralReversi.Stability

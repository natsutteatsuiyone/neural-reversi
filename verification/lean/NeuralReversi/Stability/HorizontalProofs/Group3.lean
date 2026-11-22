import NeuralReversi.Stability.Horizontal
import NeuralReversi.Stability.HorizontalProofs.Group2
namespace NeuralReversi.Stability
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
theorem full_0_24 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 24 = lineBits disc 0 24 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_25 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 25 = lineBits disc 0 25 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_26 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 26 = lineBits disc 0 26 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_27 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 27 = lineBits disc 0 27 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_28 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 28 = lineBits disc 0 28 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_29 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 29 = lineBits disc 0 29 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_30 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 30 = lineBits disc 0 30 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_31 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 31 = lineBits disc 0 31 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

end NeuralReversi.Stability

import NeuralReversi.Stability.Horizontal
import NeuralReversi.Stability.HorizontalProofs.Group5
namespace NeuralReversi.Stability
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
theorem full_0_48 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 48 = lineBits disc 0 48 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_49 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 49 = lineBits disc 0 49 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_50 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 50 = lineBits disc 0 50 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_51 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 51 = lineBits disc 0 51 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_52 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 52 = lineBits disc 0 52 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_53 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 53 = lineBits disc 0 53 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_54 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 54 = lineBits disc 0 54 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_55 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 55 = lineBits disc 0 55 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

end NeuralReversi.Stability

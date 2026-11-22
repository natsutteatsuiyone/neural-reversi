import NeuralReversi.Stability.Horizontal
namespace NeuralReversi.Stability
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
theorem full_0_1 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 1 = lineBits disc 0 1 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_2 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 2 = lineBits disc 0 2 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_3 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 3 = lineBits disc 0 3 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_4 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 4 = lineBits disc 0 4 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_5 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 5 = lineBits disc 0 5 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_6 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 6 = lineBits disc 0 6 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_0_7 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 7 = lineBits disc 0 7 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

end NeuralReversi.Stability

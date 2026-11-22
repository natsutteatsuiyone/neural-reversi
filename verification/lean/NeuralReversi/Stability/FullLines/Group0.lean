import NeuralReversi.Stability.Geometry
namespace NeuralReversi.Stability
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
theorem full_1_0 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 0 = lineBits disc 1 0 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_0 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 0 = lineBits disc 2 0 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_0 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 0 = lineBits disc 3 0 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_1 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 1 = lineBits disc 1 1 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_1 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 1 = lineBits disc 2 1 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_1 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 1 = lineBits disc 3 1 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_2 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 2 = lineBits disc 1 2 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_2 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 2 = lineBits disc 2 2 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_2 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 2 = lineBits disc 3 2 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_3 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 3 = lineBits disc 1 3 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_3 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 3 = lineBits disc 2 3 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_3 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 3 = lineBits disc 3 3 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_4 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 4 = lineBits disc 1 4 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_4 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 4 = lineBits disc 2 4 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_4 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 4 = lineBits disc 3 4 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_5 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 5 = lineBits disc 1 5 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_5 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 5 = lineBits disc 2 5 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_5 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 5 = lineBits disc 3 5 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_6 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 6 = lineBits disc 1 6 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_6 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 6 = lineBits disc 2 6 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_6 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 6 = lineBits disc 3 6 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_7 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 7 = lineBits disc 1 7 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_7 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 7 = lineBits disc 2 7 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_7 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 7 = lineBits disc 3 7 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

end NeuralReversi.Stability

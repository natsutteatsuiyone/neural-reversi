import NeuralReversi.Stability.Geometry
import NeuralReversi.Stability.FullLines.Group2
namespace NeuralReversi.Stability
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
theorem full_1_24 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 24 = lineBits disc 1 24 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_24 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 24 = lineBits disc 2 24 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_24 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 24 = lineBits disc 3 24 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_25 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 25 = lineBits disc 1 25 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_25 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 25 = lineBits disc 2 25 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_25 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 25 = lineBits disc 3 25 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_26 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 26 = lineBits disc 1 26 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_26 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 26 = lineBits disc 2 26 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_26 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 26 = lineBits disc 3 26 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_27 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 27 = lineBits disc 1 27 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_27 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 27 = lineBits disc 2 27 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_27 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 27 = lineBits disc 3 27 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_28 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 28 = lineBits disc 1 28 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_28 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 28 = lineBits disc 2 28 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_28 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 28 = lineBits disc 3 28 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_29 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 29 = lineBits disc 1 29 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_29 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 29 = lineBits disc 2 29 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_29 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 29 = lineBits disc 3 29 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_30 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 30 = lineBits disc 1 30 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_30 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 30 = lineBits disc 2 30 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_30 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 30 = lineBits disc 3 30 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_31 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 31 = lineBits disc 1 31 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_31 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 31 = lineBits disc 2 31 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_31 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 31 = lineBits disc 3 31 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

end NeuralReversi.Stability

import NeuralReversi.Stability.Geometry
import NeuralReversi.Stability.FullLines.Group3
namespace NeuralReversi.Stability
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
theorem full_1_32 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 32 = lineBits disc 1 32 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_32 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 32 = lineBits disc 2 32 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_32 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 32 = lineBits disc 3 32 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_33 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 33 = lineBits disc 1 33 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_33 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 33 = lineBits disc 2 33 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_33 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 33 = lineBits disc 3 33 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_34 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 34 = lineBits disc 1 34 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_34 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 34 = lineBits disc 2 34 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_34 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 34 = lineBits disc 3 34 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_35 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 35 = lineBits disc 1 35 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_35 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 35 = lineBits disc 2 35 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_35 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 35 = lineBits disc 3 35 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_36 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 36 = lineBits disc 1 36 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_36 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 36 = lineBits disc 2 36 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_36 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 36 = lineBits disc 3 36 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_37 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 37 = lineBits disc 1 37 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_37 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 37 = lineBits disc 2 37 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_37 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 37 = lineBits disc 3 37 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_38 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 38 = lineBits disc 1 38 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_38 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 38 = lineBits disc 2 38 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_38 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 38 = lineBits disc 3 38 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_39 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 39 = lineBits disc 1 39 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_39 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 39 = lineBits disc 2 39 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_39 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 39 = lineBits disc 3 39 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

end NeuralReversi.Stability

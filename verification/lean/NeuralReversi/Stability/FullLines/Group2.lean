import NeuralReversi.Stability.Geometry
import NeuralReversi.Stability.FullLines.Group1
namespace NeuralReversi.Stability
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
theorem full_1_16 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 16 = lineBits disc 1 16 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_16 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 16 = lineBits disc 2 16 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_16 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 16 = lineBits disc 3 16 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_17 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 17 = lineBits disc 1 17 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_17 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 17 = lineBits disc 2 17 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_17 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 17 = lineBits disc 3 17 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_18 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 18 = lineBits disc 1 18 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_18 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 18 = lineBits disc 2 18 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_18 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 18 = lineBits disc 3 18 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_19 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 19 = lineBits disc 1 19 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_19 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 19 = lineBits disc 2 19 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_19 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 19 = lineBits disc 3 19 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_20 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 20 = lineBits disc 1 20 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_20 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 20 = lineBits disc 2 20 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_20 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 20 = lineBits disc 3 20 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_21 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 21 = lineBits disc 1 21 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_21 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 21 = lineBits disc 2 21 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_21 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 21 = lineBits disc 3 21 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_22 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 22 = lineBits disc 1 22 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_22 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 22 = lineBits disc 2 22 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_22 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 22 = lineBits disc 3 22 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_23 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 23 = lineBits disc 1 23 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_23 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 23 = lineBits disc 2 23 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_23 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 23 = lineBits disc 3 23 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

end NeuralReversi.Stability

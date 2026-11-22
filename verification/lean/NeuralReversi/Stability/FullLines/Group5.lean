import NeuralReversi.Stability.Geometry
import NeuralReversi.Stability.FullLines.Group4
namespace NeuralReversi.Stability
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
theorem full_1_40 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 40 = lineBits disc 1 40 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_40 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 40 = lineBits disc 2 40 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_40 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 40 = lineBits disc 3 40 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_41 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 41 = lineBits disc 1 41 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_41 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 41 = lineBits disc 2 41 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_41 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 41 = lineBits disc 3 41 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_42 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 42 = lineBits disc 1 42 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_42 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 42 = lineBits disc 2 42 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_42 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 42 = lineBits disc 3 42 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_43 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 43 = lineBits disc 1 43 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_43 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 43 = lineBits disc 2 43 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_43 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 43 = lineBits disc 3 43 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_44 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 44 = lineBits disc 1 44 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_44 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 44 = lineBits disc 2 44 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_44 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 44 = lineBits disc 3 44 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_45 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 45 = lineBits disc 1 45 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_45 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 45 = lineBits disc 2 45 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_45 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 45 = lineBits disc 3 45 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_46 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 46 = lineBits disc 1 46 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_46 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 46 = lineBits disc 2 46 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_46 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 46 = lineBits disc 3 46 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_1_47 (disc : Word) :
    (field (getFullLines disc) 1).getLsbD 47 = lineBits disc 1 47 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_2_47 (disc : Word) :
    (field (getFullLines disc) 2).getLsbD 47 = lineBits disc 2 47 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem full_3_47 (disc : Word) :
    (field (getFullLines disc) 3).getLsbD 47 = lineBits disc 3 47 := by
  simp [field, getFullLines, rotateRight, rotateLeft, lineBits, lineCells,
    sameLine, List.range_succ, -BitVec.getLsbD_eq_getElem,
    Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

end NeuralReversi.Stability

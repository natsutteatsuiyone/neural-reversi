import NeuralReversi.Solve1.Bmi2Proofs.Part6
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
namespace NeuralReversi.Solve1.Bmi2
theorem refs_56 (p : Word) : refs p 56 = reference p 56 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_56 (p : Word) : refsOpp p 56 = reference (~~~p) 56 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_57 (p : Word) : refs p 57 = reference p 57 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_57 (p : Word) : refsOpp p 57 = reference (~~~p) 57 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_58 (p : Word) : refs p 58 = reference p 58 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_58 (p : Word) : refsOpp p 58 = reference (~~~p) 58 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_59 (p : Word) : refs p 59 = reference p 59 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_59 (p : Word) : refsOpp p 59 = reference (~~~p) 59 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_60 (p : Word) : refs p 60 = reference p 60 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_60 (p : Word) : refsOpp p 60 = reference (~~~p) 60 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_61 (p : Word) : refs p 61 = reference p 61 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_61 (p : Word) : refsOpp p 61 = reference (~~~p) 61 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_62 (p : Word) : refs p 62 = reference p 62 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_62 (p : Word) : refsOpp p 62 = reference (~~~p) 62 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_63 (p : Word) : refs p 63 = reference p 63 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_63 (p : Word) : refsOpp p 63 = reference (~~~p) 63 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
end NeuralReversi.Solve1.Bmi2

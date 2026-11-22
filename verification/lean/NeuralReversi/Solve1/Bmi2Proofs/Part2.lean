import NeuralReversi.Solve1.Bmi2Proofs.Part1
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
namespace NeuralReversi.Solve1.Bmi2
theorem refs_16 (p : Word) : refs p 16 = reference p 16 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_16 (p : Word) : refsOpp p 16 = reference (~~~p) 16 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_17 (p : Word) : refs p 17 = reference p 17 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_17 (p : Word) : refsOpp p 17 = reference (~~~p) 17 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_18 (p : Word) : refs p 18 = reference p 18 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_18 (p : Word) : refsOpp p 18 = reference (~~~p) 18 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_19 (p : Word) : refs p 19 = reference p 19 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_19 (p : Word) : refsOpp p 19 = reference (~~~p) 19 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_20 (p : Word) : refs p 20 = reference p 20 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_20 (p : Word) : refsOpp p 20 = reference (~~~p) 20 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_21 (p : Word) : refs p 21 = reference p 21 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_21 (p : Word) : refsOpp p 21 = reference (~~~p) 21 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_22 (p : Word) : refs p 22 = reference p 22 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_22 (p : Word) : refsOpp p 22 = reference (~~~p) 22 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_23 (p : Word) : refs p 23 = reference p 23 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_23 (p : Word) : refsOpp p 23 = reference (~~~p) 23 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
end NeuralReversi.Solve1.Bmi2

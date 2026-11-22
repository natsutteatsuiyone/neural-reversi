import NeuralReversi.Solve1.Bmi2Proofs.Part3
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
namespace NeuralReversi.Solve1.Bmi2
theorem refs_32 (p : Word) : refs p 32 = reference p 32 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_32 (p : Word) : refsOpp p 32 = reference (~~~p) 32 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_33 (p : Word) : refs p 33 = reference p 33 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_33 (p : Word) : refsOpp p 33 = reference (~~~p) 33 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_34 (p : Word) : refs p 34 = reference p 34 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_34 (p : Word) : refsOpp p 34 = reference (~~~p) 34 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_35 (p : Word) : refs p 35 = reference p 35 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_35 (p : Word) : refsOpp p 35 = reference (~~~p) 35 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_36 (p : Word) : refs p 36 = reference p 36 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_36 (p : Word) : refsOpp p 36 = reference (~~~p) 36 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_37 (p : Word) : refs p 37 = reference p 37 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_37 (p : Word) : refsOpp p 37 = reference (~~~p) 37 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_38 (p : Word) : refs p 38 = reference p 38 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_38 (p : Word) : refsOpp p 38 = reference (~~~p) 38 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_39 (p : Word) : refs p 39 = reference p 39 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_39 (p : Word) : refsOpp p 39 = reference (~~~p) 39 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
end NeuralReversi.Solve1.Bmi2

import NeuralReversi.Solve1.Bmi2Proofs.Part4
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
namespace NeuralReversi.Solve1.Bmi2
theorem refs_40 (p : Word) : refs p 40 = reference p 40 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_40 (p : Word) : refsOpp p 40 = reference (~~~p) 40 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_41 (p : Word) : refs p 41 = reference p 41 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_41 (p : Word) : refsOpp p 41 = reference (~~~p) 41 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_42 (p : Word) : refs p 42 = reference p 42 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_42 (p : Word) : refsOpp p 42 = reference (~~~p) 42 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_43 (p : Word) : refs p 43 = reference p 43 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_43 (p : Word) : refsOpp p 43 = reference (~~~p) 43 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_44 (p : Word) : refs p 44 = reference p 44 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_44 (p : Word) : refsOpp p 44 = reference (~~~p) 44 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_45 (p : Word) : refs p 45 = reference p 45 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_45 (p : Word) : refsOpp p 45 = reference (~~~p) 45 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_46 (p : Word) : refs p 46 = reference p 46 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_46 (p : Word) : refsOpp p 46 = reference (~~~p) 46 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_47 (p : Word) : refs p 47 = reference p 47 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_47 (p : Word) : refsOpp p 47 = reference (~~~p) 47 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
end NeuralReversi.Solve1.Bmi2

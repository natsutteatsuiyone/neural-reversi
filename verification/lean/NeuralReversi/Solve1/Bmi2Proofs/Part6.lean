import NeuralReversi.Solve1.Bmi2Proofs.Part5
set_option maxRecDepth 10000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
namespace NeuralReversi.Solve1.Bmi2
theorem refs_48 (p : Word) : refs p 48 = reference p 48 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_48 (p : Word) : refsOpp p 48 = reference (~~~p) 48 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_49 (p : Word) : refs p 49 = reference p 49 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_49 (p : Word) : refsOpp p 49 = reference (~~~p) 49 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_50 (p : Word) : refs p 50 = reference p 50 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_50 (p : Word) : refsOpp p 50 = reference (~~~p) 50 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_51 (p : Word) : refs p 51 = reference p 51 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_51 (p : Word) : refsOpp p 51 = reference (~~~p) 51 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_52 (p : Word) : refs p 52 = reference p 52 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_52 (p : Word) : refsOpp p 52 = reference (~~~p) 52 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_53 (p : Word) : refs p 53 = reference p 53 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_53 (p : Word) : refsOpp p 53 = reference (~~~p) 53 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_54 (p : Word) : refs p 54 = reference p 54 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_54 (p : Word) : refsOpp p 54 = reference (~~~p) 54 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refs_55 (p : Word) : refs p 55 = reference p 55 := by
  simp [refs, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
theorem refsOpp_55 (p : Word) : refsOpp p 55 = reference (~~~p) 55 := by
  simp [refsOpp, unionParams, LineTable.reference, LineTable.ray, pext, projectIndex, reference, raySquares, ray, List.range_succ, getLsbD_cond, -BitVec.getLsbD_eq_getElem] <;> omega
end NeuralReversi.Solve1.Bmi2

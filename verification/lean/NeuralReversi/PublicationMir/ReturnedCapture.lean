module
public import NeuralReversi.PublicationMir.Program
public import NeuralReversi.PublicationMir.ReturnedScores

@[expose] public section

namespace NeuralReversi.PublicationMir.ReturnedScores

def raw (row : SearchPublication.Row) : Move := (row.square, BitVec.ofInt 32 row.score)

/-- Every completed nonempty snapshot of machine scores produces the score
    fields checked on return, for every allowed ordering of tied moves. -/
theorem capture_returns (sort : SliceSort) (roots : List RootMoveStorage.Row)
    (nonempty : roots ≠ []) (depth : BitVec 32) (nodes : Nat)
    (fallback : Unit → SearchResult)
    (signed : ∀ r ∈ roots, (BitVec.ofInt 32 r.score).toInt = r.score) :
    ∃ it report, (record sort roots depth.toNat 5 true).map Prod.fst = some (some it) ∧
      intoResult (some it) nodes fallback = some (.bestMove report) ∧
      Matches (roots.map fun r => (r.square, r.score))
        (it.rows.map (raw ∘ project)) depth.toNat
        ⟨raw report.best,report.alternatives.map raw,BitVec.ofNat 32 report.stamp.depth,
          report.stamp.selectivity,report.endgame⟩ := by
  let ordered := sort.byReverseKey key roots
  have perm : ordered.Perm roots := sort.permutation ..
  have typed (r : RootMoveStorage.Row) (h : r ∈ ordered) := signed r (perm.mem_iff.mp h)
  have sorted : ordered.Pairwise (fun a b => b.score ≤ a.score) := sort.ordered key roots
  cases eq : ordered with
  | nil => exact (nonempty (List.length_eq_zero_iff.mp (by simpa [eq] using perm.length_eq.symm))).elim
  | cons head tail =>
    let it := capture sort roots depth.toNat 5 true
    have rows : it.rows = head :: tail := eq
    let report : SearchFallback.Report := ⟨project head,(head :: tail).map project,⟨depth.toNat,5⟩,true,nodes⟩
    refine ⟨it,report,?_,?_,?_⟩
    · simp [record,first,← show it = capture sort roots depth.toNat 5 true from rfl,rows]
    · simp [intoResult,first,rows,report,it,build,project]
      exact ⟨⟨rfl,rfl⟩,rfl⟩
    · have permRaw := perm.map (fun r => (r.square,r.score))
      have scores : (ordered.map (raw ∘ project)).map value = ordered.map (fun r => (r.square,r.score)) := by
        rw [List.map_map]
        apply List.map_congr_left
        intro r mem
        simp [Function.comp_def,value,raw,project,typed r mem]
      have order : (ordered.map (raw ∘ project)).Pairwise (fun a b => b.2.toInt ≤ a.2.toInt) := by
        apply List.pairwise_map.mpr
        apply sorted.imp_of_mem
        intro a b ha hb
        simp [raw,project,typed a ha,typed b hb]
      change Matches _ (ordered.map (raw ∘ project)) _ _
      refine ⟨scores ▸ permRaw,order,?_,?_,?_,rfl,rfl⟩
      · simp [eq,report]
      · simp [eq,report,List.map_map]
      · simp [report]

end NeuralReversi.PublicationMir.ReturnedScores

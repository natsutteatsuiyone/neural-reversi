import NeuralReversi.AspirationMir.PublicScore
import NeuralReversi.EndgameTreeCache
import NeuralReversi.PublicationMir.ReturnedCapture

namespace NeuralReversi.AspirationMir.SerialTable.MultiRows

open PvsParallel
open PvsBoardTree (Position minimax)
open RootMoveStorage (Row sortRows)
open PvsMoveLoop (scoreMax)
open PublicationMir.ReturnedScores (raw)

/-- The next line reads this row at index `pv_idx - 1` to construct its first window. -/
theorem accepted_previous (covers : Root.Covers mc b (initial.map Row.square)) (within : idx < initial.length)
    (writes : RootSuffix.Writes initial idx a z scored snap) (inside : a < snap.state.best ∧ snap.state.best < z)
    (attempt : RootOrder.SoundAttempt mc fc b (initial.map Row.square) idx a z snap.state.best snap.scores) :
    ∃ row, (sortRows scored idx)[idx]? = some row ∧ row.score = snap.state.best ∧
      (-scoreMax ≤ row.score ∧ row.score ≤ scoreMax) := by
  obtain ⟨keys,_,stored⟩ := writes.scores (Root.roots_nodup covers)
  have len : scored.length = initial.length := by simpa using congrArg List.length keys
  have index : idx < scored.length := len ▸ within
  have sorted : idx < (sortRows scored idx).length := (RootMoveStorage.sortRowsContract_proven scored idx).1.length_eq ▸ index
  have picked := RootSuffix.pickedScore index (keys ▸ attempt) inside.1 inside.2 stored sorted
  have bounded := Root.value_range (fc := fc) covers (by simpa using within : idx < (initial.map Row.square).length)
  exact ⟨_,List.getElem?_eq_getElem sorted,picked.1,picked.1 ▸ (attempt.1.exact inside.1 inside.2 ▸ bounded)⟩

structure ReportExact (mc : Bitboard.Moves.X86Config) (fc : Root.FlipConfig) (b : Position)
    (r : SearchFallback.Report) : Prop where
  flag : r.endgame = true
  level : r.stamp.selectivity = 5
  score : r.best.score = 256 * EndgameTreeCache.discValue mc fc b.val
  bestWord : (raw r.best).2.toInt = 256 * EndgameTreeCache.discValue mc fc b.val
  head : r.alternatives.head? = some r.best
  cover : Root.Covers mc b (r.alternatives.map SearchPublication.Row.square)
  ordered : r.alternatives.Pairwise (fun a z => z.score ≤ a.score)
  alternatives : ∀ move ∈ r.alternatives, (raw move).2.toInt = Root.squareValue mc fc b move.square

private theorem square_fits (b : Position) (square : Bitboard.Square) :
    (BitVec.ofInt 32 (Root.squareValue mc fc b square)).toInt = Root.squareValue mc fc b square := by
  unfold Root.squareValue
  split
  · rename_i legal
    have bounds := PvsMoveLoop.minimax_range (PvsBoardTree.build mc fc (PvsBoardTree.play mc fc b ⟨square,legal⟩).val)
    unfold Root.childValue PvsBoardTree.minimax
    unfold scoreMax at bounds
    apply BitVec.toInt_ofInt_eq_self (by decide) <;> omega
  · decide

/-- The published alternatives cover every legal move once, including ties. Their
    signed score fields agree with the individual moves, and the best score with the board. -/
theorem publish (lines : RootSuffix.Lines mc fc b 0 before final)
    (covers : Root.Covers mc b (before.map Row.square)) (nonempty : before ≠ [])
    (sort : PublicationMir.SliceSort) (depth nodes : Nat) (fallback : Unit → PublicationMir.SearchResult) :
    ∃ report, PublicationMir.completed sort final depth 5 true nodes fallback = some (.bestMove report) ∧
      ReportExact mc fc b report := by
  obtain ⟨covered,truth⟩ := lines.exactRows covers (by simp)
  obtain ⟨result,completed,top⟩ := publicMultiPV lines covers nonempty sort depth nodes fallback
  let sorted := sort.byReverseKey PublicationMir.key final
  have perm : sorted.Perm final := sort.permutation ..
  have ordered := sort.ordered PublicationMir.key final
  have kept : sorted ≠ [] := by
    intro eq
    have lengths := (perm.map Row.square).trans (covered.trans covers.symm)
    exact nonempty (List.length_eq_zero_iff.mp (by simpa [eq] using lengths.length_eq.symm))
  obtain ⟨best,tail,shape⟩ := List.exists_cons_of_ne_nil kept
  let report : SearchFallback.Report := ⟨PublicationMir.project best,(best::tail).map PublicationMir.project,⟨depth,5⟩,true,nodes⟩
  have actual : PublicationMir.completed sort final depth 5 true nodes fallback = some (.bestMove report) := by
    have keyOrder : sort.byReverseKey PublicationMir.key final = best :: tail := shape
    simp [PublicationMir.completed,PublicationMir.record,PublicationMir.first,PublicationMir.capture,
      PublicationMir.snapshot,PublicationMir.intoResult,PublicationMir.build,PublicationMir.project,keyOrder,report]
  rw [actual] at completed
  cases completed
  have value : report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val :=
    (Option.some.inj top).trans (EndgameTreeCache.scaled_discValue ..).symm
  rw [shape] at perm
  have scores (r : Row) (member : r ∈ best :: tail) := truth r (perm.mem_iff.mp member)
  have alternatives (move : SearchPublication.Row) (present : move ∈ report.alternatives) :
      (raw move).2.toInt = Root.squareValue mc fc b move.square := by
    obtain ⟨r,member,rfl⟩ := List.mem_map.mp present
    change (BitVec.ofInt 32 r.score).toInt = Root.squareValue mc fc b r.square
    rw [scores r member,square_fits]
  refine ⟨report,actual,rfl,rfl,value,?_,rfl,?_,?_,alternatives⟩
  · exact (alternatives report.best (by simp [report])).trans ((scores best (by simp)).symm.trans value)
  · change (((best :: tail).map PublicationMir.project).map SearchPublication.Row.square).Perm _
    rw [List.map_map]
    exact (perm.map Row.square).trans covered
  · apply List.pairwise_map.mpr
    change sorted.Pairwise _ at ordered
    exact shape ▸ ordered

end NeuralReversi.AspirationMir.SerialTable.MultiRows

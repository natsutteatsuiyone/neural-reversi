import NeuralReversi.PvsParallel.ConcurrentReplay
import NeuralReversi.RootMoveStorage

namespace NeuralReversi.PvsParallel.RootSuffix

open Bitboard (Square)
open Bitboard.Moves (X86Config)
open PvsBoardTree (Position minimax)
open RootMoveStorage (Row sortRows)

theorem sortRows_squares (rows : List Row) (idx : Nat) (scores : Square → Int)
    (agree : ∀ row ∈ rows.drop idx, row.score = scores row.square) :
    (sortRows rows idx).map Row.square = RootOrder.sortFrom (rows.map Row.square) idx scores := by
  simp only [sortRows, RootOrder.sortFrom, List.map_append, List.map_take, ← List.map_drop]
  congr 1
  apply List.map_mergeSort
  intro a ha b hb
  rw [agree a ha, agree b hb]

theorem pickedScore {mc : X86Config} {fc : Root.FlipConfig} {board : Position}
    {rows : List Row} {idx : Nat} (within : idx < rows.length)
    (attempt : RootOrder.SoundAttempt mc fc board (rows.map Row.square) idx alpha beta score replayed)
    (lo : alpha < score) (hi : score < beta)
    (stored : ∀ row ∈ rows.drop idx, row.score = replayed row.square) :
    let sorted := sortRows rows idx
    ∀ h : idx < sorted.length, sorted[idx].score = score ∧
      sorted[idx].score = Root.squareValue mc fc board sorted[idx].square := by
  intro sorted h
  obtain ⟨valid, choose⟩ := attempt
  obtain ⟨head, tail, order, selected⟩ := choose lo hi
  have squareOrder := sortRows_squares rows idx replayed stored
  rw [← squareOrder] at order
  have headEq := congrArg List.head? order
  change ((sorted.map Row.square).drop idx).head? = some head at headEq
  have eq : sorted[idx].square = head := by simpa [List.getElem?_eq_getElem, h] using headEq
  have sortedTail : sorted.drop idx = (rows.drop idx).mergeSort (fun a b => decide (b.score ≤ a.score)) :=
    List.drop_left' (by simp [Nat.min_eq_left (Nat.le_of_lt within)])
  have member : sorted[idx] ∈ sorted.drop idx := by
    apply List.mem_of_getElem? (i := 0)
    simpa only [← List.head?_eq_getElem?, List.head?_drop] using (List.getElem?_eq_getElem h)
  rw [sortedTail] at member
  have old := (List.mergeSort_perm _ _).mem_iff.mp member
  rw [stored _ old, eq, selected.2.1, selected.2.2.1]
  exact ⟨(valid.exact lo hi).symm,rfl⟩

/-- Physical row transitions at the final selectivity of each line.
Earlier selectivities may reorder or rescore the candidate suffix arbitrarily. -/
inductive Lines (mc : X86Config) (fc : Root.FlipConfig) (board : Position) :
    Nat → List Row → List Row → Prop
  | done (rows : List Row) : Lines mc fc board rows.length rows rows
  | accepted {idx : Nat} {before scored final : List Row}
      {alpha beta score : Int} {replayed : Square → Int} :
      (scored.map Row.square).Perm (before.map Row.square) →
      scored.take idx = before.take idx → idx < scored.length →
      alpha < score ∧ score < beta →
      RootOrder.SoundAttempt mc fc board (scored.map Row.square) idx alpha beta score replayed →
      (∀ row ∈ scored.drop idx, row.score = replayed row.square) →
      Lines mc fc board (idx + 1) (sortRows scored idx) final →
      Lines mc fc board idx before final

theorem Lines.exactRows (lines : Lines mc fc board idx before final)
    (covers : Root.Covers mc board (before.map Row.square))
    (earlier : ∀ row ∈ before.take idx, row.score = Root.squareValue mc fc board row.square) :
    Root.Covers mc board (final.map Row.square) ∧
    ∀ row ∈ final, row.score = Root.squareValue mc fc board row.square := by
  induction lines with
  | done rows => exact ⟨covers, by simpa using earlier⟩
  | @accepted idx before scored final a b score replayed permutation front within inside attempt stored _ ih =>
    have covered := permutation.trans covers
    have after := (RootMoveStorage.sortRowsContract_proven scored idx).1
    have sortedLength : idx < (sortRows scored idx).length := after.length_eq ▸ within
    have pick := pickedScore within attempt inside.1 inside.2 stored sortedLength
    apply ih ((after.map _).trans covered)
    intro row included
    rw [List.take_succ_eq_append_getElem sortedLength] at included
    rcases List.mem_append.mp included with saved | chosen
    · rw [(RootMoveStorage.sortRowsContract_proven scored idx).2.1, front] at saved
      exact earlier row saved
    · simpa only [List.mem_singleton.mp chosen] using pick.2

def publishedRow (row : Row) : SearchPublication.Row := ⟨row.square, row.score, row.pv⟩

theorem completedResult (lines : Lines mc fc board 0 before final)
    (covers : Root.Covers mc board (before.map Row.square)) (nonempty : before ≠ [])
    (depth nodes : Nat) :
    ∃ result, SearchFallback.fromSnapshot
      (SearchPublication.capture (final.map publishedRow) ⟨depth, 5⟩ true) nodes = some result ∧
      result.best.score = minimax mc fc board ∧ result.endgame = true := by
  obtain ⟨finalCovered, exactScores⟩ := lines.exactRows covers (by simp)
  let rows := final.map publishedRow
  let saved := SearchPublication.capture rows ⟨depth, 5⟩ true
  have perm : saved.rows.Perm rows := (SearchPublication.captureContract_proven rows ⟨depth, 5⟩ true).2.2.1
  have ordered := (SearchPublication.captureContract_proven rows ⟨depth, 5⟩ true).2.2.2.1
  have lookup : ∀ row ∈ saved.rows, row.score = Root.squareValue mc fc board row.square := by
    intro row member
    obtain ⟨old, original, rfl⟩ := List.mem_map.mp (perm.mem_iff.mp member)
    exact exactScores old original
  cases resultRows : saved.rows with
  | nil =>
    have noRows := perm.length_eq
    rw [resultRows] at noRows
    simp [rows] at noRows
    have same := (finalCovered.trans covers.symm).length_eq
    simp only [List.length_map] at same
    exact False.elim (nonempty (List.length_eq_zero_iff.mp (same ▸ noRows.symm)))
  | cons first rest =>
    have order : (first :: rest).Pairwise (fun a b => b.score ≤ a.score) := resultRows ▸ ordered
    have cover : Root.Covers mc board ((first :: rest).map SearchPublication.Row.square) := by
      rw [← resultRows]
      have projection := perm.map SearchPublication.Row.square
      apply projection.trans
      change ((final.map publishedRow).map SearchPublication.Row.square).Perm _
      rw [List.map_map]
      exact finalCovered
    have ranked : RootMultiPV.Ranked (Root.squareValue mc fc board)
        ((first :: rest).map SearchPublication.Row.square) [⟨first.square, first.score⟩] := by
      refine ⟨by simp, lookup first (by simp [resultRows]), ?_, trivial⟩
      intro sq present
      obtain ⟨row, member, rfl⟩ := List.mem_map.mp present
      rw [← lookup row (resultRows ▸ member)]
      rcases List.mem_cons.mp member with rfl | later
      · exact Int.le_refl _
      · exact (List.pairwise_cons.mp order).1 row later
    have exactScore := (ExactEndgameScore.ranked_first mc fc board _ _ _ cover ranked).2
    refine ⟨⟨first, first :: rest, ⟨depth, 5⟩, true, nodes⟩, ?_, exactScore, rfl⟩
    change SearchFallback.fromSnapshot saved nodes = _
    rw [SearchFallback.fromSnapshot, resultRows]
    rfl

end NeuralReversi.PvsParallel.RootSuffix


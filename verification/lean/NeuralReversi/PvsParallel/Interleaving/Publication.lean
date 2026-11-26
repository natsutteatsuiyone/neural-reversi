import NeuralReversi.PvsParallel.Interleaving.RootLadder

namespace NeuralReversi.PvsParallel.Interleaving.Publication

open AspirationMir
open AspirationMir.SerialTable
open RootMoveStorage (Row)
open PvsMoveLoop (scoreMax)
open PublicationMir (SearchResult Iteration SliceSort)
open RootLadder (Source Laws trial)

/-- The root count is precisely `SearchTask::pv_count`. Every aspiration retry gets
    its own owner and helper execution through `source`. -/
def solve (source : Source M mc fc b done_) (multi : Bool) (clock : MultiPublic.Clock)
    (cap minimum : Nat) (prior : Option Iteration) (base : Word) (rows : List Row) (m : M)
    (sort : SliceSort) (nodes : Nat) (fallback : Unit → SearchResult) : Option (SearchResult × M) := do
  let count := if multi then rows.length else 1
  let levels := MultiClock.levels cap minimum clock.prediction clock.elapsed
  let (final,next) ← MultiBatch.lines (trial source) levels base count 0 rows m
  let result ← OuterMir.finish minimum ((levels (count-1)).getLastD 5) sort final (Board.emptyCount b.val).toNat nodes prior fallback
  return (result,next)

theorem from_flags (source : Source M mc fc b done_) (laws : Laws mc fc b done_) (multi : Bool)
    (clock : MultiPublic.Clock) (cover : Root.Covers mc b (rows.map Row.square)) (pos : rows ≠ [])
    (range : -scoreMax ≤ base.toInt ∧ base.toInt ≤ scoreMax) (origin : MultiClock.Origin minimum prior)
    (enabled : 1 ≤ cap) (time : clock.ordered ((if multi then rows.length else 1)-1))
    (returned : solve source multi clock cap minimum prior base rows m sort nodes fallback = some (.bestMove report,next))
    (tag : report.endgame = true) (full : report.stamp.selectivity = 5) :
    report.best.score = 256 * EndgameTreeCache.discValue mc fc b.val ∧
      (multi = true → MultiRows.ReportExact mc fc b report) := by
  obtain ⟨⟨final,after⟩,steps,more⟩ := Option.bind_eq_some_iff.mp returned
  obtain ⟨res,printed,eq⟩ := Option.bind_eq_some_iff.mp more
  cases eq
  let count := if multi then rows.length else 1
  let levels := MultiClock.levels cap minimum clock.prediction clock.elapsed
  have all := MultiClock.all_of_report origin enabled time.1 time.2 printed tag full
  have last : levels (count-1) = OuterMir.ladder := all (count-1) (Nat.le_refl _)
  have threshold : minimum ≤ 5 := by cases origin <;> decide
  change OuterMir.finish minimum ((levels (count-1)).getLastD 5) _ _ _ _ _ _ = _ at printed
  rw [last] at printed
  change OuterMir.finish minimum 5 _ _ _ _ _ _ = _ at printed
  rw [OuterMir.finish,ite_eq_left threshold] at printed
  cases multi with
  | true =>
    have lines := RootLadder.exact_lines source laws levels rows.length 0 rows cover (by simp) range
      (by simp [MultiLine.previous]) (fun i _ hi => all i (by simp_all [count]; omega)) steps
    obtain ⟨best,pub,correct⟩ := MultiRows.publish lines cover pos sort (Board.emptyCount b.val).toNat nodes fallback
    rw [pub] at printed
    cases printed
    exact ⟨correct.score,fun _ => correct⟩
  | false =>
    obtain ⟨out,ran,tail⟩ := Option.bind_eq_some_iff.mp steps
    have ladder : levels 0 = OuterMir.ladder := all 0 (Nat.zero_le _)
    change MultiLine.run 0 base (trial source 0) (levels 0) rows m = some out at ran
    rw [ladder] at ran
    obtain ⟨active,last,rfl,ok,accepted,_,_,_,_⟩ := RootLadder.accepted_line source laws cover
      (List.length_pos_iff.mpr pos) range (by simp [MultiLine.previous]) ran
    obtain ⟨snap,best,written,sound⟩ := accepted.certificate rfl
    have before : Root.Covers mc b (last.state.rows.map Row.square) := accepted.keys ▸ ok.cover
    have len : last.state.rows.length = active.rows.length := by simpa using congrArg List.length accepted.keys
    obtain ⟨result,pub,score⟩ := publishSingle before (len ▸ ok.index) written (best ▸ accepted.inside) (best ▸ sound)
      (PublicationMir.ExactScore.afterRootSort sort) (Board.emptyCount b.val).toNat nodes fallback
    cases tail
    change PublicationMir.completed sort (MultiLevel.successor 0 last).rows _ 5 true _ _ = some result at pub
    rw [printed] at pub
    cases pub
    exact ⟨(Option.some.inj score).trans (EndgameTreeCache.scaled_discValue ..).symm,by simp⟩

end NeuralReversi.PvsParallel.Interleaving.Publication

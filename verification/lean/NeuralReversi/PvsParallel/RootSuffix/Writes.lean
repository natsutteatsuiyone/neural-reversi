module
public import NeuralReversi.PvsParallel.RootSuffix
public import NeuralReversi.MidgameAspiration.History

@[expose] public section

namespace NeuralReversi.PvsParallel.RootSuffix

open Bitboard (Square)
open RootMoveStorage (Row)
open RootOrder (Report)
open RootReplay (Snapshot)

def updateRow (row : Row) (score : Int) (isPV : Bool) (pv : List Square) : Row :=
  let history := MidgameAspiration.History.update ⟨row.average, row.meanSquared⟩ score
  { row with score := if isPV then score else -PvsMoveLoop.scoreInf
             average := history.average
             meanSquared := history.meanSquared
             pv := if isPV then pv else row.pv }

def writeFirst (square : Square) (update : Row → Row) : List Row → List Row
  | [] => []
  | row :: rest =>
    if row.square = square then update row :: rest else row :: writeFirst square update rest

private theorem writeFirst_eq_map (rows : List Row) (square : Square) (update : Row → Row)
    (unique : (rows.map Row.square).Nodup) :
    writeFirst square update rows =
      rows.map (fun row => if row.square = square then update row else row) := by
  induction rows with
  | nil => rfl
  | cons row rest ih =>
    change (row.square :: rest.map Row.square).Nodup at unique
    obtain ⟨fresh, tail⟩ := List.nodup_cons.mp unique
    by_cases same : row.square = square
    · simp only [writeFirst, same, ite_true, List.map_cons, List.cons.injEq, true_and]
      symm
      apply Eq.trans _ (List.map_id' rest)
      apply List.map_congr_left
      intro old included
      have different : old.square ≠ square := fun collision =>
        fresh (by simpa only [same, collision] using (List.mem_map_of_mem (f := Row.square) included))
      simp [different]
    · simp [writeFirst, same, ih tail]

/-- Sequence of observable root updates while one numeric trial runs, starting at its suffix reset. -/
inductive Writes (initial : List Row) (idx : Nat) (alpha beta : Int) :
    List Row → Snapshot → Prop
  | reset : Writes initial idx alpha beta (RootMoveStorage.resetFrom initial idx) (RootReplay.start alpha)
  | record {rows : List Row} {before : Snapshot} (prior : Writes initial idx alpha beta rows before)
      (report : Report) (isPV : Bool) (pv : List Square) (candidate : report.square ∈ (rows.drop idx).map Row.square)
      (visibility : isPV = (before.seen.isEmpty || decide (before.state.alpha < report.score))) :
      Writes initial idx alpha beta (writeFirst report.square (fun row => updateRow row report.score isPV pv) rows)
        (before.record beta report)

theorem Writes.scores (writes : Writes initial idx alpha beta rows latest)
    (unique : (initial.map Row.square).Nodup) :
    rows.map Row.square = initial.map Row.square ∧ rows.take idx = initial.take idx ∧
      ∀ row ∈ rows.drop idx, row.score = latest.scores row.square := by
  induction writes with
  | reset =>
    have reset := RootMoveStorage.resetContract_proven initial idx
    refine ⟨?_, reset.1, reset.2.2.2.1⟩
    unfold RootMoveStorage.resetFrom
    simp only [List.map_append, List.map_map]
    change (initial.take idx).map Row.square ++ (initial.drop idx).map Row.square = _
    rw [← List.map_append, List.take_append_drop]
  | @record rows before prior report isPV pv candidate visibility ih =>
    obtain ⟨squares, front, scores⟩ := ih
    have distinct : (rows.map Row.square).Nodup := squares ▸ unique
    have separated : ((rows.take idx).map Row.square ++ (rows.drop idx).map Row.square).Nodup := by
      rw [← List.map_append, List.take_append_drop]
      exact distinct
    let install (row : Row) := if row.square = report.square then updateRow row report.score isPV pv else row
    have names (row : Row) : (install row).square = row.square := by
      dsimp [install]
      split <;> rfl
    have fixed : ∀ r ∈ rows.take idx, install r = r := by
      intro r member
      have unequal := (List.nodup_append.mp separated).2.2 r.square
        (List.mem_map_of_mem member) report.square candidate
      simp [install, unequal]
    rw [writeFirst_eq_map rows _ _ distinct]
    change (rows.map install).map Row.square = _ ∧ (rows.map install).take idx = _ ∧ _
    refine ⟨?_, ?_, ?_⟩
    · rw [List.map_map]
      exact (List.map_congr_left (fun row _ => names row)).trans squares
    · rw [← List.map_take, List.map_congr_left fixed, List.map_id', front]
    · intro row present
      change row ∈ (rows.map install).drop idx at present
      rw [← List.map_drop] at present
      obtain ⟨old, member, rfl⟩ := List.mem_map.mp present
      rw [names]
      change (install old).score = RootOrder.writeScore before.scores report.square _ old.square
      by_cases same : old.square = report.square
      · simp [install, same, updateRow, RootOrder.writeScore, RootOrder.visibleScore, visibility]
      · simpa [install, same, RootOrder.writeScore] using scores old member

theorem Lines.acceptedWrites
    (covers : Root.Covers mc board (initial.map Row.square))
    (order : (initial.map Row.square).Perm (before.map Row.square))
    (front : initial.take idx = before.take idx)
    (within : idx < initial.length)
    (writes : Writes initial idx alpha beta scored latest)
    (inside : alpha < latest.state.best ∧ latest.state.best < beta)
    (attempt : RootOrder.SoundAttempt mc fc board (initial.map Row.square) idx alpha beta
      latest.state.best latest.scores)
    (next : Lines mc fc board (idx + 1) (RootMoveStorage.sortRows scored idx) final) :
    Lines mc fc board idx before final := by
  obtain ⟨squares, unchanged, stored⟩ := writes.scores (Root.roots_nodup covers)
  apply Lines.accepted (squares ▸ order) (unchanged.trans front) _ inside (squares ▸ attempt) stored next
  have size := congrArg List.length squares
  simpa using (show idx < (scored.map Row.square).length from size ▸ (by simpa using within))

end NeuralReversi.PvsParallel.RootSuffix

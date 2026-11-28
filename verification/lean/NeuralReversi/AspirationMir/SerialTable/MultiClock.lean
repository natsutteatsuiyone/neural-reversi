module
public import NeuralReversi.OuterMir.Correct

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.MultiClock

/-- The prediction is captured once on entry to solve_root. Each PV samples elapsed time after
    each completed level, reusing the same budget and the factor for that level. -/
def levels (cap minimum : Nat) (prediction : Option EndgameTiming.Prediction)
    (nanos : Nat → EndgameProgress.Clock) (idx : Nat) : List Nat :=
  OuterMir.lineLevels cap minimum (OuterMir.sampledStop prediction nanos idx) OuterMir.ladder

inductive Origin : Nat → Option PublicationMir.Iteration → Prop where
  | direct : Origin 1 none
  | handoff (saved : PublicationMir.Iteration) (midgame : saved.endgame = false) : Origin 2 (some saved)

private theorem through_last (cap minimum last : Nat) (stop : Nat → Bool) (prior : List Nat)
    (unique : last ∉ prior) (hit : last ∈ OuterMir.lineLevels cap minimum stop (prior ++ [last])) :
    OuterMir.lineLevels cap minimum stop (prior ++ [last]) = prior ++ [last] := by
  induction prior with
  | nil =>
    by_cases limited : cap < last
    · simp [OuterMir.lineLevels,OuterMir.step,limited] at hit
    · cases h : stop last <;> simp [OuterMir.lineLevels,OuterMir.step,limited,h]
  | cons first tail ih =>
    have different : last ≠ first := fun eq => unique (eq ▸ List.mem_cons_self)
    have fresh : last ∉ tail := fun member => unique (List.mem_cons_of_mem _ member)
    by_cases limited : cap < first
    · simp [OuterMir.lineLevels,OuterMir.step,limited] at hit
    · cases h : stop first with
      | true => simp [OuterMir.lineLevels,OuterMir.step,limited,h,different] at hit
      | false =>
        simp only [List.cons_append,OuterMir.lineLevels,OuterMir.step,ite_eq_right limited,h,Bool.not_false,ite_true] at *
        exact congrArg (List.cons first) (ih fresh ((List.mem_cons.mp hit).resolve_left different))

theorem full_of_level (hit : 5 ∈ levels cap minimum prediction nanos idx) :
    levels cap minimum prediction nanos idx = OuterMir.ladder :=
  through_last cap minimum 5 _ [1,2,3,4] (by decide) hit

theorem noClock (minimum : Nat) (nanos : Nat → EndgameProgress.Clock) (idx : Nat) :
    levels 5 minimum none nanos idx = OuterMir.ladder := rfl

/-- A returned exact flag can only come from completing the last line at the exact level.
    The direct entry has a reportable first level. Below the midgame handoff's reporting
    threshold the preserved snapshot still carries its midgame flag. -/
theorem all_of_report (origin : Origin minimum prior) (enabled : 1 ≤ cap)
    (ordered : ∀ i, i ≤ lastLine → EndgameProgress.ClockOrder (nanos i) (nanos lastLine))
    (fits : ∀ step, nanos lastLine step / 1000000 < 2 ^ 64)
    (returned : OuterMir.finish minimum ((levels cap minimum prediction nanos lastLine).getLastD priorLevel)
      sort rows depth nodes prior fallback = some (.bestMove report))
    (tag : report.endgame = true) (full : report.stamp.selectivity = 5) :
    ∀ i, i ≤ lastLine → levels cap minimum prediction nanos i = OuterMir.ladder := by
  let last := (levels cap minimum prediction nanos lastLine).getLastD priorLevel
  have current : last = 5 := by
    by_cases eligible : minimum ≤ last
    · have stamp := (OuterMir.capture_stamp _ last _ _ eligible _ _ _ _ report returned).1
      exact (congrArg SearchPublication.Stamp.selectivity stamp).symm.trans full
    · cases origin with
      | direct =>
        exact False.elim (eligible (OuterMir.direct_has_current priorLevel cap enabled _))
      | handoff saved midgame =>
        dsimp only [last] at eligible
        have old : PublicationMir.intoResult (some saved) nodes fallback = some (.bestMove report) := by
          simpa only [OuterMir.finish,ite_eq_right eligible] using returned
        unfold PublicationMir.intoResult at old
        obtain ⟨chosen,_,eq⟩ := Option.bind_eq_some_iff.mp old
        cases eq
        simp [midgame] at tag
  intro i within
  exact full_of_level (OuterMir.allLines_of_clock priorLevel cap minimum lastLine enabled prediction nanos ordered fits current i within)

/-- The last PV cannot complete a common snapshot if it stops below the reporting level,
    even when earlier PVs have already published marks. -/
theorem inherited_when_below (lastLine depth : Nat) (prior : Option SearchPublication.Snapshot)
    (rows : Nat → Nat → List SearchPublication.Row)
    (below : (levels cap minimum prediction nanos lastLine).getLastD priorLevel < minimum) :
    let attempts := (List.range (lastLine+1)).flatMap (fun i =>
      (levels cap minimum prediction nanos i).map (fun level =>
        (⟨i,⟨depth,level⟩,true,rows i level,false⟩ : SearchPublication.Attempt)))
    (SearchPublication.run minimum (SearchPublication.start (lastLine+1) prior) attempts).last = prior := by
  apply (OuterMir.unpublished_last_line minimum lastLine _ _ (by simp [SearchPublication.start]) ?_).2
  intro a member eligible same
  obtain ⟨i,_,issued⟩ := List.mem_flatMap.mp member
  obtain ⟨level,hit,rfl⟩ := List.mem_map.mp issued
  change i = lastLine at same
  subst i
  have lo := OuterMir.final_unreportable minimum cap _ priorLevel below level hit
  simp [SearchPublication.reportable] at eligible
  omega

end NeuralReversi.AspirationMir.SerialTable.MultiClock

import NeuralReversi.PvsParallel.RootSerialMir

namespace NeuralReversi.PvsParallel.RootOwnerMir

open PvsPostpone (Queue step)
open PvsMoveLoop (ChildRelation ChildSearch)
open RootReplay (Snapshot)
open Bitboard (Square)

abbrev Replies (Move : Type) := Move → Bool → Int → Int → Option Int

def observed (r : Replies Move) : ChildSearch Move := fun m pv a z => (r m pv a z).getD 0

structure Result (Move : Type) where
  visited : List Move
  remaining : Queue Move
  snapshot : Snapshot
  handoff : Bool

def Result.outcome (r : Result Move) : Prefix.Serial.Outcome :=
  if r.handoff then .handoff r.snapshot.state else .returned r.snapshot.state.best

def Result.visit (m : Move) (r : Result Move) : Result Move := {r with visited := m :: r.visited}

/-- A cutoff permits replaying the unsearched suffix. At handoff only the serial prefix was run. -/
def Result.replayMoves (r : Result Move) : List Move :=
  r.visited ++ (if r.handoff then [] else r.remaining.pending)

private theorem visit_replayMoves (m : Move) (r : Result Move) :
    (r.visit m).replayMoves = m :: r.replayMoves := rfl

def issued (r : Replies Move) (q : Queue Move) (raw depth minSplit : Nat) (free busy : Bool)
    (alpha beta : Int) : HeaderAction :=
  let head := q.pending.head?
  let scout := head.bind (fun m => r m false (-(alpha + 1)) (-alpha))
  let wide := head.bind (fun m => r m true (-beta) (-alpha))
  headerAction q.idx (q.idx + q.pending.length) raw depth minSplit free busy alpha (scout.getD 0) (wide.getD 0)

/-- The schedule gives the availability of a helper and the result of the current move claim.
Only a child requested by the compiled header must have a recorded normal return. -/
def run (square : Move → Square) (replies : Replies Move) (schedule : Nat → Bool × Bool)
    (depth minSplit : Nat) (beta : Int) : Nat → Nat → Nat → Queue Move → Snapshot → Option (Result Move)
  | 0, _, _, _, _ => none
  | fuel + 1, i, raw, q, s =>
    let free := (schedule i).1
    let busy := (schedule i).2
    let alpha := s.state.alpha
    match issued replies q raw depth minSplit free busy alpha beta with
    | .finish => some ⟨[], q, s, false⟩
    | .split => some ⟨[], q, s, true⟩
    | .postpone _ fresh =>
      match step busy q with
      | some (none, next) => run square replies schedule depth minSplit beta fuel (i+1) fresh next s
      | _ => none
    | .child nextIdx scout wide score =>
      match step busy q with
      | some (some m, next) =>
        if (!scout || (replies m false (-(alpha+1)) (-alpha)).isSome) &&
            (!wide || (replies m true (-beta) (-alpha)).isSome) then
          let d := RootUpdateMir.decision (decide (nextIdx = 1)) alpha beta s.state.best score
          let updated := RootUpdateMir.after s (square m) score d
          if d.cuts then some ⟨[m], next, updated, false⟩
          else (run square replies schedule depth minSplit beta fuel (i+1) raw next updated).map (Result.visit m)
        else none
      | _ => none

/-- Start at the first candidate. Processing a move or postponing it consumes
    at least one unit of the queue budget. The last iteration observes the exit. -/
def fromMoves (square : Move → Square) (replies : Replies Move) (schedule : Nat → Bool × Bool)
    (depth minSplit : Nat) (alpha beta : Int) (moves : List Move) : Option (Result Move) :=
  let q : Queue Move := ⟨0,moves,[]⟩
  run square replies schedule depth minSplit beta (q.budget + 1) 0 moves.length q (RootReplay.start alpha)

private theorem recorded {r : Replies Move} {done_ : ChildRelation Move}
    (source : ∀ m cp a z v, r m cp a z = some v → done_ m cp a z v)
    (m : Move) (cp : Bool) (a z : Int) (ready : (r m cp a z).isSome = true) :
    done_ m cp a z (observed r m cp a z) := by
  cases h : r m cp a z with
  | none => simp [h] at ready
  | some v => simpa [observed, h] using source m cp a z v h

theorem refines (square : Move → Square) (r : Replies Move) (done_ : ChildRelation Move)
    (source : ∀ m cp a z v, r m cp a z = some v → done_ m cp a z v)
    (schedule : Nat → Bool × Bool) (depth minSplit : Nat) (beta : Int)
    (fuel i raw : Nat) (q : Queue Move) (s : Snapshot) (res : Result Move)
    (cursor : raw - q.idx = q.fresh.length) (count : s.seen.length = q.idx)
    (returned : run square r schedule depth minSplit beta fuel i raw q s = some res) :
    Prefix.Serial.Run done_ (observed r) true beta q s.state res.visited res.remaining res.outcome ∧
      res.snapshot = RootReplay.serial square (observed r) beta res.replayMoves s ∧
      (res.handoff = true → 0 < res.remaining.idx) := by
  induction fuel generalizing i raw q s res with
  | zero => simp [run] at returned
  | succ fuel ih =>
    let free := (schedule i).1
    let busy := (schedule i).2
    let alpha := s.state.alpha
    let scout := (q.pending.head?.bind (fun m => r m false (-(alpha+1)) (-alpha))).getD 0
    let wide := (q.pending.head?.bind (fun m => r m true (-beta) (-alpha))).getD 0
    cases act : issued r q raw depth minSplit free busy alpha beta with
    | finish =>
      rw [run, act] at returned
      simp only [Option.some.injEq] at returned
      subst res
      change headerAction _ _ _ _ _ _ _ _ scout wide = .finish at act
      rw [header_normalize q.idx _ raw, cursor] at act
      have empty := (header_finish_iff ..).mp act
      exact ⟨.exhausted empty, by simp [Result.replayMoves, empty, RootReplay.serial], by simp⟩
    | split =>
      rw [run, act] at returned
      simp only [Option.some.injEq] at returned
      subst res
      refine ⟨.handoff, rfl, fun _ => ?_⟩
      change 0 < q.idx
      by_cases first : q.idx = 0
      · change headerAction _ _ _ _ _ _ _ _ scout wide = .split at act
        rw [headerAction_eq] at act
        simp [first, canHandoff, PvsPostpone.shouldPostpone] at act
        split at act <;> contradiction
      · omega
    | postpone idx fresh =>
      have action := act
      change headerAction _ _ _ _ _ _ _ _ scout wide = .postpone idx fresh at action
      rw [header_normalize q.idx _ raw, cursor] at action
      obtain ⟨next, stepEq, _, freshEq, _⟩ := header_postpone q depth minSplit free busy alpha scout wide action
      have claimed : busy = true := by
        unfold headerAction at action
        cases h : busy <;> simp [h] at action <;> grind
      rw [run, act] at returned
      rw [stepEq] at returned
      have idxEq := PvsPostpone.step_idx stepEq
      simp at idxEq
      obtain ⟨trace, same, positive⟩ := ih _ _ _ _ _ (by omega) (idxEq ▸ count) returned
      exact ⟨.postpone (claimed ▸ stepEq) trace, same, positive⟩
    | child n needsScout needsWide v =>
      have action := act
      change headerAction _ _ _ _ _ _ _ _ scout wide = .child n needsScout needsWide v at action
      rw [header_normalize q.idx _ raw, cursor] at action
      obtain ⟨m, next, stepEq, head, nextIdx, _⟩ := header_take q depth minSplit free busy alpha scout wide action
      have used : scout = observed r m false (-(alpha+1)) (-alpha) ∧ wide = observed r m true (-beta) (-alpha) := by
        simp [scout, wide, observed, head]
      rw [used.1, used.2] at action
      have first : decide (n = 1) = s.seen.isEmpty := by
        have eq := PvsPostpone.step_idx stepEq
        cases found : s.seen <;> simp [found] at count ⊢ <;> simp at eq <;> omega
      let d := RootUpdateMir.decision (decide (n = 1)) alpha beta s.state.best v
      let updated := RootUpdateMir.after s (square m) v d
      have stepState := RootUpdateMir.agrees s (square m) beta v
      have after : updated = s.record beta ⟨square m,alpha,v⟩ := by simpa [updated,d,first,alpha] using stepState.1
      have cutoff : d.cuts = Threading.SplitWindow.marksCutoff true beta s.state v := by
        simpa [d,first,alpha] using stepState.2.1
      have firstCall : decide (q.idx = 0) = (q.idx == 0) := by by_cases h : q.idx = 0 <;> simp [h]
      by_cases scoutOK : needsScout = false ∨ (r m false (-(alpha+1)) (-alpha)).isSome = true
      · by_cases wideOK : needsWide = false ∨ (r m true (-beta) (-alpha)).isSome = true
        · have gates : (!needsScout || (r m false (-(alpha+1)) (-alpha)).isSome) = true ∧
              (!needsWide || (r m true (-beta) (-alpha)).isSome) = true := by simpa using And.intro scoutOK wideOK
          obtain ⟨_, calls, eq⟩ := header_child (observed r) done_ q.idx _ _ depth minSplit free busy m alpha beta
            needsScout needsWide v action (fun needed => recorded source _ _ _ _ (scoutOK.resolve_left (by simp [needed])))
              (fun needed => recorded source _ _ _ _ (wideOK.resolve_left (by simp [needed])))
          rw [firstCall] at calls eq
          have firstScore : v = PvsMoveLoop.moveScore true s.seen.isEmpty (observed r) m s.state.alpha beta := by
            have isFirst : (q.idx == 0) = s.seen.isEmpty := by
              cases xs : s.seen <;> simp [xs] at count ⊢ <;> omega
            simpa only [isFirst, alpha] using eq
          cases stops : d.cuts with
          | true =>
            rw [run, act, stepEq] at returned
            dsimp only at returned
            rw [gates.1, gates.2] at returned
            simp only [Bool.and_self, ite_true] at returned
            rw [stops] at returned
            simp at returned
            subst res
            have cut := cutoff.symm.trans stops
            have improves := (of_decide_eq_true cut).1
            have best : updated.state.best = v := by
              rw [after]
              exact Int.max_eq_right (Int.le_of_lt improves)
            constructor
            · change Prefix.Serial.Run _ _ _ _ _ _ [m] next (.returned updated.state.best)
              rw [best, eq]
              exact .cutoff stepEq calls (eq ▸ cut)
            · refine ⟨?_, by simp⟩
              change updated = RootReplay.serial square (observed r) beta (m :: next.pending) s
              simpa only [RootReplay.serial, ← firstScore, cut, ite_true] using after
          | false =>
            rw [run, act, stepEq] at returned
            dsimp only at returned
            rw [gates.1, gates.2] at returned
            simp only [Bool.and_self, ite_true] at returned
            rw [stops] at returned
            simp at returned
            obtain ⟨tail, rec, rfl⟩ := returned
            have idxEq := PvsPostpone.step_idx stepEq
            have nextCount : updated.seen.length = next.idx := by simp [updated, RootUpdateMir.after, idxEq, count]
            obtain ⟨more, same, positive⟩ := ih _ _ _ _ _ (by simpa using cursor_step q next raw cursor stepEq) nextCount rec
            constructor
            · apply Prefix.Serial.Run.searched stepEq calls (eq ▸ cutoff.symm.trans stops)
              change Prefix.Serial.Run done_ (observed r) true beta next _ tail.visited tail.remaining tail.outcome
              change Prefix.Serial.Run done_ (observed r) true beta next updated.state tail.visited tail.remaining tail.outcome at more
              rw [after] at more
              simpa only [RootReplay.Snapshot.record, eq, alpha] using more
            · refine ⟨?_, positive⟩
              rw [visit_replayMoves, RootReplay.serial, ← firstScore, cutoff.symm.trans stops]
              exact same.trans (congrArg (RootReplay.serial square (observed r) beta tail.replayMoves) after)
        · have missing : (!needsWide || (r m true (-beta) (-alpha)).isSome) = false := by simpa using wideOK
          rw [run, act, stepEq] at returned
          dsimp only at returned
          rw [missing] at returned
          simp at returned
      · have missing : (!needsScout || (r m false (-(alpha+1)) (-alpha)).isSome) = false := by simpa using scoutOK
        rw [run, act, stepEq] at returned
        dsimp only at returned
        rw [missing] at returned
        simp at returned

open PvsBoardTree (Position GeneratedSquare build play)

theorem returnedAttempt {mc : Bitboard.Moves.X86Config} {fc : Root.FlipConfig} {b : Position}
    {order : List (GeneratedSquare mc b.val)} {replies : Replies (GeneratedSquare mc b.val)}
    {result : Result (GeneratedSquare mc b.val)} {completed : ChildResults}
    (ordered : order.Perm (Root.choices mc b roots idx))
    (children : ∀ m cp a z v, replies m cp a z = some v →
      completed .serial (build mc fc (play mc fc b m).val) cp a z v)
    (normal : run Subtype.val replies schedule depth minSplit beta fuel 0 order.length
      ⟨0, order, []⟩ (RootReplay.start alpha) = some result)
    (finished : result.handoff = false) :
    RootEvidence.AttemptWithUses completed mc fc b roots idx alpha beta
      result.snapshot.state.best result.snapshot.scores := by
  obtain ⟨trace, same, _⟩ := refines _ _ _ children _ _ _ _ _ _ _ _ _ _ (by simp) rfl normal
  have outcome : result.outcome = .returned result.snapshot.state.best := by simp [Result.outcome, finished]
  rw [outcome] at trace
  have candidates : (⟨0, order, []⟩ : Queue _).pending.Perm (Root.choices mc b roots idx) := by
    simpa [Queue.pending] using ordered
  have attempt := RootEvidence.AttemptWithUses.ofSerialRun (fc := fc) (completed := completed)
    (q := ⟨0, order, []⟩) candidates rfl trace
  have rows := congrArg Snapshot.scores same
  simpa only [rows, Result.replayMoves, finished, Bool.false_eq_true, ite_false] using attempt

private theorem serial_map (label : Nat → Move) (square : Move → Square) (search : ChildSearch Move)
    (beta : Int) (indices : List Nat) (s : Snapshot) :
    RootReplay.serial (square ∘ label) (fun i => search (label i)) beta indices s =
      RootReplay.serial square search beta (indices.map label) s := by
  induction indices generalizing s with
  | nil => rfl
  | cons i rest ih =>
    let v := PvsMoveLoop.moveScore true s.seen.isEmpty search (label i) s.state.alpha beta
    let next := s.record beta ⟨square (label i), s.state.alpha, v⟩
    let cut := Threading.SplitWindow.marksCutoff true beta s.state v
    change (if cut then next else RootReplay.serial (square ∘ label) (fun j => search (label j)) beta rest next) =
      (if cut then next else RootReplay.serial square search beta (rest.map label) next)
    rw [ih]

theorem handoffAttempt {mc : Bitboard.Moves.X86Config} {fc : Root.FlipConfig} {b : Position}
    {order : List (GeneratedSquare mc b.val)} {replies : Replies (GeneratedSquare mc b.val)}
    {result : Result (GeneratedSquare mc b.val)} {completed : ChildResults}
    (ordered : order.Perm (Root.choices mc b roots idx))
    (children : ∀ m cp a z v, replies m cp a z = some v →
      completed .serial (build mc fc (play mc fc b m).val) cp a z v)
    (normal : run Subtype.val replies schedule depth minSplit beta fuel 0 order.length
      ⟨0, order, []⟩ (RootReplay.start alpha) = some result)
    (handoff : result.handoff = true) (first : GeneratedSquare mc b.val) (rest : List (GeneratedSquare mc b.val))
    (labels : first :: rest = result.visited ++ result.remaining.pending)
    (split : SplitExecution ((first :: rest).map (fun m => build mc fc (play mc fc b m).val)).length true alpha beta score)
    (start : split.start.toNat = result.remaining.idx)
    (entry : split.entry = result.snapshot.state)
    (reports : ReportCalls completed ((first :: rest).map (fun m => build mc fc (play mc fc b m).val)) split) :
    let sq := fun i : Nat => (((first :: rest)[i]?).getD first).val
    RootEvidence.AttemptWithUses completed mc fc b roots idx alpha beta score
      (RootReplay.Split.replay sq beta split.reports result.snapshot).scores := by
  intro sq
  obtain ⟨trace, same, started⟩ := refines _ _ _ children _ _ _ _ _ _ _ _ _ _ (by simp) rfl normal
  have positive : 0 < split.start.toNat := start ▸ started handoff
  have outcome : result.outcome = .handoff split.entry := by simp [Result.outcome, handoff, entry]
  rw [outcome] at trace
  have count : result.remaining.idx = result.visited.length := by simpa using trace.reproduced.2.1
  have candidates : (⟨0, order, []⟩ : Queue _).pending.Perm (Root.choices mc b roots idx) := by
    simpa [Queue.pending] using ordered
  have attempt := RootEvidence.AttemptWithUses.ofSerialHandoff (fc := fc) (completed := completed)
    (q := ⟨0, order, []⟩) first rest candidates rfl labels split start positive trace reports
  let move := fun i : Nat => ((first :: rest)[i]?).getD first
  have visited : (List.range split.start.toNat).map move = result.visited := by
    apply List.ext_getElem
    · simp [start, count]
    · intro i _ inside
      have bounded : i < (first :: rest).length := by rw [labels, List.length_append]; omega
      simp only [List.getElem_map, List.getElem_range, move, List.getElem?_eq_getElem bounded, Option.getD_some]
      have prefixAt := congrArg (fun (xs : List _) => xs[i]?) labels
      simpa [List.getElem?_eq_getElem bounded, List.getElem?_append_left inside, List.getElem?_eq_getElem inside] using prefixAt
  have before := serial_map move Subtype.val (observed replies) beta (List.range split.start.toNat) (RootReplay.start alpha)
  rw [visited] at before
  have actual : RootReplay.serial sq (fun i => observed replies (move i)) beta (List.range split.start.toNat)
      (RootReplay.start alpha) = result.snapshot := by
    exact before.trans (by simpa [Result.replayMoves, handoff] using same.symm)
  rw [← actual]
  exact attempt

end NeuralReversi.PvsParallel.RootOwnerMir

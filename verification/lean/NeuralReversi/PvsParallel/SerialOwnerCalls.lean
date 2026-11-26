import NeuralReversi.PvsParallel.SerialCalls.ScoreWords
import NeuralReversi.PvsParallel.SerialCalls.Ready
import NeuralReversi.PvsParallel.RootOwnerMir.Serial

namespace NeuralReversi.PvsParallel.SerialOwnerCalls

open SerialCalls (MoveRequest Program)
open RootReplay (Snapshot)
open RootOwnerMir (Replies)
open RootOwnerMir.Slice (Visit)
open Bitboard (Square)

abbrev Result (Move : Type) := List (Visit Move) × Snapshot

/-- The two MIR child-call sites, in instruction order. The scout word controls
    whether the second site is reached for this move. -/
private def one (first : Bool) (m : Move) (a z : Int)
    (next : Visit Move → Int → Program (MoveRequest Move) α) : Program (MoveRequest Move) α :=
  SerialCalls.Sites.ownerWith true first a ⟨m, false, -(a + 1), -a⟩ ⟨m, true, -z, -a⟩
    fun scout wide score => next ⟨m, a, scout, wide⟩ score

private theorem one_eq (first : Bool) (m : Move) (a z : Int)
    (next : Visit Move → Int → Program (MoveRequest Move) α) :
    one first m a z next =
  if first then
    .call ⟨m, true, -z, -a⟩ fun wide => next ⟨m, a, none, some wide⟩ (-wide)
  else
    .call ⟨m, false, -(a + 1), -a⟩ fun scout =>
      if a < -scout then
        .call ⟨m, true, -z, -a⟩ fun wide => next ⟨m, a, some scout, some wide⟩ (-wide)
      else next ⟨m, a, some scout, none⟩ (-scout) := by
  simp [one, SerialCalls.Sites.ownerWith]

private def loop (square : Move → Square) (beta : Int) :
    List Move → Nat → Snapshot → List (Visit Move) → Program (MoveRequest Move) (Result Move)
  | [], _, s, rev => .pure (rev.reverse, s)
  | m :: rest, idx, s, rev =>
    one (idx == 0) m s.state.alpha beta fun visit score =>
      let d := RootUpdateMir.decision (decide (idx + 1 = 1)) s.state.alpha beta s.state.best score
      let updated := RootUpdateMir.after s (square m) score d
      if d.cuts then .pure ((visit :: rev).reverse, updated)
      else loop square beta rest (idx + 1) updated (visit :: rev)

def run (square : Move → Square) (beta : Int) (moves : List Move) (s : Snapshot) :
    Program (MoveRequest Move) (Result Move) :=
  loop square beta moves s.seen.length s []

def reply (r : Replies Move) (q : MoveRequest Move) := r q.move q.pv q.alpha q.beta

/-- Chronological call operands and return words for one completed loop iteration. -/
def words (beta : Int) (v : Visit Move) : List (MoveRequest Move × Int) :=
  (v.scout.map fun w => (⟨v.move, false, -(v.alpha + 1), -v.alpha⟩, w)).toList ++
  (v.wide.map fun w => (⟨v.move, true, -beta, -v.alpha⟩, w)).toList

private theorem eval_loop (square : Move → Square) (beta : Int) (r : Replies Move)
    (depth limit : Nat) (pending prior : List Move) (raw i : Nat) (s : Snapshot) (rev : List (Visit Move)) :
    ((loop square beta pending prior.length s rev).eval (reply r)).map (fun v => (v.2, false)) =
      (RootOwnerMir.Slice.run square r RootOwnerMir.Serial.schedule depth limit beta i prior.length raw
        (prior ++ pending) s).map (fun v => (v.snapshot, v.handoff)) := by
  induction pending generalizing prior i s rev with
  | nil =>
    have done_ : RootOwnerMir.Slice.header r (prior ++ []) prior.length raw depth limit false false s.state.alpha beta =
        .finish := by simp [RootOwnerMir.Slice.header, headerAction_eq]
    change RootOwnerMir.Slice.header r (prior ++ []) prior.length raw depth limit
      (RootOwnerMir.Serial.schedule i).1 (RootOwnerMir.Serial.schedule i).2 s.state.alpha beta = _ at done_
    rw [RootOwnerMir.Slice.run, done_]
    rfl
  | cons m rest ih =>
    have current : (prior ++ m :: rest)[prior.length]? = some m := by simp
    let a := s.state.alpha
    let scout := r m false (-(a+1)) (-a)
    let wide := r m true (-beta) (-a)
    have act : RootOwnerMir.Slice.header r (prior ++ m :: rest) prior.length raw depth limit false false a beta =
        (if prior.length = 0 then .child (prior.length+1) false true (-wide.getD 0)
        else if a < -scout.getD 0 then .child (prior.length+1) true true (-wide.getD 0)
        else .child (prior.length+1) true false (-scout.getD 0)) := by
      simp [RootOwnerMir.Slice.header, headerAction_eq, canHandoff, PvsPostpone.shouldPostpone, scout, wide]
    have recur (visit : Visit Move) (updated : Snapshot) :
        ((loop square beta rest (prior.length + 1) updated (visit :: rev)).eval (reply r)).map (fun v => (v.2, false)) =
          (RootOwnerMir.Slice.run square r RootOwnerMir.Serial.schedule depth limit beta (i + 1) (prior.length + 1)
            raw (prior ++ m :: rest) updated).map (fun v => (v.snapshot, v.handoff)) := by
      simpa only [List.length_append, List.length_cons, List.length_nil, List.append_assoc,
        List.singleton_append] using ih (prior ++ [m]) (i + 1) updated (visit :: rev)
    have finish (visit : Visit Move) (updated : Snapshot) (cuts : Bool) :
        ((if cuts then Program.pure ((visit :: rev).reverse, updated)
          else loop square beta rest (prior.length + 1) updated (visit :: rev)).eval (reply r)).map (fun v => (v.2, false)) =
          (if cuts then some ⟨prior ++ m :: rest, prior.length+1, updated, false⟩
          else RootOwnerMir.Slice.run square r RootOwnerMir.Serial.schedule depth limit beta (i+1)
            (prior.length+1) raw (prior ++ m :: rest) updated).map (fun v => (v.snapshot, v.handoff)) := by
      cases cuts
      · exact recur visit updated
      · rfl
    change RootOwnerMir.Slice.header r (prior ++ m::rest) prior.length raw depth limit
      (RootOwnerMir.Serial.schedule i).1 (RootOwnerMir.Serial.schedule i).2 s.state.alpha beta = _ at act
    rw [RootOwnerMir.Slice.run, act]
    rw [loop, one_eq]
    by_cases first : prior.length = 0
    · rw [ite_eq_left first, ite_eq_left (beq_iff_eq.mpr first), current]
      cases w : r m true (-beta) (-s.state.alpha) with
      | none => simp [Program.eval, reply, wide, a, w]
      | some result =>
        simpa [Program.eval, reply, wide, a, w] using finish ⟨m,s.state.alpha,none,some result⟩
          (RootUpdateMir.after s (square m) (-result) (RootUpdateMir.decision (decide (prior.length+1=1))
            s.state.alpha beta s.state.best (-result)))
          (RootUpdateMir.decision (decide (prior.length+1=1)) s.state.alpha beta s.state.best (-result)).cuts
    · rw [ite_eq_right first, ite_eq_right (fun h => first (beq_iff_eq.mp h)), current]
      cases sc : r m false (-(s.state.alpha+1)) (-s.state.alpha) with
      | none =>
        by_cases raises : a < -scout.getD 0
        · rw [ite_eq_left raises]
          simp [Program.eval, reply, sc]
        · rw [ite_eq_right raises]
          simp [Program.eval, reply, sc]
      | some v =>
        by_cases raises : s.state.alpha < -v
        · have good : a < -scout.getD 0 := by simpa [scout, a, sc] using raises
          rw [ite_eq_left good]
          cases w : r m true (-beta) (-s.state.alpha) with
          | none => simp [Program.eval, reply, sc, w, raises]
          | some pv =>
            simpa [Program.eval, reply, wide, a, sc, w, raises] using
              finish ⟨m,s.state.alpha,some v,some pv⟩
                (RootUpdateMir.after s (square m) (-pv) (RootUpdateMir.decision (decide (prior.length+1=1))
                  s.state.alpha beta s.state.best (-pv)))
                (RootUpdateMir.decision (decide (prior.length+1=1)) s.state.alpha beta s.state.best (-pv)).cuts
        · have low : ¬a < -scout.getD 0 := by simpa [scout, a, sc] using raises
          rw [ite_eq_right low]
          simpa [Program.eval, reply, scout, a, sc, raises] using
            finish ⟨m,s.state.alpha,some v,none⟩
              (RootUpdateMir.after s (square m) (-v) (RootUpdateMir.decision (decide (prior.length+1=1))
                s.state.alpha beta s.state.best (-v)))
              (RootUpdateMir.decision (decide (prior.length+1=1)) s.state.alpha beta s.state.best (-v)).cuts

theorem eval_handoff (square : Move → Square) (r : Replies Move) (depth limit : Nat) (alpha beta : Int)
    (moves : List Move) :
    ((run square beta moves (RootReplay.start alpha)).eval (reply r)).map (fun v => (v.2, false)) =
      (RootOwnerMir.Slice.run square r RootOwnerMir.Serial.schedule depth limit beta 0 0 moves.length moves
        (RootReplay.start alpha)).map (fun v => (v.snapshot, v.handoff)) :=
  eval_loop square beta r depth limit moves [] moves.length 0 (RootReplay.start alpha) []


theorem eval_agrees (square : Move → Square) (r : Replies Move) (depth limit : Nat) (alpha beta : Int)
    (moves : List Move) :
    ((run square beta moves (RootReplay.start alpha)).eval (reply r)).map Prod.snd =
      (RootOwnerMir.Slice.run square r RootOwnerMir.Serial.schedule depth limit beta 0 0 moves.length moves
        (RootReplay.start alpha)).map RootOwnerMir.Slice.Exit.snapshot := by
  have agree := eval_handoff square r depth limit alpha beta moves
  simpa only [Option.map_map, Function.comp_def] using congrArg (Option.map Prod.fst) agree


private inductive Issued (m : Move) (a : Int) : Bool → Visit Move → Int → Prop
  | first (v : Int) : Issued m a true ⟨m, a, none, some v⟩ (-v)
  | high (v w : Int) (raises : a < -v) : Issued m a false ⟨m, a, some v, some w⟩ (-w)
  | low (v : Int) (fails : ¬a < -v) : Issued m a false ⟨m, a, some v, none⟩ (-v)

private theorem Issued.move_eq {first : Bool} (i : Issued m a first v score) : v.move = m := by cases i <;> rfl
private theorem Issued.alpha_eq {first : Bool} (i : Issued m a first v score) : v.alpha = a := by cases i <;> rfl

private theorem one_runs (ran : Program.Runs (one first m a z next) calls final) :
    ∃ v score tail, Issued m a first v score ∧ calls = words z v ++ tail ∧
      Program.Runs (next v score) tail final := by
  cases first with
  | true =>
    simp only [one_eq, ite_true] at ran
    cases ran with
    | call more => exact ⟨_, _, _, .first _, rfl, more⟩
  | false =>
    simp only [one_eq, Bool.false_eq_true, ite_false] at ran
    cases ran with
    | call more =>
      dsimp only at more
      split at more
      next high =>
        cases more with
        | call tail => exact ⟨_, _, _, .high _ _ high, rfl, tail⟩
      next low => exact ⟨_, _, _, .low _ low, rfl, more⟩

private theorem Issued.finite {first : Bool} (site : Issued m a first visit score)
    (children : ∀ t ∈ words beta visit, SerialCalls.ScoreWords.Finite t.2) : SerialCalls.ScoreWords.Finite score := by
  have negated (v : Int) (finite : SerialCalls.ScoreWords.Finite v) : SerialCalls.ScoreWords.Finite (-v) := by
    unfold SerialCalls.ScoreWords.Finite at *
    omega
  cases site with
  | first v => exact negated v (children (⟨m,true,-beta,-a⟩,v) (by simp [words]))
  | high sc v higher => exact negated v (children (⟨m,true,-beta,-a⟩,v) (by simp [words]))
  | low v lower => exact negated v (children (⟨m,false,-(a+1),-a⟩,v) (by simp [words]))

private theorem loop_finite (square : Move → Square) (beta : Int) (pending : List Move) (idx : Nat)
    (s : Snapshot) (rev : List (Visit Move)) (high : s.state.best < 32767)
    (present : -32767 < s.state.best ∨ pending ≠ [])
    (ran : Program.Runs (loop square beta pending idx s rev) cs out)
    (finite : ∀ t ∈ cs, SerialCalls.ScoreWords.Finite t.2) : SerialCalls.ScoreWords.Finite out.2.state.best := by
  induction pending generalizing idx s rev cs out with
  | nil =>
    cases ran
    exact ⟨present.resolve_right (by simp),high⟩
  | cons m rest ih =>
    rw [loop] at ran
    obtain ⟨v,score,tail,site,rfl,more⟩ := one_runs ran
    have value := site.finite (beta := beta) fun t ht => finite t (List.mem_append_left _ ht)
    let d := RootUpdateMir.decision (decide (idx+1=1)) s.state.alpha beta s.state.best score
    let updated := RootUpdateMir.after s (square m) score d
    have best : SerialCalls.ScoreWords.Finite updated.state.best := by
      change -32767 < (if decide (s.state.best < score) then score else s.state.best) ∧
        (if decide (s.state.best < score) then score else s.state.best) < 32767
      change -32767 < score ∧ score < 32767 at value
      by_cases h : s.state.best < score <;> simp [h] <;> omega
    change Program.Runs (if d.cuts then .pure ((v :: rev).reverse,updated)
      else loop square beta rest (idx+1) updated (v :: rev)) tail out at more
    split at more
    · cases more; exact best
    · exact ih _ _ _ best.2 (.inl best.1) more fun t ht => finite t (List.mem_append_right _ ht)

theorem run_finite {square : Move → Square} (nonempty : moves ≠ [])
    (ran : Program.Runs (run square beta moves (RootReplay.start alpha)) cs out)
    (finite : ∀ t ∈ cs, SerialCalls.ScoreWords.Finite t.2) : SerialCalls.ScoreWords.Finite out.2.state.best :=
  loop_finite square beta moves _ _ _ (by change (-32767 : Int) < 32767; decide) (.inr nonempty) ran finite

private theorem updated_window (s : Snapshot) (m : Square) (score beta : Int) (first : Bool)
    (w : Aspiration.Window s.state.alpha beta) :
    Aspiration.Window (RootUpdateMir.after s m score
      (RootUpdateMir.decision first s.state.alpha beta s.state.best score)).state.alpha beta := by
  obtain ⟨lo, middle, hi⟩ := w
  by_cases raises : s.state.alpha < score <;> by_cases better : s.state.best < score <;>
    by_cases inside : score < beta <;>
      simp [RootUpdateMir.after,RootUpdateMir.decision,RootUpdateMir.changes,raises,better,inside] <;>
      constructor <;> omega

private theorem Issued.nonempty {first : Bool} (site : Issued m a first v score) : words beta v ≠ [] := by
  cases site <;> simp [words]

private theorem loop_visits (pending : List Move) (idx : Nat) (s : Snapshot) (rev : List (Visit Move))
    (ran : Program.Runs (loop square beta pending idx s rev) calls result) :
    ∃ visits, result.1 = rev.reverse ++ visits ∧ calls = visits.flatMap (words beta) ∧
      (visits.map Visit.move) <+: pending ∧ (∀ v ∈ visits, words beta v ≠ []) ∧
      (Aspiration.Window s.state.alpha beta → ∀ v ∈ visits, Aspiration.Window v.alpha beta) := by
  induction pending generalizing idx s rev calls result with
  | nil =>
    change Program.Runs (.pure (rev.reverse, s)) calls result at ran
    cases ran
    exact ⟨[], by simp, rfl, ⟨[],rfl⟩, by simp, by simp⟩
  | cons m rest ih =>
    rw [loop] at ran
    obtain ⟨v, score, tail, site, rfl, more⟩ := one_runs ran
    have move := site.move_eq
    have occupied := site.nonempty (beta := beta)
    have alpha := site.alpha_eq
    dsimp only at more
    split at more
    · cases more
      exact ⟨[v], by simp, by simp, ⟨rest, by simp [move]⟩, by simpa using occupied, by simp [alpha]⟩
    · obtain ⟨vs, got, rfl, ⟨suffix, eq⟩, earlier, valid⟩ := ih _ _ _ more
      refine ⟨v :: vs, ?_, rfl, ⟨suffix, ?_⟩, ?_, ?_⟩
      · simpa [List.append_assoc] using got
      · simpa [move] using congrArg (List.cons m) eq
      · intro visit mem
        rcases List.mem_cons.mp mem with rfl | later
        · exact occupied
        · exact earlier visit later
      · intro w u mem
        rcases List.mem_cons.mp mem with rfl | later
        · rwa [alpha]
        · exact valid (updated_window _ _ _ _ _ w) _ later

private theorem loop_ready (square : Move → Square) (beta : Int) (moves : List Move) (idx : Nat)
    (s : Snapshot) (rev : List (Visit Move)) : Program.Ready (fun _ => True) (loop square beta moves idx s rev) := by
  induction moves generalizing idx s rev with
  | nil => exact .pure
  | cons m tail ih =>
    rw [loop, one_eq]
    have next (v : Visit Move) (value : Int) :
        let d := RootUpdateMir.decision (decide (idx+1=1)) s.state.alpha beta s.state.best value
        let updated := RootUpdateMir.after s (square m) value d
        Program.Ready (fun _ => True) (if d.cuts then .pure ((v :: rev).reverse,updated)
          else loop square beta tail (idx+1) updated (v :: rev)) := by
      dsimp only
      split
      · exact .pure
      · exact ih _ _ _
    split
    · exact .call trivial fun w => next _ (-w)
    · apply Program.Ready.call trivial
      intro sc
      split
      · exact .call trivial fun w => next _ (-w)
      · exact next _ (-sc)

theorem runs_visits {square : Move → Square} (ran : Program.Runs (run square beta moves s) calls result) :
    calls = result.1.flatMap (words beta) ∧ (result.1.map Visit.move) <+: moves ∧
      (Aspiration.Window s.state.alpha beta → ∀ v ∈ result.1, Aspiration.Window v.alpha beta) := by
  obtain ⟨vs, found, rc, p, _, valid⟩ := loop_visits _ _ _ _ ran
  have h : result.1 = vs := by simpa using found
  exact h.symm ▸ ⟨rc,p,valid⟩

theorem runs_nonempty {sq : Move → Square} (ran : Program.Runs (run sq beta moves s) calls result) :
    ∀ v ∈ result.1, words beta v ≠ [] := by
  obtain ⟨vs, found, _, _, full, _⟩ := loop_visits _ _ _ _ ran
  have eq : result.1 = vs := by simpa using found
  exact eq.symm ▸ full

private theorem words_member {v : Visit Move} (mem : t ∈ words beta v) :
    t.1.move = v.move ∧ t.1.beta = -v.alpha ∧
      t.1.alpha = (if t.1.pv then -beta else -(v.alpha + 1)) := by
  cases sc : v.scout <;> cases w : v.wide <;>
    simp [words, sc, w] at mem <;> first
    | (rcases mem with rfl | rfl <;> exact ⟨rfl, rfl, rfl⟩)
    | (subst t; exact ⟨rfl, rfl, rfl⟩)

theorem runs_windows {square : Move → Square} (ran : Program.Runs (run square beta moves s) calls result)
    (w : Aspiration.Window s.state.alpha beta) (item) (mem : item ∈ calls) :
    Aspiration.Window item.1.alpha item.1.beta ∧ (item.1.pv = false → item.1.beta = item.1.alpha + 1) := by
  obtain ⟨rfl, _, valid⟩ := runs_visits ran
  obtain ⟨v, member, word⟩ := List.mem_flatMap.mp mem
  obtain ⟨_, upper, lower⟩ := words_member word
  rw [upper, lower]
  cases item.1.pv
  · exact ⟨PvsMoveLoop.scoutWindow (valid w v member), fun _ => by simp; omega⟩
  · exact ⟨PvsMoveLoop.reflectedWindow (valid w v member), by simp⟩

/-- Even if an earlier child changes alpha, every later call has a valid window. -/
theorem run_ready {square : Move → Square} {s : Snapshot} (window : Aspiration.Window s.state.alpha beta)
    (moves : List Move) :
    Program.Ready (fun q => Aspiration.Window q.alpha q.beta ∧ (q.pv = false → q.beta = q.alpha+1))
      (run square beta moves s) := by
  apply Program.Ready.of_runs (loop_ready square beta moves s.seen.length s [])
  intro calls result ran t mem
  exact runs_windows ran window t mem

private theorem keys_distinct (vs : List (Visit Move)) (distinct : (vs.map Visit.move).Nodup) :
    ((vs.flatMap (words beta)).map fun t => (t.1.move, t.1.pv)).Nodup := by
  induction vs with
  | nil => simp
  | cons v vs ih =>
    obtain ⟨fresh, rest⟩ := List.nodup_cons.mp distinct
    rw [List.flatMap_cons, List.map_append, List.nodup_append]
    refine ⟨?_, ih rest, ?_⟩
    · cases sc : v.scout <;> cases w : v.wide <;> simp [words, sc, w]
    · intro a mem b present same
      obtain ⟨x, xm, rfl⟩ := List.mem_map.mp mem
      obtain ⟨y, ym, rfl⟩ := List.mem_map.mp present
      obtain ⟨v', later, elem⟩ := List.mem_flatMap.mp ym
      have eq : v.move = v'.move :=
        (words_member xm).1.symm.trans ((congrArg Prod.fst same).trans (words_member elem).1)
      exact fresh (eq ▸ List.mem_map_of_mem later)

theorem runs_distinct {square : Move → Square} (ran : Program.Runs (run square beta moves s) calls result)
    (unique : moves.Nodup) : (calls.map fun t => (t.1.move, t.1.pv)).Nodup := by
  obtain ⟨rfl, pre, _⟩ := runs_visits ran
  exact keys_distinct _ (pre.sublist.nodup unique)

private theorem Issued.model {m : Move} {first : Bool} (site : Issued m a first v score)
    (search : PvsMoveLoop.ChildSearch Move) (done : PvsMoveLoop.ChildRelation Move)
    (supplied : ∀ t ∈ words beta v, search t.1.move t.1.pv t.1.alpha t.1.beta = t.2 ∧
      done t.1.move t.1.pv t.1.alpha t.1.beta t.2) :
    score = PvsMoveLoop.moveScore true first search m a beta ∧
      PvsMoveLoop.MoveCalls done search true first m a beta := by
  cases site with
  | first pv =>
    obtain ⟨eq, good⟩ := supplied (⟨m,true,-beta,-a⟩,pv) (by simp [words])
    dsimp only at eq good
    simp [PvsMoveLoop.moveScore, PvsMoveLoop.MoveCalls, eq, good]
  | high sc pv raises =>
    obtain ⟨scEq, scGood⟩ := supplied (⟨m,false,-(a+1),-a⟩,sc) (by simp [words])
    obtain ⟨pvEq, pvGood⟩ := supplied (⟨m,true,-beta,-a⟩,pv) (by simp [words])
    dsimp only at scEq scGood pvEq pvGood
    simp [PvsMoveLoop.moveScore, PvsMoveLoop.MoveCalls, scEq, pvEq, scGood, pvGood, raises]
  | low sc fails =>
    obtain ⟨scEq, scGood⟩ := supplied (⟨m,false,-(a+1),-a⟩,sc) (by simp [words])
    dsimp only at scEq scGood
    simp [PvsMoveLoop.moveScore, PvsMoveLoop.MoveCalls, scEq, scGood, fails]

private theorem loop_serial (pending : List Move) (idx : Nat) (s : Snapshot) (rev : List (Visit Move))
    (search : PvsMoveLoop.ChildSearch Move) (done : PvsMoveLoop.ChildRelation Move)
    (cursor : idx = s.seen.length) (ran : Program.Runs (loop square beta pending idx s rev) calls result)
    (supplied : ∀ t ∈ calls, search t.1.move t.1.pv t.1.alpha t.1.beta = t.2 ∧
      done t.1.move t.1.pv t.1.alpha t.1.beta t.2) :
    result.2 = RootReplay.serial square search beta pending s ∧
      PvsMoveLoop.ScanCalls done search true beta pending s.seen.isEmpty s.state.alpha s.state.best := by
  induction pending generalizing idx s rev calls result with
  | nil =>
    change Program.Runs (.pure (rev.reverse, s)) calls result at ran
    cases ran
    exact ⟨rfl,trivial⟩
  | cons m rest ih =>
    rw [loop] at ran
    obtain ⟨v, score, tail, site, rfl, more⟩ := one_runs ran
    have fst : (idx == 0) = s.seen.isEmpty := by rw [cursor]; cases s.seen <;> rfl
    have firstUpdate : decide (idx+1 = 1) = s.seen.isEmpty := by rw [cursor]; cases s.seen <;> rfl
    obtain ⟨same, invoked⟩ := site.model search done fun t h => supplied t (List.mem_append_left tail h)
    rw [fst] at same invoked
    obtain ⟨updated, mark, _, _⟩ := RootUpdateMir.agrees s (square m) beta score
    rw [← firstUpdate] at updated mark
    dsimp only at more
    split at more
    next cuts =>
      cases more
      have cut := mark.symm.trans cuts
      refine ⟨?_, invoked, ?_⟩
      · rw [RootReplay.serial, ← same, ite_eq_left cut, updated]
      · rw [← same, Prefix.nextScanState_eq, ite_eq_left cut]
        trivial
    next keep =>
      have go := mark.symm.trans (Bool.eq_false_iff.mpr keep)
      obtain ⟨final, completed⟩ := ih _ _ _ (by simp [RootUpdateMir.after, cursor]) more
        (fun t h => supplied t (List.mem_append_right _ h))
      refine ⟨?_, invoked, ?_⟩
      · rw [RootReplay.serial, ← same, ite_eq_right (by simp [go]), ← updated]
        exact final
      · rw [← same, Prefix.nextScanState_eq, ite_eq_right (by simp [go])]
        rw [updated] at completed
        exact completed

theorem runs_serial {square : Move → Square} (ran : Program.Runs (run square beta moves s) calls result)
    (search : PvsMoveLoop.ChildSearch Move) (done : PvsMoveLoop.ChildRelation Move)
    (supplied : ∀ t ∈ calls, search t.1.move t.1.pv t.1.alpha t.1.beta = t.2 ∧
      done t.1.move t.1.pv t.1.alpha t.1.beta t.2) :
    result.2 = RootReplay.serial square search beta moves s ∧
      PvsMoveLoop.ScanCalls done search true beta moves s.seen.isEmpty s.state.alpha s.state.best :=
  loop_serial _ _ _ _ search done rfl ran supplied

/-- The root's physical row store uses the return of the last call issued for this move. -/
def writeVisit (sq : Move → Square) (pv : Move → List Square)
    (state : Bool × List RootMoveStorage.Row) (v : Visit Move) : Bool × List RootMoveStorage.Row :=
  let score := -(v.wide.or v.scout).getD 0
  let visible := state.1 || decide (v.alpha < score)
  (false, RootSuffix.writeFirst (sq v.move) (fun row => RootSuffix.updateRow row score visible (pv v.move)) state.2)

private theorem Issued.word {first : Bool} (site : Issued m a first v score) :
    -(v.wide.or v.scout).getD 0 = score := by cases site <;> rfl

private theorem loop_rows (sq : Move → Square) (pv : Move → List Square) (pending : List Move)
    (idx : Nat) (s : Snapshot) (rev : List (Visit Move)) (search : PvsMoveLoop.ChildSearch Move)
    (cursor : idx = s.seen.length) (ran : Program.Runs (loop sq beta pending idx s rev) calls result)
    (initial : Bool × List RootMoveStorage.Row) (rows : List RootMoveStorage.Row)
    (prior : rev.reverse.foldl (writeVisit sq pv) initial = (s.seen.isEmpty, rows))
    (supplied : ∀ t ∈ calls, search t.1.move t.1.pv t.1.alpha t.1.beta = t.2) :
    (result.1.foldl (writeVisit sq pv) initial).2 = RootSerialMir.rowsAfter sq pv search beta pending s rows := by
  induction pending generalizing idx s rev calls result rows with
  | nil =>
    change Program.Runs (.pure (rev.reverse, s)) calls result at ran
    cases ran
    exact congrArg Prod.snd prior
  | cons m rest ih =>
    rw [loop] at ran
    obtain ⟨v, score, tail, site, rfl, more⟩ := one_runs ran
    have fst : (idx == 0) = s.seen.isEmpty := by rw [cursor]; cases s.seen <;> rfl
    have firstUpdate : decide (idx+1 = 1) = s.seen.isEmpty := by rw [cursor]; cases s.seen <;> rfl
    obtain ⟨same, _⟩ := site.model search (fun _ _ _ _ _ => True) fun t h =>
      ⟨supplied t (List.mem_append_left tail h), trivial⟩
    rw [fst] at same
    let d := RootUpdateMir.decision s.seen.isEmpty s.state.alpha beta s.state.best score
    let written := RootSuffix.writeFirst (sq m) (fun row => RootSuffix.updateRow row score d.visible (pv m)) rows
    have visible := (RootUpdateMir.agrees s (sq m) beta score).2.2.2
    have record : (v :: rev).reverse.foldl (writeVisit sq pv) initial = (false, written) := by
      simp only [List.reverse_cons, List.foldl_append, List.foldl_cons, List.foldl_nil, prior, writeVisit,
        site.word, site.move_eq, site.alpha_eq, written, d, visible]
    rw [RootSerialMir.rowsAfter, ← same]
    dsimp only at more
    rw [firstUpdate] at more
    split at more
    next cut =>
      cases more
      rw [ite_eq_left cut]
      exact congrArg Prod.snd record
    next keep =>
      rw [ite_eq_right keep]
      exact ih _ _ _ (by simp [RootUpdateMir.after, cursor]) more written record
        fun t h => supplied t (List.mem_append_right _ h)

theorem runs_rows {sq : Move → Square} (ran : Program.Runs (run sq beta moves s) calls result)
    (pv : Move → List Square) (rows : List RootMoveStorage.Row) (search : PvsMoveLoop.ChildSearch Move)
    (supplied : ∀ t ∈ calls, search t.1.move t.1.pv t.1.alpha t.1.beta = t.2) :
    (result.1.foldl (writeVisit sq pv) (s.seen.isEmpty, rows)).2 = RootSerialMir.rowsAfter sq pv search beta moves s rows :=
  loop_rows sq pv _ _ _ [] search rfl ran _ rows rfl supplied

end NeuralReversi.PvsParallel.SerialOwnerCalls

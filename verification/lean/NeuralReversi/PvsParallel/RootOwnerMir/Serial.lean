module
public import NeuralReversi.PvsParallel.RootOwnerMir.Slice.Calls
public import NeuralReversi.PvsParallel.RootSerialMir

@[expose] public section

namespace NeuralReversi.PvsParallel.RootOwnerMir.Serial

open Aspiration (Window)
open Bitboard (Square)
open PvsMoveLoop (ChildSearch ChildRelation MoveCalls ScanCalls moveScore scoreInf)
open Slice (Path Supplies Visit)
open RootReplay (Snapshot start serial)
open Threading.SplitWindow (marksCutoff)

def schedule : Nat → Bool × Bool := fun _ => (false, false)

private theorem issued_child (square : Move → Square) (search : ChildSearch Move)
    (done : ChildRelation Move) (prior : List Move) (m : Move) (rest : List Move)
    (s : Snapshot) (fresh depth limit : Nat) (beta score : Int) (ns nw : Bool)
    (cursor : prior.length = s.seen.length)
    (act : headerAction prior.length (prior.length + (m :: rest).length) fresh depth limit false false
      s.state.alpha (search m false (-(s.state.alpha+1)) (-s.state.alpha))
      (search m true (-beta) (-s.state.alpha)) = .child (prior.length+1) ns nw score)
    (calls : ∀ pv word, (Visit.record m s.state.alpha (search m false (-(s.state.alpha+1)) (-s.state.alpha))
      (search m true (-beta) (-s.state.alpha)) ns nw).word pv = some word →
      done m pv (if pv then -beta else -(s.state.alpha+1)) (-s.state.alpha) word) :
    let d := RootUpdateMir.decision (decide (prior.length+1 = 1)) s.state.alpha beta s.state.best score
    let updated := RootUpdateMir.after s (square m) score d
    MoveCalls done search true s.seen.isEmpty m s.state.alpha beta ∧
      score = moveScore true s.seen.isEmpty search m s.state.alpha beta ∧
      updated = s.record beta ⟨square m, s.state.alpha, score⟩ ∧
      d.cuts = marksCutoff true beta s.state score := by
  have first : decide (prior.length = 0) = s.seen.isEmpty := by rw [cursor]; cases s.seen <;> rfl
  have firstUpdate : decide (prior.length+1 = 1) = s.seen.isEmpty := by
    rw [cursor]
    cases s.seen <;> rfl
  obtain ⟨_, invoked, same⟩ := header_child search done prior.length _ fresh depth limit false false m
    s.state.alpha beta ns nw score act
    (fun h => calls false _ (by simp [Visit.word, Visit.record, h]))
    (fun h => calls true _ (by simp [Visit.word, Visit.record, h]))
  obtain ⟨updated, cut, _, _⟩ := RootUpdateMir.agrees s (square m) beta score
  simpa only [first, firstUpdate] using And.intro invoked (And.intro same (And.intro updated cut))

private theorem path_refines (square : Move → Square) (search : ChildSearch Move)
    (done : ChildRelation Move)
    (path : Path square (fun _ => search) schedule depth limit beta i prior pending fresh s visits out)
    (cursor : prior.length = s.seen.length)
    (calls : ∀ v ∈ visits, ∀ pv word, v.word pv = some word →
      done v.move pv (v.lower pv beta) (-v.alpha) word) :
    out.handoff = false ∧ out.snapshot = serial square search beta pending s ∧
      ScanCalls done search true beta pending s.seen.isEmpty s.state.alpha s.state.best := by
  induction path with
  | done => exact ⟨rfl, rfl, trivial⟩
  | split act =>
    change headerAction _ _ _ _ _ false false _ _ _ = _ at act
    simp [headerAction_eq, canHandoff, PvsPostpone.shouldPostpone] at act
    split at act <;> try contradiction
    split at act <;> contradiction
  | postpone act _ _ =>
    change headerAction _ _ _ _ _ false false _ _ _ = _ at act
    simp [headerAction_eq, canHandoff, PvsPostpone.shouldPostpone] at act
    split at act <;> try contradiction
    split at act <;> contradiction
  | cutoff act cuts =>
    have one := issued_child square search done _ _ _ _ _ _ _ _ _ _ _ cursor act
      (fun pv word used => calls _ List.mem_cons_self pv word used)
    obtain ⟨invoked, same, updated, mark⟩ := one
    have cut := mark.symm.trans cuts
    refine ⟨rfl, ?_, invoked, ?_⟩
    · rw [serial, ← same, ite_eq_left cut, updated]
    · rw [← same, Prefix.nextScanState_eq, ite_eq_left cut]
      trivial
  | next act keep more ih =>
    have one := issued_child square search done _ _ _ _ _ _ _ _ _ _ _ cursor act
      (fun pv word used => calls _ List.mem_cons_self pv word used)
    obtain ⟨invoked, same, updated, mark⟩ := one
    have go := mark.symm.trans keep
    obtain ⟨handoff, final, ran⟩ := ih (by simp [RootUpdateMir.after, cursor])
      (fun v member => calls v (List.mem_cons_of_mem _ member))
    refine ⟨handoff, ?_, invoked, ?_⟩
    · rw [serial, ← same, ite_eq_right (by simp [go]), ← updated]
      exact final
    · rw [← same, Prefix.nextScanState_eq, ite_eq_right (by simp [go])]
      rw [updated] at ran
      exact ran

private theorem visit_word (search : ChildSearch Move) (m : Move) (a beta : Int) (ns nw p : Bool)
    (got : (Visit.record m a (search m false (-(a+1)) (-a)) (search m true (-beta) (-a)) ns nw).word p = some word) :
    search m p (if p then -beta else -(a+1)) (-a) = word := by
  cases p <;>
    simp only [Visit.word, Visit.record, Bool.false_eq_true, ite_false, ite_true] at got ⊢ <;>
    split at got <;> cases got <;> rfl

private theorem path_words (search : ChildSearch Move)
    (path : Path sq (fun _ => search) sched depth limit beta i prior pending fresh s visits out) :
    ∀ v ∈ visits, ∀ p word, v.word p = some word → search v.move p (v.lower p beta) (-v.alpha) = word := by
  induction path with
  | done | split => simp
  | postpone _ _ ih => exact ih
  | cutoff =>
    intro v member p word got
    obtain rfl := List.mem_singleton.mp member
    exact visit_word search _ _ _ _ _ p (got := got)
  | next _ _ _ ih =>
    intro v member p word got
    rcases List.mem_cons.mp member with rfl | later
    · exact visit_word search _ _ _ _ _ p (got := got)
    · exact ih v later p word got

theorem owner_returns (sq : Move → Square) (replies : Replies Move) (done : ChildRelation Move)
    (depth limit : Nat) (alpha beta : Int) (moves : List Move) (w : Window alpha beta)
    (ih : ∀ m p a z, Window a z → (p = false → z = a + 1) →
      ∃ score, replies m p a z = some score ∧ done m p a z score) :
    ∃ visits out,
      Path sq (fun _ => observed replies) schedule depth limit beta 0 [] moves moves.length (start alpha) visits out ∧
      Supplies replies beta visits ∧
      RootOwnerMir.Slice.run sq replies schedule depth limit beta 0 0 moves.length moves (start alpha) = some out ∧
      out.handoff = false ∧ out.snapshot = serial sq (observed replies) beta moves (start alpha) ∧
      ScanCalls done (observed replies) true beta moves true alpha (-scoreInf) := by
  obtain ⟨visits, out, path⟩ := RootOwnerMir.Slice.path_exists sq (fun _ => observed replies) schedule
    depth limit beta 0 [] moves moves.length (start alpha)
  have operands := path.operands w (fun _ => True) (fun _ _ _ _ _ => trivial)
  have words := path_words (observed replies) path
  have actual (v : Visit Move) (member : v ∈ visits) (p : Bool) (word : Int) (got : v.word p = some word) :
      replies v.move p (v.lower p beta) (-v.alpha) = some word ∧
        done v.move p (v.lower p beta) (-v.alpha) word := by
    have window : Window (v.lower p beta) (-v.alpha) := by
      cases p
      · exact PvsMoveLoop.scoutWindow (operands v member).1
      · exact PvsMoveLoop.reflectedWindow (operands v member).1
    have null : p = false → -v.alpha = v.lower p beta + 1 := by
      intro n
      subst p
      simp [Visit.lower]
      omega
    obtain ⟨score, returned, proof⟩ := ih v.move p (v.lower p beta) (-v.alpha) window null
    have same := words v member p word got
    change (replies v.move p (v.lower p beta) (-v.alpha)).getD 0 = word at same
    rw [returned, Option.getD_some] at same
    exact same ▸ ⟨returned, proof⟩
  have supplied : Supplies replies beta visits := fun v mem p word got => (actual v mem p word got).1
  have replayed := path.replays replies supplied
  have refined := path_refines sq (observed replies) done path rfl (fun v mem p word got =>
    (actual v mem p word got).2)
  exact ⟨visits, out, path, supplied, replayed, refined⟩

end NeuralReversi.PvsParallel.RootOwnerMir.Serial

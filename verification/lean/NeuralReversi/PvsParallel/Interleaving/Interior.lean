import NeuralReversi.PvsParallel.Interleaving.WeakFrame
import NeuralReversi.AspirationMir.SerialTable.PvInterior

namespace NeuralReversi.PvsParallel.Interleaving.Interior

open Aspiration (Window)
open AspirationMir.SerialTable.PvInterior (InsideRange Child move_inside)
open PvsMoveLoop (ChildSearch ChildRelation MoveCalls moveScore scoreInf)
open Threading
open SplitWindow (State publish)
open Schedule (Task Job)
open SerialCalls.ScoreWords (Finite)

structure Good (a z : Int) (s : State) : Prop where
  window : Window s.alpha z
  cap : s.alpha ≤ max a s.best
  best : InsideRange a z s.best

private theorem start (win : Window a z) : Good a z ⟨a,-scoreInf⟩ :=
  ⟨win,Int.le_max_left ..,fun above => (Int.not_lt.mpr win.lower above).elim⟩

private theorem Good.score (init : Good a z s) (above : s.best < v)
    (localRange : InsideRange s.alpha z v) : InsideRange a z v := by
  intro lo hi
  exact localRange (by have := init.cap; omega) hi

private theorem Good.post (init : Good a z s) (localRange : InsideRange s.alpha z v) :
    Good a z (publish true z s v) := by
  have next := (SplitWindow.stateContract_proven true a z s [v] init.window init.cap).2
  refine ⟨next.1,next.2.1,?_⟩
  change InsideRange a z (max s.best v)
  by_cases rise : s.best < v
  · rw [Int.max_eq_right (Int.le_of_lt rise)]
    exact init.score rise localRange
  · rw [Int.max_eq_left (Int.le_of_not_gt rise)]
    exact init.best

private theorem ownerMove (run : MoveCalls done_ reply true first m a z)
    (win : Window a z) (child : ∀ m p l h v, Window l h → done_ m p l h v → Child m p l h v) :
    InsideRange a z (moveScore true first reply m a z) := by
  apply move_inside
  cases first with
  | true => exact child _ _ _ _ _ (PvsMoveLoop.reflectedWindow win) run
  | false => exact ⟨by simp [Child],fun raised => child _ _ _ _ _ (PvsMoveLoop.reflectedWindow win) (run.2 raised)⟩

private def atExit (a z : Int) : Prefix.Serial.Outcome → Prop
  | .returned score => InsideRange a z score
  | .handoff entry => Good a z entry

private theorem owner (run : Prefix.Serial.Run done_ reply pv z q s visited remaining out)
    (kind : pv = true) (init : Good a z s)
    (child : ∀ m p l h v, Window l h → done_ m p l h v → Child m p l h v) :
    atExit a z out := by
  subst pv
  induction run with
  | exhausted => exact init.best
  | handoff => exact init
  | postpone _ _ ih => exact ih init
  | cutoff _ calls cut =>
    exact init.score (of_decide_eq_true cut).1 (ownerMove calls init.window child)
  | searched _ calls _ _ ih => exact ih (init.post (ownerMove calls init.window child))

private theorem word (f : Finite v) : (BitVec.ofInt 32 v).toInt = v :=
  BitVec.toInt_ofInt_eq_self (by decide) (by have := f.1; omega) (by have := f.2; omega)

/-- Only a completed wide re-search can improve on the current locked alpha inside the
    window, even if the scout and re-search both read older alpha words. -/
private theorem job (j : Job) (t : Task) (ps : List Tickets.Post) (pv : t.pv = true)
    (entry : Good orig t.beta t.entry) (calls : j.Calls done_ t ps)
    (finite : ∀ i cp a z v, done_ i cp a z v → Finite v)
    (child : ∀ i cp a z v, Window a z → done_ i cp a z v → Child i cp a z v) :
    InsideRange (t.locked ps).alpha t.beta (j.post t ps).report.score := by
  let a := j.alpha t ps false
  let fresh := j.alpha t ps true
  let sc := j.reply j.index false (-(a+1)) (-a)
  let wd := j.reply j.index true (-t.beta) (-fresh)
  let i := j.input t ps
  have aw := alphaAt_valid t.pv orig t.beta t.entry ((ps.map Tickets.Post.report).map SplitResult.Report.score) j.scoutAt entry.window entry.cap
  have fw := alphaAt_valid t.pv orig t.beta t.entry ((ps.map Tickets.Post.report).map SplitResult.Report.score) j.wideAt entry.window entry.cap
  change Window a t.beta ∧ a ≤ (t.locked ps).alpha ∧ _ at aw
  change Window fresh t.beta ∧ fresh ≤ (t.locked ps).alpha ∧ _ at fw
  have represent : i.alpha.toInt = a := BitVec.toInt_ofInt_eq_self (by decide)
    (by have := aw.1.lower; omega) (by have := aw.1.ordered; have := aw.1.upper; omega)
  have sf : Finite sc := finite _ _ _ _ _ calls.1
  have scEq : i.scout.toInt = sc := word sf
  have neg : (-i.scout).toInt = -sc := (Solve2.neg_safe _ (by rw [scEq]; have := sf.1; omega)).trans (congrArg Neg.neg scEq)
  have test : SplitWorkerMir.retry i = SplitWindow.reSearch t.pv a sc := by
    simp only [SplitWorkerMir.retry,SplitWindow.reSearch,neg,represent]; rfl
  by_cases again : SplitWindow.reSearch t.pv a sc = true
  · have used : done_ j.index true (-t.beta) (-fresh) wd := calls.2 again
    have wf : Finite wd := finite _ _ _ _ _ used
    have wideEq : i.wide.toInt = wd := by
      change (BitVec.ofInt 32 (if SplitWindow.reSearch t.pv a sc then wd else 0)).toInt = wd
      rw [ite_eq_left again,word wf]
    change InsideRange _ _ (Worker.report i).score
    simp only [Worker.report,SplitWorkerMir.selected,SplitWorkerMir.issued,test,again,ite_true]
    rw [Solve2.neg_safe _ (by rw [wideEq]; have := wf.1; omega),wideEq]
    intro lo hi
    have bound : -PvsMoveLoop.scoreMax ≤ wd ∧ wd ≤ PvsMoveLoop.scoreMax := child _ _ _ _ _ (PvsMoveLoop.reflectedWindow fw.1) used rfl
      (by change -t.beta < wd; omega) (by change wd < -fresh; have := fw.2.1; omega)
    change -PvsMoveLoop.scoreMax ≤ -wd ∧ -wd ≤ PvsMoveLoop.scoreMax
    omega
  · have no := Bool.eq_false_iff.mpr again
    have small : -sc ≤ a := by simpa [SplitWindow.reSearch,pv] using no
    change InsideRange _ _ (Worker.report i).score
    simp only [Worker.report,SplitWorkerMir.selected,SplitWorkerMir.issued,test,no,Bool.false_eq_true,ite_false,neg]
    exact fun improves => (Int.not_lt.mpr (Int.le_trans small aw.2.1) improves).elim

private theorem workers (run : Schedule.RunWith outside t done_ len idx ownerId events s) (pv : t.pv = true)
    (entry : Good orig t.beta t.entry) (finite : ∀ i cp a z v, done_ i cp a z v → Finite v)
    (child : ∀ i cp a z v, Window a z → done_ i cp a z v → Child i cp a z v) : Good orig t.beta (t.locked s.reports) := by
  induction run with
  | initial => exact entry
  | book _ _ _ step ih | finish _ _ _ step ih => simpa [step.reports_eq,Tickets.Action.posts] using ih
  | @work actions before a after prior edge quiet cause check ih =>
    rw [edge.reports_eq]
    cases a <;> first | exact List.append_nil _ ▸ ih | skip
    rename_i worker p
    obtain ⟨j,rfl,calls⟩ := check p (by simp [Tickets.Action.posts])
    have next := ih.post (job j t _ pv entry calls finite child)
    simpa [Task.locked,SplitReturn.stateAfter,SplitWindow.scan,Tickets.Action.posts,pv] using next

theorem turn (path : Turn done_ moves true a z score) (win : Window a z)
    (finite : ∀ m cp lo hi v, done_ m cp lo hi v → Finite v)
    (child : ∀ m cp lo hi v, Window lo hi → done_ m cp lo hi v → Child m cp lo hi v) : InsideRange a z score := by
  cases path with
  | returned _ run => exact owner run rfl (start win) child
  | joined _ _ _ _ _ _ run _ j returned =>
    have early := owner run rfl (start win) child
    have bounds := workers j.path rfl early (fun _ => finite _) (fun _ => child _)
    exact (j.finished returned).2.2 ▸ bounds.best

theorem root {b : PvsBoardTree.Position} {done_ : ChildRelation (PvsBoardTree.GeneratedSquare mc b.val)}
    (trial : RootSchedule.Trial mc b initial idx a z done_ score rows) (win : Window a z)
    (finite : ∀ m cp lo hi v, done_ m cp lo hi v → Finite v)
    (child : ∀ m cp lo hi v, Window lo hi → done_ m cp lo hi v → Child m cp lo hi v) : InsideRange a z score := by
  cases trial with
  | owner r replies moves pv order source ran done_ =>
    obtain ⟨trace,_,_⟩ := RootOwnerMir.refines Subtype.val replies _ source _ _ _ _ _ _ _ _ _ _ (by simp) rfl ran
    have safe := owner trace rfl (start win) child
    simpa only [RootOwnerMir.Result.outcome,done_,Bool.false_eq_true,ite_false,atExit] using safe
  | split r replies moves pv order source ran handed positive width j used =>
    obtain ⟨trace,_,_⟩ := RootOwnerMir.refines Subtype.val replies _ source _ _ _ _ _ _ _ _ _ _ (by simp) rfl ran
    have early : Good a z r.snapshot.state := by
      simpa only [RootOwnerMir.Result.outcome,handed,ite_true,atExit] using owner trace rfl (start win) child
    have safe := workers j.path rfl early
      (fun _ cp l h v ⟨m,_,got⟩ => finite m cp l h v got)
      (fun _ cp l h v w ⟨m,_,got⟩ => child m cp l h v w got)
    exact (j.finished used).2.2 ▸ safe.best

/-- A selective PV may return an approximate score. Every interior return remains
    in the engine's disc-score range, including after nested split searches. -/
theorem weak (run : WeakFrame.Frame mc fc prior reads b pv a z v) (win : Window a z) (kind : pv = true)
    (finite : ∀ b p lo hi n, prior b p lo hi n → Finite n)
    (prev : ∀ b p lo hi n, Window lo hi → prior b p lo hi n → p = true → InsideRange lo hi n) :
    InsideRange a z v := by
  cases run with
  | localReturn w narrow body =>
    have valid := BoardFrame.preserves _ _ (fun _ _ _ _ _ _ _ h => h.elim) (fun _ _ h => h.elim)
      _ _ _ _ _ w narrow body
    intro lo hi
    rw [valid.exact lo hi]
    exact PvsMoveLoop.minimax_range _
  | pass _ _ returned =>
    intro lo hi
    have range := prev _ _ _ _ _ (PvsMoveLoop.reflectedWindow win) returned kind (by omega) (by omega)
    omega
  | leaf | table | enhanced | probability => contradiction
  | moves _ _ _ _ _ path =>
    subst pv
    exact turn path win (fun _ => finite _) (fun m p l h s w used => prev _ p l h s w used)

end NeuralReversi.PvsParallel.Interleaving.Interior

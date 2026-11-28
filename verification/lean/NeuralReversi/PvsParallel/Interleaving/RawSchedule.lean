module
public import NeuralReversi.PvsParallel.Interleaving.Mask

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.RawSchedule

open Aspiration (Window)
open PvsMoveLoop (ChildRelation ChildSearch)
open Schedule (Task Job)
open Tickets (Post)
open Threading
open AtomicWord (Read Write signed32At)
open SplitResult (Report)

structure Sample (hist : CutoffVisibility.History) (words : List Write) (t : Task) (posts : List Post) where
  first : Fin words.length
  last : Fin words.length
  load : Read words
  order : SplitWindow.AlphaRead hist load first last
  stores : SplitWindow.StoreWindow words first last t.pv t.beta t.entry ((posts.map Post.report).map Report.score)

def Sample.value (s : Sample hist words t posts) := signed32At words s.load.source

private theorem Sample.prefix (s : Sample hist words t posts) (w : Window t.entry.alpha t.beta)
    (cap : t.entry.alpha ≤ max orig t.entry.best) : ∃ n,
    s.value = alphaAt t.pv t.beta t.entry ((posts.map Post.report).map Report.score) n := by
  obtain ⟨n,_,eq⟩ := alphaAt_of_load s.order s.stores w cap
  exact ⟨n,eq⟩

/-- The actual relaxed loads supply the search operands. Recording another load is required
    exactly when the scout requests a wide child. Its score and the cut bit come from the
    worker instructions, evaluated under the split-point lock. -/
structure Commit (hist : CutoffVisibility.History) (words : List Write) (t : Task) (ps : List Post) where
  index : Nat
  scout : Sample hist words t ps
  wide : Option (Sample hist words t ps)
  child : ChildSearch Nat
  locked : SplitWorkerMir.Locked
  live : locked.stopped = false
  alpha : locked.alpha.toInt = (t.locked ps).alpha
  best : locked.best.toInt = (t.locked ps).best
  reload : wide.isSome = SplitWindow.reSearch t.pv scout.value
    (child index false (-(scout.value+1)) (-scout.value))

def Commit.fresh (c : Commit hist words t ps) := (c.wide.getD c.scout).value

def Commit.input (c : Commit hist words t ps) :=
  Worker.issue t.pv t.root t.cutNode t.depth c.index c.scout.value t.beta c.fresh c.child

def Commit.post (c : Commit hist words t ps) : Post :=
  ⟨Worker.report c.input, (SplitWorkerMir.commit c.input c.locked (SplitWorkerMir.issued c.input).chosen).cut⟩

def Commit.calls (c : Commit hist words t ps) (done_ : ChildRelation Nat) : Prop :=
  done_ c.index false (-(c.scout.value+1)) (-c.scout.value)
      (c.child c.index false (-(c.scout.value+1)) (-c.scout.value)) ∧
  (c.wide.isSome = true → done_ c.index true (-t.beta) (-c.fresh) (c.child c.index true (-t.beta) (-c.fresh)))

theorem Commit.job (c : Commit hist words t ps) (w : Window t.entry.alpha t.beta)
    (cap : t.entry.alpha ≤ max orig t.entry.best) (calls : c.calls done_) :
    ∃ j : Job, c.post = j.post t ps ∧ j.Calls done_ t ps := by
  obtain ⟨n,sc⟩ := c.scout.prefix w cap
  obtain ⟨m,re⟩ := (c.wide.getD c.scout).prefix w cap
  let j : Job := ⟨c.index,n,m,c.child⟩
  have a : c.scout.value = j.alpha t ps false := sc
  have f : c.fresh = j.alpha t ps true := re
  have i : c.input = j.input t ps := by simp only [Commit.input,Job.input,←a,←f,j]
  have beta : (j.input t ps).beta.toInt = t.beta := by
    change (BitVec.ofInt 32 t.beta).toInt = t.beta
    have lo := w.lower
    have lt := w.ordered
    have hi := w.upper
    exact BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
  refine ⟨j,?_,?_⟩
  · unfold Commit.post
    rw [i,(j.native t ps c.locked c.live c.alpha c.best beta).1]
    rfl
  · change _ ∧ _
    rw [←a,←f]
    exact ⟨calls.1,fun yes => calls.2 (c.reload.trans yes)⟩

abbrev Run (hist : CutoffVisibility.History) (words : List Write) (outside : Prop) (t : Task)
    (done_ : ChildRelation Nat) (len : Nat) (start : BitVec w) (owner : SplitCompletion.Worker) :=
  Retirement.ExecutionWith outside (fun s p => ∃ c : Commit hist words t s.reports, p = c.post ∧ c.calls done_) len start owner

theorem Run.schedule (run : Run hist words outside t done_ len start owner actions s)
    (w : Window t.entry.alpha t.beta) (cap : t.entry.alpha ≤ max orig t.entry.best) :
    Schedule.RunWith outside t done_ len start owner actions s := by
  induction run with
  | initial => exact .initial
  | work _ op quiet why checked ih =>
    refine .work ih op quiet why ?_
    intro p member
    obtain ⟨c,rfl,args⟩ := checked p member
    exact c.job w cap args
  | book i e _ op ih => exact .book i e ih op
  | finish i e _ op ih => exact .finish i e ih op

theorem join (o : Mask.Observed hist prior owner g next) (stop : SplitReturn.Observation.OwnerCheck hist)
    (run : Run hist words (SplitReturn.Observation.ExternalStop hist stop o.waitAt) t done_ len start owner g.actions s)
    (w : Window t.entry.alpha t.beta) (cap : t.entry.alpha ≤ max orig t.entry.best) :
    ∃ j : Join t done_ len start, j.state = s ∧ j.wait.read.event = o.waitAt ∧
      j.wait.read.source.val = prior.length+1+g.actions.length ∧ j.activation.actions = g.actions ∧
      j.answer = SplitReturn.checked stop.stopped (t.locked s.reports).best :=
  Mask.join o stop (run.schedule w cap)

end NeuralReversi.PvsParallel.Interleaving.RawSchedule

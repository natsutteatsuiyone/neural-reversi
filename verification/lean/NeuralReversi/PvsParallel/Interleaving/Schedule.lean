module
public import NeuralReversi.PvsParallel.Interleaving.Alpha
public import NeuralReversi.PvsParallel.Interleaving.Retirement
public import NeuralReversi.PvsParallel.Interleaving.Worker

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.Schedule

open Aspiration (Window SoundResult)
open PvsMoveLoop (Bound ChildRelation ChildSearch scoreMax)
open Threading
open SplitResult (Report)
open Tickets (Post)

structure Task where
  pv : Bool
  root : Bool
  cutNode : Bool
  depth : Nat
  beta : Int
  entry : SplitWindow.State

def reports (s : Tickets.State w) := s.reports.map Post.report

def Task.locked (t : Task) (posts : List Post) :=
  SplitReturn.stateAfter t.pv t.beta t.entry (posts.map Post.report)

/-- Prefix lengths encode each unlocked load independently. Recursive returns belong
    to this ticket, so equal requests in distinct workers may have different fail-soft results. -/
structure Job where
  index : Nat
  scoutAt : Nat
  wideAt : Nat
  reply : ChildSearch Nat

def Job.alpha (j : Job) (t : Task) (posts : List Post) (wide : Bool) :=
  alphaAt t.pv t.beta t.entry ((posts.map Post.report).map Report.score) (if wide then j.wideAt else j.scoutAt)

def Job.input (j : Job) (t : Task) (ps : List Post) :=
  Worker.issue t.pv t.root t.cutNode t.depth j.index (j.alpha t ps false) t.beta (j.alpha t ps true) j.reply

def Job.post (j : Job) (t : Task) (ps : List Post) : Post :=
  let r := Worker.report (j.input t ps)
  ⟨r,SplitWindow.marksCutoff t.pv t.beta (t.locked ps) r.score⟩

def Job.Calls (j : Job) (done_ : ChildRelation Nat) (t : Task) (ps : List Post) : Prop :=
  let a := j.alpha t ps false
  let f := j.alpha t ps true
  let sc := j.reply j.index false (-(a+1)) (-a)
  done_ j.index false (-(a+1)) (-a) sc ∧
    (SplitWindow.reSearch t.pv a sc = true →
      done_ j.index true (-t.beta) (-f) (j.reply j.index true (-t.beta) (-f)))

theorem Job.sound {value : Nat → Int} (j : Job) (t : Task) (ps : List Post) (calls : j.Calls done_ t ps)
    (w : Window t.entry.alpha t.beta) (cap : t.entry.alpha ≤ max original t.entry.best)
    (narrow : t.pv = false → t.beta = t.entry.alpha+1)
    (children : ∀ cp a b s, Window a b → (cp = false → b = a+1) →
      done_ j.index cp a b s → SoundResult (-(value j.index)) a b s) :
    let p := j.post t ps
    p.report.searchedAlpha ≤ (t.locked ps).alpha ∧
      Bound (value p.report.index) p.report.searchedAlpha t.beta p.report.score := by
  let words := (ps.map Post.report).map Report.score
  have sc := alphaAt_valid t.pv original t.beta t.entry words j.scoutAt w cap
  have wide := alphaAt_valid t.pv original t.beta t.entry words j.wideAt w cap
  have h := Worker.issue_spec (v := value j.index) t.pv t.root t.cutNode t.depth j.index
    (j.alpha t ps false) t.beta (j.alpha t ps true) j.reply sc.1 (fun _ => wide.1)
    (fun win => children _ _ _ _ win (by intro; omega) calls.1)
    (fun yes win => children _ _ _ _ win (by simp) (calls.2 yes))
  change Worker.Issued _ (j.input t ps) _ _ _ _ _ at h
  have usable := Worker.sound (j.input t ps) (t.locked ps).alpha (h.betaWord ▸ h.alphaWord ▸ sc.1)
    (h.alphaWord ▸ sc.2.1) (fun noPv => by rw [h.betaWord,h.alphaWord,show j.alpha t ps false = t.entry.alpha from sc.2.2 noPv]; exact narrow noPv)
    (fun yes => by rw [h.betaWord,h.freshWord yes]; exact ⟨wide.1,wide.2.1⟩) h.children
  exact ⟨usable.2,h.betaWord ▸ usable.1⟩

/-- Check the Rust commit's cutoff bit and the updated score registers against the
    same publication consumed by the scheduler. -/
theorem Job.native (j : Job) (t : Task) (ps : List Post) (l : SplitWorkerMir.Locked)
    (running : l.stopped = false) (a : l.alpha.toInt = (t.locked ps).alpha)
    (best : l.best.toInt = (t.locked ps).best) (beta : (j.input t ps).beta.toInt = t.beta) :
    let i := j.input t ps
    let update := SplitWorkerMir.commit i l (SplitWorkerMir.issued i).chosen
    update.cut = (j.post t ps).cut ∧
      SplitWorkerMir.lockedState l update = t.locked (ps ++ [j.post t ps]) := by
  constructor
  · rw [Worker.marks _ _ running,a,best,beta]; rfl
  · rw [Worker.publish _ _ running,a,best,beta]
    simp [Task.locked,SplitReturn.stateAfter,SplitWindow.scan,Job.post]
    rfl

/-- The scheduler supplies runnable events in any order. A report's index, score
    and cut bit are computed by the job whose ticket the worker holds. -/
abbrev RunWith (outside : Prop) (t : Task) (done_ : ChildRelation Nat) (len : Nat) (start : BitVec w)
    (owner : SplitCompletion.Worker) (mask : List SplitCompletion.Action) (s : Tickets.State w) :=
  Retirement.ExecutionWith outside (fun before p => ∃ j : Job, p = j.post t before.reports ∧
    j.Calls done_ t before.reports) len start owner mask s

abbrev Run (t : Task) (done_ : ChildRelation Nat) (len : Nat) (start : BitVec w)
    (owner : SplitCompletion.Worker) (mask : List SplitCompletion.Action) (s : Tickets.State w) :=
  RunWith True t done_ len start owner mask s

def AtPosts (P : List Post → Post → Prop) (ps : List Post) : Prop :=
  ∀ earlier p later, ps = earlier ++ p :: later → P earlier p

private theorem snoc (old : AtPosts P ps) (one : P ps last) : AtPosts P (ps ++ [last]) := by
  intro earlier r later eq
  rcases List.eq_nil_or_concat later with rfl | ⟨ys,z,rfl⟩
  · obtain ⟨rfl,same⟩ := List.append_inj eq (by simpa using congrArg List.length eq)
    cases same
    exact one
  · have split : ps ++ [last] = (earlier ++ r :: ys) ++ [z] := by simpa [List.concat_eq_append,List.append_assoc] using eq
    exact old earlier r ys (List.append_inj_left' split rfl)

theorem Run.posts {value : Nat → Int} {start : BitVec w} (run : RunWith outside t done_ len start owner mask s)
    (fits : len < 2^w) (window : Window t.entry.alpha t.beta) (cap : t.entry.alpha ≤ max original t.entry.best)
    (narrow : t.pv = false → t.beta = t.entry.alpha+1)
    (children : ∀ i, start.toNat ≤ i → i < len → ∀ cp a b score, Window a b →
      (cp = false → b = a+1) → done_ i cp a b score → SoundResult (-(value i)) a b score) :
    AtPosts (fun before p => p.report.searchedAlpha ≤ (t.locked before).alpha ∧
      Bound (value p.report.index) p.report.searchedAlpha t.beta p.report.score ∧
      p.cut = SplitWindow.marksCutoff t.pv t.beta (t.locked before) p.report.score) s.reports := by
  induction run with
  | initial => simp [AtPosts,Tickets.initial]
  | book _ _ _ step ih | finish _ _ _ step ih => simpa [step.reports_eq,Tickets.Action.posts] using ih
  | @work actions before a after prior step effect cause check ih =>
    rw [step.reports_eq]
    cases a <;> first | exact List.append_nil _ ▸ ih | skip
    rename_i worker p
    obtain ⟨j,rfl,calls⟩ := check p (by simp [Tickets.Action.posts])
    apply snoc ih
    have valid := (prior.tickets.step step).valid fits (p := j.post t before.reports)
      (by rw [step.reports_eq]; simp [Tickets.Action.posts])
    obtain ⟨le,bound⟩ := j.sound t before.reports calls window cap narrow (children j.index valid.1 valid.2)
    exact ⟨le,bound,rfl⟩

structure Result (t : Task) (original : Int) (value : Nat → Int) (start len : Nat)
    (prefixValue : Int) (s : Tickets.State w) : Prop where
  distinct : ((reports s).map Report.index).Nodup
  valid : ∀ r ∈ reports s, r.index ∈ SplitResult.slots start len
  steps : SplitReturn.ReportSteps t.pv t.beta t.entry value (reports s)
  bound : Bound (SplitResult.nodeValue value start len prefixValue) original t.beta (t.locked s.reports).best

theorem Run.accepted {value : Nat → Int} {start : BitVec w} (run : RunWith outside t done_ len start owner mask s)
    (fits : len < 2^w) (finished : s.retired) (normal : s.outside = false)
    (window : Window t.entry.alpha t.beta) (cap : t.entry.alpha ≤ max original t.entry.best)
    (narrow : t.pv = false → t.beta = t.entry.alpha+1)
    (seed : Bound prefixValue original t.beta t.entry.best) (prefixCap : prefixValue ≤ scoreMax)
    (limits : ∀ i ∈ SplitResult.slots start.toNat len, value i ≤ scoreMax)
    (children : ∀ i, start.toNat ≤ i → i < len → ∀ cp a b score, Window a b →
      (cp = false → b = a+1) → done_ i cp a b score → SoundResult (-(value i)) a b score) :
    Result t original value start.toNat len prefixValue s := by
  have post := Run.posts run fits window cap narrow children
  have steps : SplitReturn.ReportSteps t.pv t.beta t.entry value (reports s) := by
    intro earlier r later part
    obtain ⟨pre,rest,all,first,more⟩ := List.map_eq_append_iff.mp part
    obtain ⟨p,tail,rfl,rfl,_⟩ := List.map_eq_cons_iff.mp more
    obtain ⟨used,bound,_⟩ := post pre p tail all
    exact ⟨first ▸ used,bound⟩
  have valid : ∀ r ∈ reports s, r.index ∈ SplitResult.slots start.toNat len := by
    intro r member
    obtain ⟨p,present,rfl⟩ := List.mem_map.mp member
    exact SplitResult.mem_slots.mpr (run.tickets.valid fits present)
  have merged := SplitReturn.mergeRemaining t.pv original t.beta t.entry value (reports s) [] (reports s)
    window cap steps rfl
  refine ⟨by simpa [reports,List.map_map,Tickets.State.indices,Function.comp_def] using run.tickets.unique fits,
    valid,steps,merged.bound_of_coverage seed limits prefixCap valid ?_⟩
  rcases run.tickets.outcome fits finished with stopped | ⟨p,present,cut⟩ | covered
  · simp [normal] at stopped
  · obtain ⟨earlier,later,part⟩ := List.mem_iff_append.mp present
    have witness := SplitWindow.cutoffWitnessContract_proven t.pv original t.beta t.entry
      ((earlier.map Post.report).map Report.score) ((later.map Post.report).map Report.score) p.report.score
      window cap narrow ((post earlier p later part).2.2.symm.trans cut)
    have endState : t.beta ≤ (t.locked s.reports).best ∨ scoreMax ≤ (t.locked s.reports).best := by
      simpa [part,Task.locked,SplitReturn.stateAfter] using witness
    exact endState.imp_right Or.inl
  · right; right
    intro i member
    obtain ⟨p,hp,same⟩ := List.mem_map.mp (covered.mem_iff.mpr member)
    exact ⟨p.report,List.mem_map_of_mem hp,same⟩

end NeuralReversi.PvsParallel.Interleaving.Schedule

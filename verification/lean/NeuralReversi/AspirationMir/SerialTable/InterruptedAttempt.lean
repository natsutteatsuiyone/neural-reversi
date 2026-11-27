import NeuralReversi.PvsParallel.SerialTable.EndgameInterrupted
import NeuralReversi.PublicationMir.InterruptedExact

namespace NeuralReversi.AspirationMir.SerialTable.InterruptedAttempt

open PvsParallel
open PvsParallel.SerialTable (Layout Memory)
open PvsParallel.SerialTable.EndgameInterrupted (Poll outcomeAt outcomeAt_returns)
open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position minimax)
open TranspositionTable (Selectivity)
open RootMoveStorage (Row)
open SerialStack (State)
open SerialCache (Receipt)
open Repeated (ReadyMemory)

/-- Child polls follow the polled search model. The root poll is the abort check
    after a root call returns and before its aspiration window is widened. -/
structure Stops where
  child : Nat → State → Poll
  root : Nat → State → Nat → Bool

def noStops : Stops := ⟨fun _ _ _ _ _ => false, fun _ _ _ => false⟩

/-- A completed aspiration level supplies its own certificate for exact root
    rows. Callers do not supply child scores, a root certificate or saved history. -/
structure Accepted (p : Pool) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (b : Position) (level : Selectivity) (last : Receipt (Memory l)) : Prop
    extends SelectiveLadder.Accepted mc b (ReadyMemory p mc fc) last where
  exactRows : level.val = 5 → PublicationMir.ExactScore.RowScores (minimax mc fc b) last.answer.rows

/-- One endgame level, including failed aspiration attempts and polls inside
    every root trial. Stops retain the actual updated TT/cache memory but cannot
    produce an accepted receipt. -/
def run (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (level : Selectivity) (stops : Stops) :
    Nat → State → Memory l → Option (Option (Receipt (Memory l)) × Memory l)
  | 0, _, _ => none
  | fuel+1, s, before => do
    let op := ops.order level.val s
    let (k,answer,after) ← outcomeAt 0 p c b (op.base (fc := fc)) op.root
      level (ops.screen level.val s) (stops.child level.val s) s before
    match answer with
    | none => return (none,after)
    | some ans =>
      if stops.root level.val s k then return (none,after)
      let v : Receipt (Memory l) := ⟨⟨s,k,ans⟩,before,after⟩
      let (regs,again) ← applyWiden s.regs v.call.score
      if again then run p c b ops fc level stops fuel (SerialStack.next v.toVisit regs) after
      else return (some v,after)

private theorem completed_facts (p : Pool) (c : SerialScout.Config) (b : Position)
    (base : SerialExecution.Inputs mc fc) (root : SerialRecursion.RootInput)
    (level : Selectivity) (screen : PvsParallel.SerialTable.WeakStack.Screen)
    (s : State) (before : Memory l) (k : Nat) (ans : SmallRoot.Answer) (after : Memory l)
    (addr : c.address = p.address) (initial : ReadyMemory p mc fc before)
    (cover : Root.Covers mc b (s.rows.map Row.square)) (positive : 0 < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt)
    (whole : (if level.val = 5 then PvsParallel.SerialTable.MultiExact.trial 0 p c b base root s before
      else PvsParallel.SerialTable.MultiWeak.trial 0 p c b base root level screen s before) = some (k,ans,after)) :
    (-32767 < ans.value ∧ ans.value < 32767) ∧ ans.rows.map Row.square = s.rows.map Row.square ∧
      (level.val = 5 → SerialStack.Certified mc fc b s ans) := by
  by_cases full : level.val = 5
  · rw [ite_eq_left full] at whole
    obtain ⟨j,answer,next,ran,cert,_⟩ :=
      PvsParallel.SerialTable.MultiExact.trial_returns 0 p c b base root s before addr initial cover positive window
    rw [whole] at ran
    cases ran
    obtain ⟨snap,best,writes,attempt⟩ := cert
    exact ⟨⟨attempt.1.lower,attempt.1.upper⟩,(writes.scores (Root.roots_nodup cover)).1,
      fun _ => ⟨snap,best,writes,attempt⟩⟩
  · rw [ite_eq_right full] at whole
    have weak : level.val < 5 := by have := level.isLt; omega
    obtain ⟨j,answer,next,ran,_,finite,names,_⟩ :=
      PvsParallel.SerialTable.MultiWeak.trial_returns 0 p c addr b base root level weak screen s before
        initial cover positive window
    rw [whole] at ran
    cases ran
    exact ⟨finite,names,fun equal => (full equal).elim⟩

/-- Every ready level returns in the existing twelve-attempt bound, for arbitrary
    child and root stop observations. Completed receipts establish their exact
    row invariant; interruptions retain valid memory and return no receipt. -/
theorem returns (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (level : Selectivity) (stops : Stops)
    (fuel : Nat) (s : State) (before : Memory l) (addr : c.address = p.address)
    (initial : ReadyMemory p mc fc before)
    (cover : Root.Covers mc b (s.rows.map Row.square)) (positive : 0 < s.rows.length)
    (ready : Ready n s.regs) (budget : n + fuel = 12) :
    ∃ answer after, run p c b ops fc level stops fuel s before = some (answer,after) ∧
      ReadyMemory p mc fc after ∧ ∀ v, answer = some v → v.after = after ∧ Accepted p mc fc b level v := by
  induction fuel generalizing n s before with
  | zero => have := ready.limit; omega
  | succ fuel ih =>
    let op := ops.order level.val s
    obtain ⟨k,value,after,ran,safe,whole⟩ := outcomeAt_returns 0 p c b
      (op.base (fc := fc)) op.root level (ops.screen level.val s) (stops.child level.val s)
      s before addr initial cover positive ready.window
    cases value with
    | none =>
      exact ⟨none,after,by simp [run,op] at ran ⊢; rw [ran]; rfl,safe,by simp⟩
    | some ans =>
      cases stopped : stops.root level.val s k with
      | true =>
        refine ⟨none,after,?_,safe,by simp⟩
        simp [run,op] at ran ⊢
        rw [ran]
        simp [stopped]
      | false =>
        obtain ⟨finite,names,exactProof⟩ := completed_facts p c b (op.base (fc := fc)) op.root level
          (ops.screen level.val s) s before k ans after addr initial cover positive ready.window (whole ans rfl)
        let v : Receipt (Memory l) := ⟨⟨s,k,ans⟩,before,after⟩
        have signed : (BitVec.ofInt 32 ans.value).toInt = ans.value :=
          BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)
        have finiteWord : -32767 < (BitVec.ofInt 32 ans.value).toInt ∧
            (BitVec.ofInt 32 ans.value).toInt < 32767 := by simpa only [signed] using finite
        have covered : Root.Covers mc b (ans.rows.map Row.square) := names ▸ cover
        have size : ans.rows.length = s.rows.length := by simpa using congrArg List.length names
        have pos : 0 < ans.rows.length := size ▸ positive
        rcases ready_widen ready _ finiteWord with stop | ⟨regs,wide,nextReady⟩
        · have accepted : Accepted p mc fc b level v := {
            toAccepted := ⟨safe,covered,pos,ready.window,widen_false stop,signed⟩
            exactRows := fun full => PublicationMir.InterruptedExact.certified_rows
              (exactProof full) cover positive (by simpa only [signed] using widen_false stop) }
          refine ⟨some v,after,?_,safe,?_⟩
          · simp [run,op] at ran ⊢
            rw [ran]
            simp [stopped,SerialStack.Visit.call,stop,v]
          · intro chosen same
            cases same
            exact ⟨rfl,accepted⟩
        · obtain ⟨answer,last,rest,good,accepted⟩ :=
            ih (SerialStack.next v.toVisit regs) after safe covered pos nextReady (by omega)
          refine ⟨answer,last,?_,good,accepted⟩
          simp [run,op] at ran ⊢
          rw [ran]
          simpa only [stopped,Bool.false_eq_true,↓reduceIte,SerialStack.Visit.call,
            wide,bind,Option.bind] using rest

/-- The corresponding ordinary root trial at this level, with the same move
    ordering, screening, worker caches and physical transposition table. -/
def normalTrial (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (level : Selectivity) : SerialCache.Trial (Memory l) := fun s before =>
  let op := ops.order level.val s
  if level.val = 5 then PvsParallel.SerialTable.MultiExact.trial 0 p c b (op.base (fc := fc)) op.root s before
  else PvsParallel.SerialTable.MultiWeak.trial 0 p c b (op.base (fc := fc)) op.root level
    (ops.screen level.val s) s before

/-- Removing every stop observation preserves the accepted ordinary aspiration
    receipt, its root rows and its final physical memory. -/
theorem run_uninterrupted (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (level : Selectivity) (fuel : Nat) (s : State) (before : Memory l) :
    run p c b ops fc level noStops fuel s before =
      (SerialCache.run (normalTrial p c b ops fc level) fuel s before).map
        (fun out => (some out.2,out.2.after)) := by
  induction fuel generalizing s before with
  | zero => rfl
  | succ fuel ih =>
    let op := ops.order level.val s
    have step := PvsParallel.SerialTable.EndgameInterrupted.outcomeAt_uninterrupted 0 p c b
      (op.base (fc := fc)) op.root level (ops.screen level.val s) s before
    change outcomeAt 0 p c b (op.base (fc := fc)) op.root level (ops.screen level.val s)
      (fun _ _ _ => false) s before = (normalTrial p c b ops fc level s before).map
        (fun (k,ans,after) => (k,some ans,after)) at step
    cases baseRun : normalTrial p c b ops fc level s before with
    | none => simp [run,SerialCache.run,noStops,step,op,baseRun]
    | some out =>
      rcases out with ⟨k,ans,after⟩
      let v : Receipt (Memory l) := ⟨⟨s,k,ans⟩,before,after⟩
      cases wide : applyWiden s.regs (BitVec.ofInt 32 ans.value) with
      | none => simp [run,SerialCache.run,noStops,step,op,baseRun,SerialStack.Visit.call,wide]
      | some pair =>
        rcases pair with ⟨regs,again⟩
        cases again with
        | false => simp [run,SerialCache.run,noStops,step,op,baseRun,SerialStack.Visit.call,wide]
        | true =>
          simp only [op] at step
          simp only [run,SerialCache.run,noStops,step,baseRun,Option.map_some,bind,Option.bind,
            Bool.false_eq_true,↓reduceIte,SerialStack.Visit.call,wide]
          change run p c b ops fc level noStops fuel (SerialStack.next v.toVisit regs) after = _
          rw [ih]
          cases rest : SerialCache.run (normalTrial p c b ops fc level) fuel (SerialStack.next v.toVisit regs) after
          · rfl
          · rfl

/-- The totality theorem is non-vacuous: with no stops, a ready level always
    produces a completed receipt rather than an interruption marker. -/
theorem uninterrupted_returns (p : Pool) (c : SerialScout.Config) (b : Position)
    (ops : FullLadder.Inputs mc) (fc : Cutoffs.FlipConfig) (level : Selectivity)
    (fuel : Nat) (s : State) (before : Memory l) (addr : c.address = p.address)
    (initial : ReadyMemory p mc fc before)
    (cover : Root.Covers mc b (s.rows.map Row.square)) (positive : 0 < s.rows.length)
    (ready : Ready n s.regs) (budget : n + fuel = 12) :
    ∃ last, run p c b ops fc level noStops fuel s before = some (some last,last.after) ∧
      Accepted p mc fc b level last := by
  obtain ⟨answer,after,ran,_,accepted⟩ := returns p c b ops fc level noStops fuel s before
    addr initial cover positive ready budget
  have normal := ran
  rw [run_uninterrupted] at normal
  obtain ⟨⟨prior,last⟩,_,same⟩ := Option.map_eq_some_iff.mp normal
  obtain ⟨rfl,rfl⟩ := Prod.mk.inj same
  exact ⟨last,ran,(accepted last rfl).2⟩

end NeuralReversi.AspirationMir.SerialTable.InterruptedAttempt

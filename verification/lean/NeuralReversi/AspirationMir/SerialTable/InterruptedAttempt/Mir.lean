module
public import NeuralReversi.AspirationMir.Polled
public import NeuralReversi.AspirationMir.SerialTable.InterruptedAttempt

@[expose] public section

namespace NeuralReversi.AspirationMir.SerialTable.InterruptedAttempt.Mir

open PvsParallel
open PvsParallel.SerialTable (Layout Memory)
open EndgameSearch.CacheHistory (Pool)
open PvsBoardTree (Position)
open RootMoveStorage (Row)
open TranspositionTable (Selectivity)
open SerialStack (State)
open SerialCache (Receipt)
open Repeated (ReadyMemory)

abbrev Trial (M : Type) := State → M → Option (Nat × Option SmallRoot.Answer × M)

/-- A root invocation and the two distinct ways its result can be rejected.
    `none` is the child-stop marker; `rootStop` is the supplied value of the
    subsequent root-side read. The runner shortcuts that read on a child stop;
    retaining its supplied value makes the coherence obligation explicit. -/
structure Observation (M : Type) where
  state : State
  closing : Nat
  answer : Option SmallRoot.Answer
  before : M
  after : M
  rootStop : Bool

variable {M : Type}

def Observation.raw (o : Observation M) : Word :=
  BitVec.ofInt 32 ((o.answer.map SmallRoot.Answer.value).getD 0)

def Observation.call (o : Observation M) : Call :=
  ⟨o.state.index,o.state.regs.alpha,o.state.regs.beta,o.raw⟩

/-- The polled interpreter retains an already observed child stop. Arbitrary
    `Stops` do not by themselves prove that the later native root read is true. -/
def Observation.effectiveAbort (o : Observation M) : Bool :=
  o.answer.isNone || o.rootStop

def Observation.result (o : Observation M) : Option (Receipt M) :=
  if o.effectiveAbort then none else
    o.answer.map (fun answer => ⟨⟨o.state,o.closing,answer⟩,o.before,o.after⟩)

theorem Observation.result_none (o : Observation M) :
    o.result = none ↔ o.effectiveAbort = true := by
  obtain ⟨state,closing,answer,before,after,rootStop⟩ := o
  cases answer <;> cases rootStop <;> simp [Observation.result,Observation.effectiveAbort]

theorem Observation.result_call (o : Observation M) (v : Receipt M) (found : o.result = some v) :
    o.call = v.call := by
  obtain ⟨state,closing,answer,before,after,rootStop⟩ := o
  cases answer <;> cases rootStop <;> simp [Observation.result,Observation.effectiveAbort] at found
  subst v
  rfl

def observe (root : State → Nat → Bool) (s : State) (before : M) (k : Nat)
    (answer : Option SmallRoot.Answer) (after : M) : Observation M :=
  ⟨s,k,answer,before,after,root s k⟩

/-- Every constructor records an actual root outcome. Retry constructors retain
    its rows and memory; a stopped outcome never supplies an accepted receipt. -/
inductive Path (trial : Trial M) (root : State → Nat → Bool) :
    State → M → List (Observation M) → Observation M → Prop
  | child (returned : trial s before = some (k,none,after)) :
      Path trial root s before [] (observe root s before k none after)
  | stopped (returned : trial s before = some (k,some ans,after)) (stop : root s k = true) :
      Path trial root s before [] (observe root s before k (some ans) after)
  | accept (returned : trial s before = some (k,some ans,after)) (live : root s k = false)
      (inside : applyWiden s.regs (BitVec.ofInt 32 ans.value) = some (s.regs,false)) :
      Path trial root s before [] (observe root s before k (some ans) after)
  | retry (returned : trial s before = some (k,some ans,after)) (live : root s k = false)
      (wide : applyWiden s.regs (BitVec.ofInt 32 ans.value) = some (regs,true))
      (rest : Path trial root (SerialStack.next ⟨s,k,ans⟩ regs) after prior last) :
      Path trial root s before (observe root s before k (some ans) after :: prior) last

variable {trial : Trial M}

theorem Path.indices (path : Path trial root s before prior last) :
    ((prior ++ [last]).map (fun o => o.state.index)) = List.range' s.index (prior.length+1) := by
  induction path with
  | child | stopped | accept => simp [observe,List.range'_succ]
  | retry _ _ _ _ ih =>
    simp only [List.cons_append,List.map_cons,List.length_cons,List.range'_succ]
    exact congrArg (_ :: ·) ih

theorem Path.carried (path : Path trial root s before prior last) :
    (prior ++ [last]).map Observation.before = before :: prior.map Observation.after := by
  induction path with
  | child | stopped | accept => rfl
  | retry _ _ _ _ ih => exact congrArg (_ :: ·) ih

theorem Path.last_index (path : Path trial root s before prior last) :
    last.state.index = s.index + prior.length := by
  induction path with
  | child | stopped | accept => simp [observe]
  | retry _ _ _ _ ih => simpa [SerialStack.next,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using ih

theorem Path.root_observed (path : Path trial root s before prior last) :
    ∀ o ∈ prior ++ [last], o.rootStop = root o.state o.closing := by
  induction path with
  | child | stopped | accept =>
    intro o member
    obtain rfl := List.mem_singleton.mp member
    rfl
  | retry _ _ _ _ ih =>
    intro o member
    rcases List.mem_cons.mp member with rfl | later
    · rfl
    · exact ih o later

theorem Path.invocations (path : Path trial root s before prior last) :
    ∀ o ∈ prior ++ [last], trial o.state o.before = some (o.closing,o.answer,o.after) := by
  induction path with
  | child returned | stopped returned _ | accept returned _ _ =>
    intro o member
    obtain rfl := List.mem_singleton.mp member
    exact returned
  | retry returned _ _ _ ih =>
    intro o member
    rcases List.mem_cons.mp member with rfl | later
    · exact returned
    · exact ih o later

def key (o : Observation M) := (o.state.index,o.state.regs.alpha,o.state.regs.beta)

/-- Both oracles are built from the root calls actually consumed by `run`.
    Their defaults are unreachable on the corresponding MIR execution. -/
def queryFor (all : List (Observation M)) : Query := fun i a z =>
  ((all.map (fun o => (key o,o.raw))).lookup (i,a,z)).getD 0

def abortFor (all : List (Observation M)) : Nat → Bool := fun i =>
  ((all.map (fun o => (o.state.index,o.effectiveAbort))).lookup i).getD false

def rootAbortFor (all : List (Observation M)) : Nat → Bool := fun i =>
  ((all.map (fun o => (o.state.index,o.rootStop))).lookup i).getD false

theorem Path.observed (path : Path trial root s before prior last)
    (o : Observation M) (member : o ∈ prior ++ [last]) :
    o.call.Observed (queryFor (prior ++ [last])) := by
  let all := prior ++ [last]
  have distinct : (all.map (fun o => o.state.index)).Nodup := path.indices ▸ List.nodup_range'
  have projected : ((all.map key).map Prod.fst).Nodup := by
    simpa only [List.map_map,Function.comp_def,key] using distinct
  have unique : (all.map key).Nodup :=
    projected.of_map Prod.fst (fun _ _ ne eq => ne (congrArg Prod.fst eq))
  have found := SerialCalls.Program.lookup_mem (calls := all.map (fun o => (key o,o.raw)))
    (by simpa only [List.map_map,Function.comp_def] using unique)
    (List.mem_map_of_mem (f := fun o => (key o,o.raw)) member)
  change ((all.map _).lookup (key o)).getD 0 = o.raw
  rw [found]
  rfl

private theorem lookup_observation (path : Path trial root s before prior last)
    (f : Observation M → α) (o : Observation M) (member : o ∈ prior ++ [last]) :
    ((prior ++ [last]).map (fun v => (v.state.index,f v))).lookup o.state.index = some (f o) := by
  have distinct : ((prior ++ [last]).map (fun o => o.state.index)).Nodup :=
    path.indices ▸ List.nodup_range'
  exact SerialCalls.Program.lookup_mem
    (by simpa only [List.map_map,Function.comp_def] using distinct)
    (List.mem_map_of_mem (f := fun v => (v.state.index,f v)) member)

theorem Path.abort_observed (path : Path trial root s before prior last)
    (o : Observation M) (member : o ∈ prior ++ [last]) :
    abortFor (prior ++ [last]) o.state.index = o.effectiveAbort := by
  unfold abortFor
  rw [lookup_observation path _ o member]
  rfl

theorem Path.root_abort_observed (path : Path trial root s before prior last)
    (o : Observation M) (member : o ∈ prior ++ [last]) :
    rootAbortFor (prior ++ [last]) o.state.index = root o.state o.closing := by
  unfold rootAbortFor
  rw [lookup_observation path _ o member]
  exact path.root_observed o member

private theorem current_eq (o : Observation M) (seen : o.call.Observed query)
    (idx : m.issued = o.state.index) (regs : m.regs = o.state.regs) :
    currentCall query m = o.call := by
  have value : query o.state.index o.state.regs.alpha o.state.regs.beta = o.raw := seen
  simp [currentCall,Observation.call,idx,regs,value]

private theorem steps (path : Path trial root s before prior last) (query : Query) (abort : Nat → Bool)
    (seen : ∀ o ∈ prior ++ [last], o.call.Observed query)
    (flags : ∀ o ∈ prior ++ [last], abort o.state.index = o.effectiveAbort)
    (m : Machine) (pc : m.pc = 2) (idx : m.issued = s.index) (regs : m.regs = s.regs) :
    Returns query abort m last.call.score last.call := by
  induction path generalizing m with
  | @child s before k after found =>
    let o := observe root s before k none after
    have current := current_eq o (seen o List.mem_cons_self) idx regs
    have stop : abort m.issued = true := by
      rw [idx]
      exact (flags o List.mem_cons_self).trans (by simp [o,observe,Observation.effectiveAbort])
    change Returns query abort m o.call.score o.call
    rw [← current]
    exact Polled.aborted pc stop
  | @stopped s before k ans after found stopped =>
    let o := observe root s before k (some ans) after
    have current := current_eq o (seen o List.mem_cons_self) idx regs
    have stop : abort m.issued = true := by
      rw [idx]
      exact (flags o List.mem_cons_self).trans (by simp [o,observe,Observation.effectiveAbort,stopped])
    change Returns query abort m o.call.score o.call
    rw [← current]
    exact Polled.aborted pc stop
  | @accept s before k ans after found live inside =>
    let o := observe root s before k (some ans) after
    have current := current_eq o (seen o List.mem_cons_self) idx regs
    have active : abort m.issued = false := by
      rw [idx]
      exact (flags o List.mem_cons_self).trans (by simp [o,observe,Observation.effectiveAbort,live])
    change Returns query abort m o.call.score o.call
    rw [← current]
    exact Polled.accept pc active (by rw [current,regs]; exact inside)
  | @retry s before k ans after regs' prior last found live wide rest ih =>
    let o := observe root s before k (some ans) after
    have current := current_eq o (seen o List.mem_cons_self) idx regs
    have active : abort m.issued = false := by
      rw [idx]
      exact (flags o List.mem_cons_self).trans (by simp [o,observe,Observation.effectiveAbort,live])
    apply Polled.retry pc active (show applyWiden m.regs (currentCall query m).score = some (regs',true) by
      rw [current,regs]; exact wide)
    exact ih (fun o member => seen o (.tail _ member)) (fun o member => flags o (.tail _ member))
      (nextTrial query m regs') rfl (congrArg (·+1) idx) rfl

private theorem Path.returns_for (m : Machine) (pc : m.pc = 0)
    (path : Path trial root (SerialStack.first m rows) before prior last)
    (query : Query) (abort : Nat → Bool)
    (seen : ∀ o ∈ prior ++ [last], o.call.Observed query)
    (flags : ∀ o ∈ prior ++ [last], abort o.state.index = o.effectiveAbort) :
    Returns query abort m last.call.score last.call := by
  cases m with
  | mk pc' regs idx old flag out =>
    change pc' = 0 at pc
    subst pc'
    exact .next rfl (.next rfl (steps path query abort seen flags
      ⟨2,{regs with delta := BitVec.ofInt 32 initialDelta},idx,old,flag,out⟩ rfl rfl rfl))

theorem Path.returns (m : Machine) (pc : m.pc = 0)
    (path : Path trial root (SerialStack.first m rows) before prior last) :
    Returns (queryFor (prior ++ [last])) (abortFor (prior ++ [last])) m last.call.score last.call :=
  path.returns_for m pc _ _ path.observed path.abort_observed

/-- This is deliberately restricted to reached observations. It is the remaining
    link needed to identify the interpreter's latched stop with the native root
    poll, rather than silently assuming those independent inputs coincide. -/
def Coherent (all : List (Observation M)) : Prop :=
  ∀ o ∈ all, o.answer = none → o.rootStop = true

theorem Coherent.effective_eq_root (coherent : Coherent (M := M) all) (o : Observation M) (member : o ∈ all) :
    o.effectiveAbort = o.rootStop := by
  cases answer : o.answer with
  | none => simp [Observation.effectiveAbort,answer,coherent o member answer]
  | some ans => simp [Observation.effectiveAbort,answer]

/-- A source-level stop-retention proof is needed only for the invocations
    actually reached by this path, with their actual input and output memory. -/
theorem Path.coherent_of (path : Path trial root s before prior last)
    (retained : ∀ o ∈ prior ++ [last], trial o.state o.before = some (o.closing,none,o.after) →
      root o.state o.closing = true) : Coherent (prior ++ [last]) := by
  intro o member stopped
  rw [path.root_observed o member]
  apply retained o member
  simpa only [stopped] using path.invocations o member

/-- With the explicit reached-state coherence premise, exactly the same MIR
    execution uses the supplied root polls alone. -/
theorem Path.returns_root (m : Machine) (pc : m.pc = 0)
    (path : Path trial root (SerialStack.first m rows) before prior last)
    (coherent : Coherent (prior ++ [last])) :
    Returns (queryFor (prior ++ [last])) (rootAbortFor (prior ++ [last])) m last.call.score last.call := by
  apply path.returns_for m pc _ _ path.observed
  intro o member
  unfold rootAbortFor
  rw [lookup_observation path _ o member]
  exact (coherent.effective_eq_root o member).symm

/-- The exact same polled root invocation used by `InterruptedAttempt.run`. -/
def rootTrial (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (level : Selectivity) (stops : Stops) : Trial (Memory l) := fun s before =>
  let op := ops.order level.val s
  PvsParallel.SerialTable.EndgameInterrupted.outcomeAt 0 p c b (op.base (fc := fc)) op.root
    level (ops.screen level.val s) (stops.child level.val s) s before

/-- Recover the consumed call sequence from the executable runner, including
    its final rejected invocation. This direction takes no supplied trace. -/
theorem path_of_run (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (level : Selectivity) (stops : Stops)
    (fuel : Nat) (s : State) (before : Memory l) (answer : Option (Receipt (Memory l))) (after : Memory l)
    (ran : InterruptedAttempt.run p c b ops fc level stops fuel s before = some (answer,after)) :
    ∃ prior last, Path (rootTrial p c b ops fc level stops) (stops.root level.val) s before prior last ∧
      last.result = answer ∧ last.after = after ∧ prior.length < fuel := by
  induction fuel generalizing s before answer after with
  | zero => simp [InterruptedAttempt.run] at ran
  | succ fuel ih =>
    change (do
      let (k,value,next) ← rootTrial p c b ops fc level stops s before
      match value with
      | none => return (none,next)
      | some ans =>
        if stops.root level.val s k then return (none,next)
        let v : Receipt (Memory l) := ⟨⟨s,k,ans⟩,before,next⟩
        let (regs,again) ← applyWiden s.regs v.call.score
        if again then InterruptedAttempt.run p c b ops fc level stops fuel (SerialStack.next v.toVisit regs) next
        else return (some v,next)) = some (answer,after) at ran
    cases called : rootTrial p c b ops fc level stops s before with
    | none => simp [called] at ran
    | some out =>
      obtain ⟨k,value,next⟩ := out
      simp only [called,bind,Option.bind] at ran
      cases value with
      | none =>
        obtain ⟨rfl,rfl⟩ := Prod.mk.inj (Option.some.inj ran)
        exact ⟨[],observe (stops.root level.val) s before k none next,.child called,rfl,rfl,Nat.zero_lt_succ _⟩
      | some ans =>
        let v : Receipt (Memory l) := ⟨⟨s,k,ans⟩,before,next⟩
        cases stopped : stops.root level.val s k with
        | true =>
          simp only [stopped,↓reduceIte] at ran
          obtain ⟨rfl,rfl⟩ := Prod.mk.inj (Option.some.inj ran)
          refine ⟨[],observe (stops.root level.val) s before k (some ans) next,
            .stopped called stopped,?_,rfl,Nat.zero_lt_succ _⟩
          simp [Observation.result,Observation.effectiveAbort,observe,stopped]
        | false =>
          simp only [stopped,Bool.false_eq_true,↓reduceIte] at ran
          change ((applyWiden s.regs (BitVec.ofInt 32 ans.value)).bind fun (regs,again) =>
            if again then InterruptedAttempt.run p c b ops fc level stops fuel (SerialStack.next v.toVisit regs) next
            else some (some v,next)) = some (answer,after) at ran
          cases wide : applyWiden s.regs (BitVec.ofInt 32 ans.value) with
          | none => simp [wide] at ran
          | some pair =>
            obtain ⟨regs,again⟩ := pair
            simp only [wide,Option.bind_some] at ran
            cases again with
            | false =>
              simp only [Bool.false_eq_true,↓reduceIte] at ran
              obtain ⟨rfl,rfl⟩ := Prod.mk.inj (Option.some.inj ran)
              have inside := widen_false wide
              refine ⟨[],observe (stops.root level.val) s before k (some ans) next,
                .accept called stopped (widen_inside _ _ inside.1 inside.2),?_,rfl,Nat.zero_lt_succ _⟩
              simp [Observation.result,Observation.effectiveAbort,observe,stopped,v]
            | true =>
              simp only [↓reduceIte] at ran
              obtain ⟨prior,last,path,chosen,kept,bound⟩ :=
                ih (SerialStack.next v.toVisit regs) next answer after ran
              exact ⟨observe (stops.root level.val) s before k (some ans) next :: prior,last,
                .retry called stopped wide path,chosen,kept,Nat.succ_lt_succ bound⟩

/-- A non-vacuous execution witness combines root memory and numerical safety
    with the actual generated MIR path. The call list is constructed by the
    theorem below, rather than supplied as an oracle premise. -/
structure Witness (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (level : Selectivity) (stops : Stops)
    (machine : Machine) (rows : List Row) (before : Memory l) where
  prior : List (Observation (Memory l))
  last : Observation (Memory l)
  entry : machine.pc = 0
  ran : InterruptedAttempt.run p c b ops fc level stops 12 (SerialStack.first machine rows) before =
    some (last.result,last.after)
  path : Path (rootTrial p c b ops fc level stops) (stops.root level.val)
    (SerialStack.first machine rows) before prior last
  bound : prior.length < 12
  returned : Returns (queryFor (prior ++ [last])) (abortFor (prior ++ [last])) machine last.call.score last.call
  safe : ReadyMemory p mc fc last.after
  accepted : ∀ v, last.result = some v → v.after = last.after ∧ Accepted p mc fc b level v

theorem Witness.classified (w : Witness p c b ops fc level stops machine rows before) :
    Polled.Classified (queryFor (w.prior ++ [w.last])) (abortFor (w.prior ++ [w.last]))
      w.last.call.score w.last.call :=
  Polled.returned_classified w.returned w.entry

theorem Witness.accepted_call (w : Witness p c b ops fc level stops machine rows before)
    (v : Receipt _) (found : w.last.result = some v) : w.last.call = v.call :=
  w.last.result_call v found

theorem Witness.aborted_iff (w : Witness p c b ops fc level stops machine rows before) :
    w.last.result = none ↔ abortFor (w.prior ++ [w.last]) w.last.state.index = true := by
  rw [w.path.abort_observed w.last (List.mem_append_right _ List.mem_cons_self)]
  exact w.last.result_none

theorem Witness.calls_used (w : Witness p c b ops fc level stops machine rows before) :
    w.last.state.index = machine.issued + w.prior.length ∧ w.last.state.index ≤ machine.issued+11 := by
  have last := w.path.last_index
  change w.last.state.index = machine.issued + w.prior.length at last
  exact ⟨last,by have := w.bound; omega⟩

theorem Witness.returned_root (w : Witness p c b ops fc level stops machine rows before)
    (coherent : Coherent (w.prior ++ [w.last])) :
    Returns (queryFor (w.prior ++ [w.last])) (rootAbortFor (w.prior ++ [w.last]))
      machine w.last.call.score w.last.call :=
  w.path.returns_root machine w.entry coherent

/-- The existing totality theorem now supplies its own checked MIR trace for
    arbitrary child/root stops, including interruption before the first receipt. -/
theorem returns (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (fc : Cutoffs.FlipConfig) (level : Selectivity) (stops : Stops)
    (machine : Machine) (rows : List Row) (before : Memory l) (addr : c.address = p.address)
    (initial : ReadyMemory p mc fc before) (cover : Root.Covers mc b (rows.map Row.square))
    (positive : 0 < rows.length) (entry : machine.pc = 0)
    (window : Aspiration.Window machine.regs.alpha.toInt machine.regs.beta.toInt) :
    Nonempty (Witness p c b ops fc level stops machine rows before) := by
  obtain ⟨answer,after,ran,safe,accepted⟩ := InterruptedAttempt.returns p c b ops fc level stops 12
    (SerialStack.first machine rows) before addr initial cover positive (first_ready _ window) rfl
  obtain ⟨prior,last,path,chosen,kept,bound⟩ :=
    path_of_run p c b ops fc level stops 12 (SerialStack.first machine rows) before answer after ran
  refine ⟨⟨prior,last,entry,?_,path,bound,path.returns machine entry,?_,?_⟩⟩
  · rw [chosen,kept]
    exact ran
  · rw [kept]
    exact safe
  · intro v found
    rw [chosen] at found
    rw [kept]
    exact accepted v found

def Contract : Prop :=
  ∀ (l : Layout) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (p : Pool) (c : SerialScout.Config) (b : Position) (ops : FullLadder.Inputs mc)
    (level : Selectivity) (stops : Stops) (machine : Machine) (rows : List Row) (before : Memory l),
    c.address = p.address → ReadyMemory p mc fc before → Root.Covers mc b (rows.map Row.square) →
    0 < rows.length → machine.pc = 0 →
    Aspiration.Window machine.regs.alpha.toInt machine.regs.beta.toInt →
    Nonempty (Witness p c b ops fc level stops machine rows before)

theorem contract_proven : Contract := by
  intro l mc fc p c b ops level stops machine rows before addr initial cover positive entry window
  exact returns p c b ops fc level stops machine rows before addr initial cover positive entry window

end NeuralReversi.AspirationMir.SerialTable.InterruptedAttempt.Mir

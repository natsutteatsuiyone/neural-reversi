import NeuralReversi.PvsParallel.SerialTable.MultiExact
import NeuralReversi.PvsParallel.SerialTable.MultiWeak
import NeuralReversi.PvsParallel.SerialTable.MidgamePoll

namespace NeuralReversi.PvsParallel.SerialTable.EndgameInterrupted

open AspirationMir.SerialTable.Repeated (ReadyMemory)
open EndgameSearch.CacheHistory (Pool)
open SerialExecution (Frame Inputs push)
open SerialCalls (Program Request)
open SerialTermination (Entry Allowed)
open TranspositionTable (Selectivity)
open PvsBoardTree (Position)
open AspirationMir.SerialStack (State)
open AspirationMir.SmallRoot (Answer)
open RootMoveStorage (Row)

abbrev Poll := Nat → Frame → Request → Bool
private abbrev Child (l : Layout) := Nat → Frame → Memory l → Option (Stack.Output l (Option Int))

/-- A pass has no post-move poll. A PV scout which raises alpha still launches its wide
    child, even if the scout returned zero while unwinding a stop. -/
private def postMove (ctx : Frame) (r : Request) (v : Int) : Bool :=
  r.depth != ctx.request.depth && (!ctx.request.pv || MidgamePoll.afterMove r.pv r.beta v)

private def interpret (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (poll : Poll) (child : Child l) (ctx : Frame) (clean : Bool)
    (i : Nat) (m : Memory l) : Program Request α → Option (Stack.Output l (Option α))
  | .fail => none
  | .pure v => some (i,if clean then some v else none,m)
  | .call r resume => do
    let f ← push mc fc ctx r
    let (k,value,n) ← child (i+1) f m
    let clean := clean && value.isSome
    let v := value.getD 0
    if postMove ctx r v && (!clean || poll k ctx r) then return (k,none,n)
    interpret mc fc poll child ctx clean k n (resume v)

private theorem interpret_ok {code : Program Request α} {valid : Memory l → Prop}
    (live : Program.Ready allowed code) (poll : Poll) (child : Child l) (whole : Stack.Child l) (ctx : Frame)
    (enters : ∀ r, allowed r → ∀ j m, valid m → ∃ f k v n, push mc fc ctx r = some f ∧
      child j f m = some (k,v,n) ∧ valid n ∧ ∀ w, v = some w → whole j f m = some (k,w,n))
    (i : Nat) (m : Memory l) (initial : valid m) (clean : Bool) :
    ∃ k value n, interpret mc fc poll child ctx clean i m code = some (k,value,n) ∧ valid n ∧
      ∀ out, value = some out → clean = true ∧ Stack.interpret (mc := mc) (fc := fc) whole i ctx m code = some (k,out,n) := by
  induction live generalizing i m clean with
  | pure =>
    refine ⟨_,_,_,rfl,initial,?_⟩
    cases clean <;> simp [Stack.interpret]
  | @call r resume ok _ ih =>
    obtain ⟨f,k,v,n,pushed,sub,post,wholeChild⟩ := enters r ok (i+1) m initial
    let liveNext := clean && v.isSome
    let word := v.getD 0
    cases stop : (postMove ctx r word && (!liveNext || poll k ctx r)) with
    | true => exact ⟨k,none,n,by simp only [interpret,pushed,sub,bind,Option.bind]; rw [show (postMove ctx r (v.getD 0) && (!(clean && v.isSome) || poll k ctx r)) = true from stop]; rfl,post,by simp⟩
    | false =>
      obtain ⟨j,x,next,ran,g,done_⟩ := ih word k n post liveNext
      refine ⟨j,x,next,?_,g,?_⟩
      · simp only [interpret,pushed,sub,bind,Option.bind]
        rw [show (postMove ctx r (v.getD 0) && (!(clean && v.isSome) || poll k ctx r)) = false from stop]
        exact ran
      · intro out returned
        have ⟨both,rest⟩ := done_ out returned
        obtain ⟨prior,childSome⟩ := Bool.and_eq_true_iff.mp both
        obtain ⟨w,hw⟩ := Option.isSome_iff_exists.mp childSome
        have result := wholeChild w hw
        refine ⟨prior,?_⟩
        dsimp only [word] at rest
        rw [hw] at rest
        simpa [Stack.interpret,pushed,result] using rest

private def normal (p : Pool) (c : SerialScout.Config) (base : Inputs mc fc)
    (level : Selectivity) (screen : WeakStack.Screen) (fuel : Nat) : Stack.Child l :=
  if level.val = 5 then Stack.runFuel p c base fuel else WeakStack.runFuel p c base level screen fuel

private def body (m : Memory l) (c : SerialScout.Config) (base : Inputs mc fc)
    (level : Selectivity) (screen : WeakStack.Screen) (i : Nat) (ctx : Frame) :=
  if level.val = 5 then Execution.body c (m.inputs c base) i ctx else WeakStack.body m c base level screen i ctx

private theorem normal_ok (p : Pool) (c : SerialScout.Config) (base : Inputs mc fc) (level : Selectivity)
    (screen : WeakStack.Screen) (fuel i : Nat) (ctx : Frame) (m : Memory l) (addr : c.address = p.address)
    (rank : Board.playRank mc ctx.request.board.val < fuel) (entry : Entry ctx) (ok : ReadyMemory p mc fc m) :
    ∃ k w n, normal p c base level screen fuel i ctx m = some (k,w,n) ∧ ReadyMemory p mc fc n := by
  by_cases full : level.val = 5
  · obtain ⟨k,w,n,ran,good,_⟩ := Stack.fuel_returns p c addr base fuel i ctx m rank entry ok.1.1
    have h := ExactRetention.frame ok.1.2 ok.2 ran good
    exact ⟨k,w,n,by simpa [normal,full] using ran,⟨good,h.1⟩,h.2⟩
  · have weak : level.val < 5 := by have := level.isLt; omega
    obtain ⟨k,w,n,ran,valid,scores,_⟩ := WeakTrial.weak_fuel p c addr base level weak screen fuel i ctx m rank entry ok.1 ok.2
    exact ⟨k,w,n,by simpa [normal,full] using ran,valid,scores⟩

private theorem normal_step (p : Pool) (c : SerialScout.Config) (base : Inputs mc fc) (level : Selectivity)
    (screen : WeakStack.Screen) (fuel i : Nat) (ctx : Frame) (m : Memory l)
    (regular : ¬(ctx.request.pv = false ∧ ctx.request.depth.toNat ≤ 12))
    (ran : Stack.interpret (mc := mc) (fc := fc) (normal p c base level screen fuel) i ctx m
      (body m c base level screen i ctx) = some (k,out,n)) :
    normal p c base level screen (fuel+1) i ctx m =
      some (k,out.score,n.apply (m.tt.image.slot ctx.request.board) out.store level) := by
  by_cases full : level.val = 5
  · have eq : level = Cutoffs.exactSelectivity := Fin.ext full
    subst level
    simp only [normal,body,full,ite_true] at ran ⊢
    simp [Stack.runFuel,regular,ran]
  · simp only [normal,body,full,ite_false] at ran ⊢
    simp [WeakStack.runFuel,regular,ran]

private def runFuel (p : Pool) (c : SerialScout.Config) (base : Inputs mc fc) (level : Selectivity)
    (screen : WeakStack.Screen) (poll : Poll) : Nat → Child l
  | 0, _, _, _ => none
  | n+1, i, ctx, m =>
    if ctx.request.pv = false ∧ ctx.request.depth.toNat ≤ 12 then
      (normal p c base level screen (n+1) i ctx m).map fun (k,w,next) => (k,some w,next)
    else do
      let (k,value,next) ← interpret mc fc poll (runFuel p c base level screen poll n) ctx true i m
        (body m c base level screen i ctx)
      match value with
      | none => return (k,none,next)
      | some out => return (k,some out.score,next.apply (m.tt.image.slot ctx.request.board) out.store level)

private theorem run_ok (p : Pool) (c : SerialScout.Config) (base : Inputs mc fc) (level : Selectivity)
    (screen : WeakStack.Screen) (poll : Poll) (n i : Nat) (ctx : Frame) (m : Memory l) (addr : c.address = p.address)
    (rank : Board.playRank mc ctx.request.board.val < n) (entry : Entry ctx) (ok : ReadyMemory p mc fc m) :
    ∃ k v post, runFuel p c base level screen poll n i ctx m = some (k,v,post) ∧ ReadyMemory p mc fc post ∧
      ∀ w, v = some w → normal p c base level screen n i ctx m = some (k,w,post) := by
  induction n generalizing i ctx m with
  | zero => omega
  | succ n ih =>
    obtain ⟨k,w,next,full,safe⟩ := normal_ok p c base level screen (n+1) i ctx m addr rank entry ok
    by_cases leaf : ctx.request.pv = false ∧ ctx.request.depth.toNat ≤ 12
    · exact ⟨k,some w,next,by simp [runFuel,leaf,full],safe,fun _ h => by cases h; exact full⟩
    have live : Program.Ready (Allowed mc fc ctx) (body m c base level screen i ctx) := by
      unfold body
      split
      · obtain ⟨valid,hints⟩ := m.facts c base ok.1.1
        have h := SerialTermination.frame_ready c _ valid hints i ctx entry
        rw [← Execution.body_score] at h
        exact Stack.ready_of_map h
      · exact WeakStack.body_ready m c base level screen i ctx ok.1 entry leaf
    obtain ⟨j,value,post,ran,g,h⟩ := interpret_ok live poll (runFuel p c base level screen poll n)
      (normal p c base level screen n) ctx (fun r allowed j prior good => by
        obtain ⟨f,pushed,entered,smaller⟩ := SerialTermination.child_entry entry r allowed
        obtain ⟨z,x,after,sub,holds,same⟩ := ih j f prior (by omega) entered good
        exact ⟨f,z,x,after,pushed,sub,holds,same⟩) i m ok true
    cases value with
    | none => exact ⟨j,none,post,by simp [runFuel,leaf,ran],g,by simp⟩
    | some out =>
      have same := normal_step p c base level screen n i ctx m leaf (h out rfl).2
      rw [full] at same
      obtain ⟨rfl,rfl,rfl⟩ := Prod.mk.inj (Option.some.inj same)
      exact ⟨k,some out.score,_,by simp [runFuel,leaf,ran],safe,fun _ eq => by cases eq; exact full⟩

private def completed (out : Stack.Output l α) : Stack.Output l (Option α) := (out.1,some out.2.1,out.2.2)

private theorem interpret_uninterrupted (child : Child l) (whole : Stack.Child l)
    (same : ∀ i f m, child i f m = (whole i f m).map completed) (code : Program Request α)
    (ctx : Frame) (i : Nat) (m : Memory l) :
    interpret mc fc (fun _ _ _ => false) child ctx true i m code =
      (Stack.interpret (mc := mc) (fc := fc) whole i ctx m code).map completed := by
  induction code generalizing i m with
  | fail | pure => rfl
  | call r resume ih =>
    cases entered : push mc fc ctx r with
    | none => simp [interpret,Stack.interpret,entered]
    | some f =>
      cases frame : whole (i+1) f m with
      | none => simp [interpret,Stack.interpret,entered,same,frame]
      | some out =>
        rcases out with ⟨k,v,n⟩
        simpa [interpret,Stack.interpret,entered,same,frame,completed] using ih v k n

private theorem uninterrupted (p : Pool) (c : SerialScout.Config) (base : Inputs mc fc) (level : Selectivity)
    (screen : WeakStack.Screen) (fuel i : Nat) (ctx : Frame) (m : Memory l) :
    runFuel p c base level screen (fun _ _ _ => false) fuel i ctx m =
      (normal p c base level screen fuel i ctx m).map completed := by
  induction fuel generalizing i ctx m with
  | zero => simp [runFuel,normal,Stack.runFuel,WeakStack.runFuel]
  | succ n ih =>
    by_cases leaf : ctx.request.pv = false ∧ ctx.request.depth.toNat ≤ 12
    · rw [runFuel,ite_eq_left leaf]
      rfl
    · let code := body m c base level screen i ctx
      let whole := normal (l := l) p c base level screen n
      have sub := interpret_uninterrupted (mc := mc) (fc := fc)
        (runFuel p c base level screen (fun _ _ _ => false) n) whole ih code ctx i m
      have step : normal p c base level screen (n+1) i ctx m =
          (Stack.interpret (mc := mc) (fc := fc) whole i ctx m code).map
            (fun (k,out,after) => (k,out.score,after.apply (m.tt.image.slot ctx.request.board) out.store level)) := by
        by_cases full : level.val = 5
        · have eq : level = Cutoffs.exactSelectivity := Fin.ext full
          subst level
          simp only [whole,code,normal,body,full,ite_true,Stack.runFuel,leaf,ite_false]
          cases Stack.interpret (mc := mc) (fc := fc) (Stack.runFuel p c base n) i ctx m (Execution.body c (m.inputs c base) i ctx) <;> rfl
        · simp only [whole,code,normal,body,full,ite_false,WeakStack.runFuel,leaf]
          cases Stack.interpret (mc := mc) (fc := fc) (WeakStack.runFuel p c base level screen n) i ctx m (WeakStack.body m c base level screen i ctx) <;> rfl
      rw [runFuel,ite_eq_right leaf,step]
      change (interpret mc fc _ _ ctx true i m code).bind _ = _
      rw [sub]
      cases Stack.interpret (mc := mc) (fc := fc) whole i ctx m code <;> rfl

private def rootCode (idx : Nat) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc) (root : SerialRecursion.RootInput)
    (s : State) (m : Memory l) :=
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  Root.frame mc fc b ctx.request.depth c.paired (base s.opening ctx).memory s.rows idx
    (Trial.ordering m base b root s) s.regs.alpha s.regs.beta

/-- Retain the completion marker and physical root rows of the polled execution.
    A stopped child has no root answer; its last physical memory is still retained. -/
def outcomeAt (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (level : Selectivity) (screen : WeakStack.Screen) (poll : Poll)
    (s : State) (prior : Memory l) : Option (Nat × Option Answer × Memory l) := do
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  let child := fun i f m => runFuel p c base level screen poll (Board.playRank mc f.request.board.val + 1) i f m
  let (k,value,n) ← interpret mc fc poll child ctx true s.opening prior (rootCode idx c b base root s prior)
  match value with
  | none => return (k,none,n)
  | some out => return (k,some out.answer,n.apply (prior.tt.image.slot b) out.proposal level)

/-- The previous memory-only interface is exactly the projection of the richer
    result; clients concerned only with cache and TT preservation remain unchanged. -/
def trialAt (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (level : Selectivity) (screen : WeakStack.Screen) (poll : Poll)
    (s : State) (prior : Memory l) : Option (Memory l) :=
  (outcomeAt idx p c b base root level screen poll s prior).map (fun out => out.2.2)

/-- Arbitrary polls preserve memory validity. Whenever an answer survives, the
    exact same root answer, closing activation and physical memory occur in the
    corresponding uninterrupted executable trial; no no-stop premise is needed. -/
theorem outcomeAt_returns (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (level : Selectivity) (screen : WeakStack.Screen) (poll : Poll)
    (s : State) (prior : Memory l) (addr : c.address = p.address) (ok : ReadyMemory p mc fc prior)
    (cover : PvsParallel.Root.Covers mc b (s.rows.map Row.square)) (pos : idx < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) :
    ∃ k answer after, outcomeAt idx p c b base root level screen poll s prior = some (k,answer,after) ∧
      ReadyMemory p mc fc after ∧ ∀ ans, answer = some ans →
        (if level.val = 5 then MultiExact.trial idx p c b base root s prior
          else MultiWeak.trial idx p c b base root level screen s prior) = some (k,ans,after) := by
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  let code := rootCode idx c b base root s prior
  let child := fun i (f : Frame) (m : Memory l) => runFuel p c base level screen poll (Board.playRank mc f.request.board.val + 1) i f m
  let whole := fun i (f : Frame) (m : Memory l) => normal p c base level screen (Board.playRank mc f.request.board.val + 1) i f m
  have entry : Entry ctx := by
    simp only [ctx,Entry,SerialRootExecution.context,ContextPathMir.start_eq]
    exact ⟨EndgameSearch.emptyState_initial b,rfl,window,by simp⟩
  have ready : Program.Ready (Allowed mc fc ctx) (code.map Root.Exit.answer) := by
    rw [show code.map Root.Exit.answer = _ from Root.erase ..]
    exact SerialRootCalls.frame_ready_at_depth cover pos window
  obtain ⟨k,value,n,ran,safe,done_⟩ := interpret_ok (code := code) (Stack.ready_of_map ready) poll child whole ctx
    (fun r allowed j m good => by
      obtain ⟨f,pushed,admitted,_⟩ := SerialTermination.child_entry entry r allowed
      obtain ⟨last,v,post,run,valid,same⟩ := run_ok p c base level screen poll _ j f m addr
        (Nat.lt_succ_self _) admitted good
      exact ⟨f,last,v,post,pushed,run,valid,same⟩) s.opening prior ok true
  cases value with
  | none =>
    refine ⟨k,none,n,?_,safe,by simp⟩
    change (interpret mc fc poll child ctx true s.opening prior code).bind _ = _
    rw [ran]; rfl
  | some out =>
    have original := (done_ out rfl).2
    let post := n.apply (prior.tt.image.slot b) out.proposal level
    refine ⟨k,some out.answer,post,?_,?_⟩
    · change (interpret mc fc poll child ctx true s.opening prior code).bind _ = _
      rw [ran]; rfl
    · by_cases exactLevel : level.val = 5
      · have eq : level = Cutoffs.exactSelectivity := Fin.ext exactLevel
        have returned : MultiExact.trial idx p c b base root s prior = some (k,out.answer,post) := by
          simp only [whole,normal,exactLevel,ite_true] at original
          unfold MultiExact.trial
          change (Stack.interpret (mc := mc) (fc := fc) (Stack.run p c base) s.opening ctx prior code).bind _ = _
          rw [show Stack.interpret (mc := mc) (fc := fc) (Stack.run p c base) s.opening ctx prior code = _ from original]
          simp [post,eq]
        obtain ⟨j,ans,next,computed,_,good⟩ := MultiExact.trial_returns idx p c b base root s prior addr ok cover pos window
        rw [returned] at computed
        cases computed
        exact ⟨good,fun ans same => by cases same; simpa only [exactLevel,ite_true] using returned⟩
      · have weak : level.val < 5 := by have := level.isLt; omega
        have returned : MultiWeak.trial idx p c b base root level screen s prior = some (k,out.answer,post) := by
          simp only [whole,normal,exactLevel,ite_false] at original
          unfold MultiWeak.trial
          change (Stack.interpret (mc := mc) (fc := fc) (WeakStack.run p c base level screen) s.opening ctx prior code).bind _ = _
          rw [show Stack.interpret (mc := mc) (fc := fc) (WeakStack.run p c base level screen) s.opening ctx prior code = _ from original]
          rfl
        obtain ⟨j,ans,next,computed,good,_⟩ := MultiWeak.trial_returns idx p c addr b base root level weak screen s prior ok cover pos window
        rw [returned] at computed
        cases computed
        exact ⟨good,fun ans same => by cases same; simpa only [exactLevel,ite_false] using returned⟩

/-- A completed polled run supplies its corresponding ordinary execution as a
    conclusion, so downstream publication proofs need not assume a certificate. -/
theorem outcomeAt_completed (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (level : Selectivity) (screen : WeakStack.Screen) (poll : Poll)
    (s : State) (prior : Memory l) (k : Nat) (ans : Answer) (after : Memory l)
    (addr : c.address = p.address) (ok : ReadyMemory p mc fc prior)
    (cover : PvsParallel.Root.Covers mc b (s.rows.map Row.square)) (pos : idx < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt)
    (ran : outcomeAt idx p c b base root level screen poll s prior = some (k,some ans,after)) :
    ReadyMemory p mc fc after ∧
      (if level.val = 5 then MultiExact.trial idx p c b base root s prior
        else MultiWeak.trial idx p c b base root level screen s prior) = some (k,ans,after) := by
  obtain ⟨j,value,post,run,good,whole⟩ :=
    outcomeAt_returns idx p c b base root level screen poll s prior addr ok cover pos window
  rw [ran] at run
  cases run
  exact ⟨good,whole ans rfl⟩

/-- At full selectivity an accepted polled outcome has the existing root write
    and minimax-bound certificate, obtained from its execution rather than supplied
    by the caller. This statement applies to every root suffix index. -/
theorem outcomeAt_exact (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (level : Selectivity) (screen : WeakStack.Screen) (poll : Poll)
    (s : State) (prior : Memory l) (k : Nat) (ans : Answer) (after : Memory l)
    (addr : c.address = p.address) (ok : ReadyMemory p mc fc prior)
    (cover : PvsParallel.Root.Covers mc b (s.rows.map Row.square)) (pos : idx < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) (full : level.val = 5)
    (ran : outcomeAt idx p c b base root level screen poll s prior = some (k,some ans,after)) :
    MultiExact.Certified mc fc b idx s ans ∧ ReadyMemory p mc fc after ∧
      MultiExact.trial idx p c b base root s prior = some (k,ans,after) := by
  obtain ⟨good,whole⟩ := outcomeAt_completed idx p c b base root level screen poll s prior k ans after
    addr ok cover pos window ran
  rw [ite_eq_left full] at whole
  obtain ⟨j,answer,post,run,cert,_⟩ := MultiExact.trial_returns idx p c b base root s prior addr ok cover pos window
  rw [whole] at run
  cases run
  exact ⟨cert,good,whole⟩

theorem trialAt_returns (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (level : Selectivity) (screen : WeakStack.Screen) (poll : Poll)
    (s : State) (prior : Memory l) (addr : c.address = p.address) (ok : ReadyMemory p mc fc prior)
    (cover : PvsParallel.Root.Covers mc b (s.rows.map Row.square)) (pos : idx < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) :
    ∃ after, trialAt idx p c b base root level screen poll s prior = some after ∧ ReadyMemory p mc fc after := by
  obtain ⟨k,answer,after,ran,good,_⟩ := outcomeAt_returns idx p c b base root level screen poll s prior addr ok cover pos window
  exact ⟨after,by simp [trialAt,ran],good⟩

/-- With no observed stop the entire root answer, not just its memory projection,
    agrees with the original exact or selective trial. -/
theorem outcomeAt_uninterrupted (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (level : Selectivity) (screen : WeakStack.Screen)
    (s : State) (prior : Memory l) :
    outcomeAt idx p c b base root level screen (fun _ _ _ => false) s prior =
      (if level.val = 5 then MultiExact.trial idx p c b base root s prior
        else MultiWeak.trial idx p c b base root level screen s prior).map
          (fun (k,answer,after) => (k,some answer,after)) := by
  let ctx := SerialRootExecution.context b s.regs.alpha s.regs.beta
  let code := rootCode idx c b base root s prior
  let whole : Stack.Child l := fun i f m => normal p c base level screen (Board.playRank mc f.request.board.val + 1) i f m
  let child : Child l := fun i f m => runFuel p c base level screen (fun _ _ _ => false) (Board.playRank mc f.request.board.val + 1) i f m
  have same := interpret_uninterrupted (mc := mc) (fc := fc) child whole
    (fun _ _ _ => uninterrupted p c base level screen _ _ _ _) code ctx s.opening prior
  change (interpret mc fc _ child ctx true s.opening prior code).bind _ = _
  rw [same]
  by_cases full : level.val = 5
  · have eq : level = Cutoffs.exactSelectivity := Fin.ext full
    subst level
    simp only [whole,normal,full,ite_true]
    unfold MultiExact.trial
    change ((Stack.interpret (mc := mc) (fc := fc) (Stack.run p c base) s.opening ctx prior code).map completed).bind _ = (((Stack.interpret (mc := mc) (fc := fc) (Stack.run p c base) s.opening ctx prior code).bind _).map _)
    cases Stack.interpret (mc := mc) (fc := fc) (Stack.run p c base) s.opening ctx prior code <;> rfl
  · simp only [whole,normal,full,ite_false]
    unfold MultiWeak.trial
    change ((Stack.interpret (mc := mc) (fc := fc) (WeakStack.run p c base level screen) s.opening ctx prior code).map completed).bind _ = (((Stack.interpret (mc := mc) (fc := fc) (WeakStack.run p c base level screen) s.opening ctx prior code).bind _).map _)
    cases Stack.interpret (mc := mc) (fc := fc) (WeakStack.run p c base level screen) s.opening ctx prior code <;> rfl

theorem trialAt_uninterrupted (idx : Nat) (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (level : Selectivity) (screen : WeakStack.Screen)
    (s : State) (prior : Memory l) :
    trialAt idx p c b base root level screen (fun _ _ _ => false) s prior =
      (if level.val = 5 then MultiExact.trial idx p c b base root s prior
        else MultiWeak.trial idx p c b base root level screen s prior).map (fun out => out.2.2) := by
  simp only [trialAt,outcomeAt_uninterrupted,Option.map_map,Function.comp_def]

def trial (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (level : Selectivity) (screen : WeakStack.Screen) (poll : Poll)
    (s : State) (prior : Memory l) : Option (Memory l) :=
  trialAt 0 p c b base root level screen poll s prior

theorem trial_returns (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (level : Selectivity) (screen : WeakStack.Screen) (poll : Poll)
    (s : State) (prior : Memory l) (addr : c.address = p.address) (ok : ReadyMemory p mc fc prior)
    (cover : PvsParallel.Root.Covers mc b (s.rows.map Row.square)) (pos : 0 < s.rows.length)
    (window : Aspiration.Window s.regs.alpha.toInt s.regs.beta.toInt) :
    ∃ after, trial p c b base root level screen poll s prior = some after ∧ ReadyMemory p mc fc after :=
  trialAt_returns 0 p c b base root level screen poll s prior addr ok cover pos window

theorem trial_uninterrupted (p : Pool) (c : SerialScout.Config) (b : Position) (base : Inputs mc fc)
    (root : SerialRecursion.RootInput) (level : Selectivity) (screen : WeakStack.Screen)
    (s : State) (prior : Memory l) :
    trial p c b base root level screen (fun _ _ _ => false) s prior =
      (if level.val = 5 then Trial.trial p c b base root s prior
        else WeakTrial.trial p c b base root level screen s prior).map (fun out => out.2.2) :=
  trialAt_uninterrupted 0 p c b base root level screen s prior

end NeuralReversi.PvsParallel.SerialTable.EndgameInterrupted

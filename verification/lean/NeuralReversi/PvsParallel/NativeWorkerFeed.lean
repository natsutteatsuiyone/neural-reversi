import NeuralReversi.PvsParallel.NativeFrames
import NeuralReversi.PvsParallel.SplitWorkerMir.Loop
import NeuralReversi.PvsParallel.ContextPathMir.Program
import NeuralReversi.PvsParallel.NativeWorkerFeed.Program

namespace NeuralReversi.PvsParallel.NativeWorkerFeed

open NativeFrames (Capture Row Record)
open PvsBoardTree (Position GeneratedSquare play build)
open SplitWorkerMir (Query Input Step FrameFeed)
open Threading.AtomicWord (Event SequencedBefore)
open EndgameSearch.CacheOwner (Node)
open Bitboard.Moves (X86Config)

structure Open where
  stamp : Nat
  site : Nat
  node : Node

structure Return where
  dispatch : Nat
  site : Nat
  opening : Nat
  step : Nat
  score : ScoreRefresh.Score
  usable : Bool

private def leave (stack : List Open) (out : List Return) (step opening : Nat) (node : Node)
    (raw : Option ScoreRefresh.Score) (usable : Bool) : Option (List Open × List Return) := do
  let top :: rest := stack | none
  if top.stamp != opening || top.node != node then none else
  match rest,node,raw with
  | caller :: _, .pvs, some score =>
    if caller.node == .splitSearch then some (rest,⟨caller.stamp,top.site,opening,step,score,usable⟩ :: out)
    else some (rest,out)
  | _,_,_ => some (rest,out)

private def advance (worker : Nat) (state : List Open × List Return) (record : Record) :
    Option (List Open × List Return) :=
  let (stack,out) := state
  match record with
  | .table _ => some (stack, out)
  | .localCall w ev =>
    if w != worker then some (stack, out) else
    match ev with
    | .entry step site node .. => some (⟨step,site,node⟩ :: stack, out)
    | .scored .. => some (stack, out)
    | .leave step opening node _ normal => leave stack out step opening node normal true
    | .discarded step opening raw => leave stack out step opening .pvs (some raw) false

/-- Recover direct split-worker calls from the worker's actual enter/leave stack. A scored
    child with no normal leave has no entry in the returned list. -/
def returns (worker : Nat) (stream : List Record) : Option (List Return) := do
  let (_, result) ← stream.foldlM (advance worker) ([],[])
  return result

private theorem leave_origin (stack : List Open) (out : List Return)
    (ran : leave stack out step opening node raw usable = some next) (mem : r ∈ next.2) :
    r ∈ out ∨ r.opening = opening ∧ r.step = step ∧ node = .pvs ∧ raw = some r.score ∧ r.usable = usable := by
  unfold leave at ran
  repeat' first | split at ran | contradiction
  all_goals cases ran
  all_goals first | exact Or.inl mem | rcases List.mem_cons.mp mem with rfl | old
  all_goals first | exact Or.inl old | exact Or.inr ⟨rfl,rfl,rfl,rfl,rfl⟩

/-- Every usable word was returned by a normal PVS leave on this worker. -/
theorem returned_event (ran : returns worker stream = some results) (mem : r ∈ results) (normal : r.usable = true) :
    ∃ stamp, Record.localCall worker (.leave r.step r.opening .pvs stamp (some r.score)) ∈ stream := by
  obtain ⟨last,parsed,end_⟩ := Option.bind_eq_some_iff.mp ran
  cases end_
  have visit (rs : List Record) (s : List Open × List Return)
      (done_ : rs.foldlM (advance worker) s = some last) :
      r ∈ s.2 ∨ ∃ stamp, Record.localCall worker (.leave r.step r.opening .pvs stamp (some r.score)) ∈ rs := by
    induction rs generalizing s with
    | nil => cases Option.some.inj done_; exact Or.inl mem
    | cons rec rest ih =>
      rw [List.foldlM_cons] at done_
      obtain ⟨next,one,more⟩ := Option.bind_eq_some_iff.mp done_
      rcases ih next more with now | ⟨stamp,tail⟩
      · cases rec with
        | table => cases one; exact Or.inl now
        | localCall w event =>
          by_cases same : w = worker
          · subst w
            cases event with
            | entry | scored => simp [advance] at one; subst next; exact Or.inl now
            | leave step opening node stamp word =>
              simp [advance] at one
              rcases leave_origin _ _ one now with old | ⟨opening_,step_,rfl,rfl,_⟩
              · exact Or.inl old
              · exact Or.inr ⟨stamp,by simp [opening_,step_]⟩
            | discarded =>
              simp [advance] at one
              rcases leave_origin _ _ one now with old | info
              · exact Or.inl old
              · simp [normal] at info
          · simp [advance,same] at one; subst next; exact Or.inl now
      · exact Or.inr ⟨stamp,List.mem_cons_of_mem _ tail⟩
  exact (visit stream ([],[]) parsed).resolve_left (by simp)

abbrev labels (mc : X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (order : List (GeneratedSquare mc b.val)) :=
  order.map fun m => build mc fc (play mc fc b m).val

/-- Match immutable split storage to the square and flip mask used by the owner. Priorities
    are already consumed when the list is sorted and do not enter the child board. -/
def frozen (mc : X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (moves : List Move.Move)
    (order : List (GeneratedSquare mc b.val)) : Bool :=
  decide ((moves.map fun m => (m.sq,m.flipped)) =
    order.map fun m => (m.val,Flip.Dispatch.resultValue fc m.val b.val.player b.val.opponent))

theorem site_enters_move (pv : Bool) : ContextPathMir.childSite true (site pv).1 = some .move := by
  cases pv <;> rfl

theorem ticket_at {rc : List Move.Move} {old : BitVec w} (got : ticket rc old = some (mv,ordinal)) :
    rc[old.toNat]? = some mv ∧ old.toNat < rc.length := by
  unfold ticket MoveList.Concurrent.next MoveList.Concurrent.advance at got
  split at got
  next h =>
    cases cell : rc[old.toNat]? with
    | none => simp [cell] at got
    | some value =>
      simp [cell] at got
      exact ⟨congrArg some got.1,h⟩
  next => simp at got

structure Prefix where
  worker : Nat
  dispatch : Nat
  claimed : Nat
  nextClaim : Option Nat
  scoutOpening : Nat
  wideOpening : Nat
  stops : Step → Event

def Prefix.opening (p : Prefix) (pv : Bool) := if pv then p.wideOpening else p.scoutOpening

def goodRow (r : Row) (worker opening : Nat) (time : Event) (board : Position) (q : Query) : Prop :=
  NativeFrames.whole r = true ∧ r.opening = opening ∧ r.frame.worker = worker ∧ time.worker = worker ∧
  r.frame.returnStep = some time.step ∧ r.frame.position = board ∧ r.frame.depth = q.depth ∧
  r.frame.pv = q.pv ∧ r.frame.cut = q.cut ∧ r.frame.alpha = q.alpha.toInt ∧
  r.frame.beta = q.beta.toInt ∧ r.frame.score = q.result.toInt

instance (r : Row) (worker opening : Nat) (time : Event) (board : Position) (q : Query) :
    Decidable (goodRow r worker opening time board q) := by unfold goodRow; infer_instance

def rawRecorded (results : List Return) (p : Prefix) (q : Query) : Bool :=
  results.any fun r => decide (r.dispatch = p.dispatch ∧ r.site = (site q.pv).2 ∧
    p.dispatch < p.claimed ∧ p.claimed < r.opening ∧ (∀ next ∈ p.nextClaim, r.step < next) ∧
    r.opening = p.opening q.pv ∧ r.step = (p.stops (.returned q)).step ∧ r.score.raw = q.result)

def recorded (results : List Return) (p : Prefix) (q : Query) : Bool :=
  rawRecorded (results.filter Return.usable) p q

def stepOk (mc : X86Config) (fc : Cutoffs.FlipConfig) (b : Position)
    (order : List (GeneratedSquare mc b.val)) (rc : List Move.Move) (old : BitVec w)
    (rows : List Row) (results : List Return) (p : Prefix) : Step → Bool
  | .returned q =>
    match ticket rc old, order[q.index]? with
    | some (mv,_), some chosen =>
      decide (old.toNat = q.index ∧ mv.sq = chosen.val ∧
        mv.flipped = Flip.Dispatch.resultValue fc chosen.val b.val.player b.val.opponent) &&
      recorded results p q &&
      rows.any (fun r => decide (goodRow r p.worker (p.opening q.pv) (p.stops (.returned q))
        (play mc fc b chosen).val q))
    | _, _ => false
  | _ => true

def ordered (time : Step → Event) (steps : List Step) : Bool :=
  decide (steps.Pairwise fun a b => (time a).worker = (time b).worker ∧ (time a).step < (time b).step)

private theorem order_adjacent (time : Step → Event) (pre post : List Step) (a b : Step)
    (ok : ordered time (pre ++ a :: b :: post) = true) : SequencedBefore (time a) (time b) := by
  exact (List.pairwise_cons.mp (List.pairwise_append.mp (of_decide_eq_true ok)).2.1).1 b (by simp)

def check (mc : X86Config) (fc : Cutoffs.FlipConfig) (b : Position) (rc : List Move.Move)
    (order : List (GeneratedSquare mc b.val)) (old : BitVec w) (rows : List Row) (stream : List Record)
    (p : Prefix) (i : Input) (budget : Nat) : Bool :=
  match SplitWorkerMir.search i, returns p.worker stream, ticket rc old with
  | some body, some results, some (_,ordinal) =>
    let steps := (body.steps ++ SplitWorkerMir.lockedSteps).take budget
    frozen mc fc b rc order && decide (i.index = old.toNat ∧ ordinal = i.index+1) &&
      ordered p.stops steps && steps.all (stepOk mc fc b order rc old rows results p)
  | _, _, _ => false

private theorem step_return {mc fc} {b : Position} {order : List (GeneratedSquare mc b.val)}
    {rc old rows results p q}
    (checked : stepOk mc fc b order rc (old : BitVec w) rows results p (.returned q) = true)
    (writes identify before sequenced) :
    let log := NativeFrames.logOf mc fc rows writes identify before sequenced
    ∃ e ∈ log.events, ∃ frame, e.id = identify (p.stops (.returned q)) ∧
      e.action = .returned frame ⟨childAt (labels mc fc b order) q.index, q.pv, q.alpha.toInt, q.beta.toInt, q.result.toInt⟩ := by
  dsimp only
  cases got : ticket rc old with
  | none => simp [stepOk,got] at checked
  | some pair =>
    cases chosen : order[q.index]? with
    | none => simp [stepOk,got,chosen] at checked
    | some m =>
      simp only [stepOk,got,chosen,Bool.and_eq_true,List.any_eq_true,decide_eq_true_eq] at checked
      obtain ⟨r,member,eligible,_,worker,timeWorker,returned,board,_,pv,_,alpha,beta,score⟩ := checked.2
      let log := NativeFrames.logOf mc fc rows writes identify before sequenced
      have logged : r.frame ∈ log.frames := List.mem_map.mpr ⟨r,List.mem_filter.mpr ⟨member,eligible⟩,rfl⟩
      have point : (⟨r.frame.worker,(p.stops (.returned q)).step⟩ : Event) = p.stops (.returned q) := by
        cases h : p.stops (.returned q) with
        | mk n k => simp_all
      have square : childAt (labels mc fc b order) q.index = build mc fc (play mc fc b m).val := by
        simp [childAt,labels,List.getElem?_map,chosen]
      refine ⟨_,log.returned r.frame logged returned,r.frame.id,congrArg identify point,?_⟩
      change ConcurrentReplay.Action.returned r.frame.id (r.frame.result mc fc) = _
      rw [square]
      unfold FrameReturns.Frame.result
      change ConcurrentReplay.Action.returned _ ⟨build mc fc r.frame.position,_,_,_,_⟩ = _
      rw [board,pv,alpha,beta,score]

theorem feed {mc fc} {b : Position} {rc order rows stream p i budget} {old : BitVec w}
    (accepted : check mc fc b rc order old rows stream p i budget = true)
    (body : SplitWorkerMir.Searched) (ran : SplitWorkerMir.search i = some body)
    (writes identify before sequenced) :
    FrameFeed (labels mc fc b order) identify (NativeFrames.logOf mc fc rows writes identify before sequenced).events
      p.stops ((body.steps ++ SplitWorkerMir.lockedSteps).take budget) := by
  cases observed : returns p.worker stream with
  | none => simp [check,ran,observed] at accepted
  | some results =>
    cases got : ticket rc old with
    | none => simp [check,ran,observed,got] at accepted
    | some pair =>
      simp only [check,ran,observed,got,Bool.and_eq_true,List.all_eq_true] at accepted
      constructor
      · constructor
        intro a b pre post shape
        exact order_adjacent p.stops pre post a b (shape ▸ accepted.1.2)
      · intro q mem
        exact step_return (accepted.2 _ mem) writes identify before sequenced

/-- Only a complete normal worker body can credit its split. -/
def credit (i : Input) (locked : SplitWorkerMir.Locked) (budget : Nat) :
    Option Threading.SplitResult.Report := do
  let body ← SplitWorkerMir.search i
  if (body.steps ++ SplitWorkerMir.lockedSteps).length ≤ budget then
    SplitWorkerMir.recordLocked i body locked
  else none

theorem credited_feed {mc fc} {b : Position} {rc order rows stream p i budget} {old : BitVec w}
    (accepted : check mc fc b rc order old rows stream p i budget = true)
    (credited : credit i locked budget = some report) (writes identify before sequenced) :
    ∃ body, SplitWorkerMir.search i = some body ∧ SplitWorkerMir.recordLocked i body locked = some report ∧
      FrameFeed (labels mc fc b order) identify (NativeFrames.logOf mc fc rows writes identify before sequenced).events
        p.stops (body.steps ++ SplitWorkerMir.lockedSteps) := by
  cases ran : SplitWorkerMir.search i with
  | none => simp [credit,ran] at credited
  | some body =>
    have h : (body.steps ++ SplitWorkerMir.lockedSteps).length ≤ budget ∧
        SplitWorkerMir.recordLocked i body locked = some report := by simpa [credit,ran] using credited
    exact ⟨body,rfl,h.2,
      by simpa only [List.take_of_length_le h.1] using feed accepted body ran writes identify before sequenced⟩

theorem cancelled_credit (i : Input) (locked : SplitWorkerMir.Locked) (budget : Nat) (stop : locked.stopped = true) :
    credit i locked budget = none := by
  cases h : SplitWorkerMir.search i with
  | none => simp [credit,h]
  | some body =>
    simp [credit,h,SplitWorkerMir.recordLocked_eq,stop]

structure Task where
  depth : TranspositionTable.Depth
  pv : Bool
  root : Bool
  cut : Bool
  beta : SplitWorkerMir.Word

private def checkReturn (p : Prefix) (pv stopped : Bool) (load ending : Event) (ret : Return) : Bool :=
  decide (load.worker = p.worker ∧ ending.worker = p.worker ∧
    ret.dispatch = p.dispatch ∧ ret.site = (site pv).2 ∧ ret.opening = p.opening pv ∧
    p.claimed < load.step ∧ load.step < ret.opening ∧ ret.opening < ending.step ∧ ret.step = ending.step ∧
    (stopped = true ∨ ret.usable = true))

def readWord (p : Prefix) (results : List Return) (pv : Bool) : Option SplitWorkerMir.Word :=
  (results.find? fun r => decide (r.dispatch = p.dispatch ∧ r.site = (site pv).2 ∧ r.opening = p.opening pv)).map
    (fun r => r.score.raw)

private def ticketReturns (p : Prefix) (results : List Return) (lock : Event) : List Return :=
  (results.filter fun r => decide (r.dispatch = p.dispatch ∧ p.claimed < r.opening ∧ r.step < lock.step)).reverse

/-- The lock observation is later than all calls of this ticket. A stopped ticket still
    has to show the physical scout and any re-search selected by that scout word. -/
def completeControl (p : Prefix) (task : Task) (results : List Return) (alpha : SplitWorkerMir.Word)
    (loads ends : Bool → Event) (lock : Event) (stopped : Bool) : Bool :=
  match ticketReturns p results lock with
  | [] => false
  | scout :: rest =>
    decide (lock.worker = p.worker ∧ p.dispatch < p.claimed ∧ (∀ n ∈ p.nextClaim, lock.step < n)) &&
    (decide (readWord p results false = some scout.score.raw) && checkReturn p false stopped (loads false) (ends false) scout) &&
    (if task.pv && decide (alpha.toInt < (-scout.score.raw).toInt) then
      match rest with
      | [wide] => decide (scout.step < (loads true).step ∧ readWord p results true = some wide.score.raw) &&
          checkReturn p true stopped (loads true) (ends true) wide
      | _ => false
    else rest.isEmpty)

private theorem control_sites {p task results alpha loads ends lock stopped}
    (checked : completeControl p task results alpha loads ends lock stopped = true) :
    p.dispatch < p.claimed ∧ (∀ n ∈ p.nextClaim, lock.step < n) ∧
    ∃ scout ∈ ticketReturns p results lock, readWord p results false = some scout.score.raw ∧
      checkReturn p false stopped (loads false) (ends false) scout = true ∧
      (task.pv = true → alpha.toInt < (-scout.score.raw).toInt →
        ∃ wide ∈ ticketReturns p results lock, readWord p results true = some wide.score.raw ∧
          scout.step < (loads true).step ∧ checkReturn p true stopped (loads true) (ends true) wide = true) := by
  unfold completeControl at checked
  cases items : ticketReturns p results lock with
  | nil => simp [items] at checked
  | cons scout rest =>
    obtain ⟨⟨gate,word,stamp⟩,next⟩ := (by simpa only [items,Bool.and_eq_true,decide_eq_true_eq] using checked :
      ((lock.worker = p.worker ∧ p.dispatch < p.claimed ∧ ∀ n ∈ p.nextClaim, lock.step < n) ∧ readWord p results false = some scout.score.raw ∧ checkReturn p false stopped (loads false) (ends false) scout = true) ∧ _)
    refine ⟨gate.2.1,gate.2.2,scout,by simp,word,stamp,?_⟩
    intro pv above
    simp [pv,above] at next
    cases rest with
    | nil => contradiction
    | cons wide more =>
      cases more with
      | cons => contradiction
      | nil =>
        obtain ⟨⟨before,raw⟩,ok⟩ := (by simpa only [Bool.and_eq_true,decide_eq_true_eq] using next :
          (scout.step < (loads true).step ∧ readWord p results true = some wide.score.raw) ∧
            checkReturn p true stopped (loads true) (ends true) wide = true)
        exact ⟨wide,by simp,raw,before,ok⟩

theorem complete_words {p task results alpha loads ends lock stopped}
    (checked : completeControl p task results alpha loads ends lock stopped = true) :
    ∃ scout ∈ results, readWord p results false = some scout.score.raw ∧
      (stopped = false → scout.usable = true) ∧
      (task.pv = true → alpha.toInt < (-scout.score.raw).toInt →
        ∃ wide ∈ results, readWord p results true = some wide.score.raw ∧ scout.step < (loads true).step ∧
          (stopped = false → wide.usable = true)) := by
  obtain ⟨_,_,s,mem,word,used,more⟩ := control_sites checked
  have member {r : Return} (h : r ∈ ticketReturns p results lock) := (List.mem_filter.mp (List.mem_reverse.mp h)).1
  have usable {mode r} (h : checkReturn p mode stopped (loads mode) (ends mode) r = true) (live : stopped = false) :
      r.usable = true := by simpa [live] using (of_decide_eq_true h).2.2.2.2.2.2.2.2.2
  refine ⟨s,member mem,word,usable used,?_⟩
  intro pv above
  obtain ⟨v,inList,w,before,ok⟩ := more pv above
  exact ⟨v,member inList,w,before,usable ok⟩

def childWord (p : Prefix) (results : List Return) (pv : Bool) : SplitWorkerMir.Word :=
  (readWord p results pv).getD 0

def received (task : Task) (p : Prefix) (results : List Return) (old : BitVec w)
    (alpha fresh : SplitWorkerMir.Word) : Input :=
  ⟨old.toNat,task.depth.toNat,task.pv,task.root,task.cut,task.beta,alpha,fresh,
    childWord p results false,childWord p results true⟩

/-- A credited query has its own usable return in the interval between this
    iterator claim and the next. The returned register is the word used by MIR. -/
theorem issued_return (p : Prefix) (results : List Return) (task : Task) (old : BitVec w)
    (alpha fresh : SplitWorkerMir.Word) (loads ends : Bool → Event) (lock : Event)
    (control : completeControl p task results alpha loads ends lock false = true)
    (stamps : ∀ q : Query, p.stops (.returned q) = ends q.pv)
    (ran : SplitWorkerMir.search (received task p results old alpha fresh) = some body)
    (issued : Step.returned q ∈ body.steps ++ SplitWorkerMir.lockedSteps) :
    q.index = old.toNat ∧ recorded results p q = true ∧ (ends q.pv).worker = p.worker ∧
      ∃ r ∈ results, r.usable = true ∧ r.opening = p.opening q.pv ∧ r.step = (ends q.pv).step ∧ r.score.raw = q.result := by
  obtain ⟨dispatch,next,scout,atScout,word,scoutOk,more⟩ := control_sites control
  have one (q : Query) (r : Return) (mem : r ∈ ticketReturns p results lock)
      (checked : checkReturn p q.pv false (loads q.pv) (ends q.pv) r = true)
      (value : r.score.raw = q.result) (index : q.index = old.toNat) :
      q.index = old.toNat ∧ recorded results p q = true ∧ (ends q.pv).worker = p.worker ∧
        ∃ r ∈ results, r.usable = true ∧ r.opening = p.opening q.pv ∧ r.step = (ends q.pv).step ∧ r.score.raw = q.result := by
    obtain ⟨_,worker,caller,site_,opened,_,_,_,returned,usable⟩ := of_decide_eq_true checked
    have paid : r.usable = true := usable.resolve_left (by decide)
    obtain ⟨member,range⟩ := List.mem_filter.mp (List.mem_reverse.mp mem)
    obtain ⟨_,claimed,prior⟩ := of_decide_eq_true range
    refine ⟨index,List.any_eq_true.mpr ⟨r,List.mem_filter.mpr ⟨member,paid⟩,decide_eq_true ?_⟩,worker,
      r,member,paid,opened,returned,value⟩
    exact ⟨caller,site_,dispatch,claimed,fun n mem => Nat.lt_trans prior (next n mem),opened,stamps q ▸ returned,value⟩
  let i := received task p results old alpha fresh
  rw [SplitWorkerMir.issued_of_search ran] at issued
  have scoutValue : i.scout = scout.score.raw := by simp [i,received,childWord,word]
  have scoutProof := one (SplitWorkerMir.scoutQuery i) scout atScout scoutOk scoutValue.symm rfl
  cases again : SplitWorkerMir.retry i with
  | false =>
    have eq : q = SplitWorkerMir.scoutQuery i := by simpa [SplitWorkerMir.issued,←show i = received .. from rfl,again,SplitWorkerMir.lockedSteps] using issued
    exact eq ▸ scoutProof
  | true =>
    have cases_ : q = SplitWorkerMir.scoutQuery i ∨ q = SplitWorkerMir.wideQuery i := by
      simpa [SplitWorkerMir.issued,←show i = received .. from rfl,again,SplitWorkerMir.lockedSteps] using issued
    rcases cases_ with rfl | rfl
    · exact scoutProof
    · obtain ⟨pv,improved⟩ := Bool.and_eq_true_iff.mp again
      obtain ⟨wide,atWide,raw,_,wideOk⟩ := more pv (scoutValue ▸ of_decide_eq_true improved)
      exact one _ _ atWide wideOk (by simp [SplitWorkerMir.wideQuery,i,received,childWord,raw]) rfl

/-- Both the exact frames and the child return words come from the same physical record
    stream. A missing word can only survive validation in a prefix before that return. -/
def decode {pool : EndgameSearch.CacheHistory.Pool} (mc : X86Config) (fc : Cutoffs.FlipConfig)
    (inputs : Nat → Nat → Option NativeFrames.Args) (stream : List Record) (parent : Position)
    (rc : List Move.Move) (order : List (GeneratedSquare mc parent.val)) (old : BitVec w)
    (p : Prefix) (task : Task) (alpha fresh : SplitWorkerMir.Word) (budget : Nat) : Option (Capture pool × Input) := do
  let c ← NativeFrames.decode pool inputs stream
  let results ← returns p.worker stream
  let i := received task p results old alpha fresh
  if check mc fc parent rc order old c.rows stream p i budget then some (c,i) else none

theorem decoded_credit {pool : EndgameSearch.CacheHistory.Pool} {mc fc inputs stream parent rc order}
    {old : BitVec w} {p task alpha fresh budget} {c : Capture pool} {i}
    (physical : decode mc fc inputs stream parent rc order old p task alpha fresh budget = some (c,i))
    (used : credit i locked budget = some report) (identify before sequenced) :
    NativeFrames.decode pool inputs stream = some c ∧ ∃ body, SplitWorkerMir.search i = some body ∧
      SplitWorkerMir.recordLocked i body locked = some report ∧ FrameFeed (labels mc fc parent order) identify
        (NativeFrames.logOf mc fc c.rows c.table identify before sequenced).events p.stops
        (body.steps ++ SplitWorkerMir.lockedSteps) := by
  unfold decode at physical
  cases cap : NativeFrames.decode pool inputs stream with
  | none => simp [cap] at physical
  | some actual =>
    cases out : returns p.worker stream with
    | none => simp [cap,out] at physical
    | some result =>
      rw [cap,out] at physical
      change (if check mc fc parent rc order old actual.rows stream p (received task p result old alpha fresh) budget
        then some (actual, received task p result old alpha fresh) else none) = some (c,i) at physical
      split at physical
      next hit =>
        cases physical
        exact ⟨rfl,credited_feed hit used _ identify before sequenced⟩
      next => contradiction

theorem missing_return {results p q} (absent : ∀ r ∈ results, r.opening ≠ (p : Prefix).opening q.pv ∨
    r.step ≠ (p.stops (.returned q)).step) : recorded results p q = false := by
  apply Bool.eq_false_iff.mpr
  intro h
  obtain ⟨r,mem,witness⟩ := List.any_eq_true.mp h
  obtain ⟨_,_,_,_,_,opening,step,_⟩ := of_decide_eq_true witness
  exact (absent r (List.mem_filter.mp mem).1).elim (fun no => no opening) (fun no => no step)

theorem partial_credit (ran : SplitWorkerMir.search i = some body)
    (pending : budget < (body.steps ++ SplitWorkerMir.lockedSteps).length) :
    credit i locked budget = none := by
  have few := Nat.not_le.mpr pending
  unfold credit
  rw [ran]
  change (if _ then _ else _) = _
  rw [ite_eq_right few]

/-- The modification order is for the physical iterator counter. A successful relaxed RMW
    reads its immediate predecessor and installs old+1 at this worker's debit stamp. -/
structure Debit (w : Nat) where
  address : Nat
  writes : List Threading.AtomicWord.Write
  source : Fin writes.length
  installed : Fin writes.length

def Debit.old (r : Debit w) : BitVec w := (r.writes.get r.source).value.setWidth w

def Debit.accept (r : Debit w) (address worker claimed : Nat) : Bool :=
  let write := r.writes.get r.installed
  decide ((w = 32 ∨ w = 64) ∧ r.address = address ∧ r.installed.val = r.source.val + 1 ∧
    write.rmw = true ∧ write.event = ⟨worker,claimed⟩ ∧ write.value.setWidth w = r.old + 1)

def credited (i : Input) (locked : SplitWorkerMir.Locked) (budget : Nat) (r : Threading.SplitResult.Report) : Bool :=
  (credit i locked budget).any fun actual =>
    decide (actual.index = r.index ∧ actual.searchedAlpha = r.searchedAlpha ∧ actual.score = r.score)

theorem credit_eq (seen : credited i locked budget r = true) : credit i locked budget = some r := by
  cases h : credit i locked budget with
  | none => simp [credited,h] at seen
  | some actual =>
    cases actual
    cases r
    simpa [credited,h] using seen

def forReport {moves : List PvsMoveLoop.Tree} (split : SplitExecution moves.length pv alpha beta score)
    (earlier : List Threading.SplitResult.Report) (r : Threading.SplitResult.Report)
    (later : List Threading.SplitResult.Report) (same : split.reports = earlier ++ r :: later)
    (worker dispatch claimed : Nat) (nextClaim : Option Nat) (scoutOpening wideOpening : Nat)
    (ends : Bool → Event) : Prefix :=
  ⟨worker,dispatch,claimed,nextClaim,scoutOpening,wideOpening,SplitWorkerMir.stamp
    (split.reportReads earlier r later same) (split.reportPublications earlier r later same).val ends⟩

def segment? {pool : EndgameSearch.CacheHistory.Pool} {mc fc inputs stream b rc order}
    (split : SplitExecution (labels mc fc b order).length pv alpha beta score)
    (earlier : List Threading.SplitResult.Report) (r : Threading.SplitResult.Report)
    (later : List Threading.SplitResult.Report) (same : split.reports = earlier ++ r :: later)
    (address : Nat) (debit : Debit w) (worker dispatch claimed : Nat) (nextClaim : Option Nat)
    (scoutOpening wideOpening : Nat) (ends : Bool → Event) {task scout reread budget}
    {c : Capture pool} {i : Input} (locked : SplitWorkerMir.Locked)
    (source : decode mc fc inputs stream b rc order debit.old
      (forReport split earlier r later same worker dispatch claimed nextClaim scoutOpening wideOpening ends)
      task scout reread budget = some (c,i))
    (identify before sequenced) :
    Option (SplitWorkerMir.Segment (moves := labels mc fc b order) split earlier r later same identify
      (NativeFrames.logOf mc fc c.rows c.table identify before sequenced).events) :=
  let raw := split.reportReads earlier r later same
  if !debit.accept address worker claimed then none else
  match h : SplitWorkerMir.search i with
  | none => none
  | some body =>
    if words : credited i locked budget r = true ∧ -2147483648 < i.alpha.toInt ∧ i.alpha.toInt < 2147483647 ∧ -2147483648 < i.scout.toInt ∧
        (SplitWorkerMir.retry i = true → -2147483648 < i.beta.toInt ∧ -2147483648 < i.fresh.toInt ∧ -2147483648 < i.wide.toInt) ∧
        i.pv = pv ∧ i.beta.toInt = beta ∧ i.alpha.toInt = raw.alpha ∧ i.scout.toInt = raw.scout ∧
        (SplitWorkerMir.retry i = true → i.fresh.toInt = raw.fresh ∧ i.wide.toInt = raw.wide) then
      some (by
        obtain ⟨paid,lo,hi,safeScout,safeRetry,node,betaWord,a,s,wider⟩ := words
        have checked := (decoded_credit source (credit_eq paid) identify before sequenced).2
        have evidence : SplitWorkerMir.recordLocked i body locked = some r ∧
            FrameFeed (labels mc fc b order) identify (NativeFrames.logOf mc fc c.rows c.table identify before sequenced).events
              (forReport split earlier r later same worker dispatch claimed nextClaim scoutOpening wideOpening ends).stops
              (body.steps ++ SplitWorkerMir.lockedSteps) := by
          obtain ⟨actual,ran,paid,events⟩ := checked
          rw [h] at ran
          cases ran
          exact ⟨paid,events⟩
        exact ⟨i,body,locked,h,evidence.1,⟨lo,hi,safeScout,fun t => (safeRetry t).1,
          fun t => (safeRetry t).2.1,fun t => (safeRetry t).2.2⟩,ends,evidence.2,node,betaWord,
          a,s,fun t => (wider t).1,fun t => (wider t).2⟩)
    else none

end NeuralReversi.PvsParallel.NativeWorkerFeed

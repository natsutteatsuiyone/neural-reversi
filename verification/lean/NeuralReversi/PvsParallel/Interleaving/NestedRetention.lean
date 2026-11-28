module
public import NeuralReversi.PvsParallel.Interleaving.StopSchedule
public import NeuralReversi.PvsParallel.SlotFrameMir.Score

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.NestedRetention

open Threading SlotFrameMir
open AtomicWord (Event Write SequencedBefore)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Action RetainedThrough)
open SlotFrameMir.RawProjection

private def maskOfActivation (a : SplitCompletion.Activation ws) : RawMask ws where
  owner := a.owner
  actions := a.actions
  mark := a.start
  bounded := a.endInBounds
  initial := a.ownerBit
  rmw := a.updates

variable {clock : Event → Nat}

private theorem local_order (causal : ∀ x y, HappensBefore h x y → clock x < clock y)
    (same : x.worker = y.worker) (ordered : clock x < clock y) : SequencedBefore x y := by
  refine ⟨same,?_⟩
  by_cases reverse : y.step < x.step
  · have earlier := causal _ _ (.sequenced ⟨same.symm, reverse⟩)
    omega
  · by_cases equal : x.step = y.step
    · have identical : x = y := by cases x; cases y; simp_all
      subst y; omega
    · omega

/-- The actual mask slice for the parent generation, including redundant physical
    resets, and the owner's zero retirement on the effective RMW projection. -/
structure Generation (h : History) where
  slot : Slot
  writes : List Write
  mask : RawMask writes
  closed : ClosedAfterFirst mask
  waited : SplitCompletion.Retirement h (words mask) (activation mask closed)
  actors : ∀ i w e, mask.actions.get i = .finish w e → e.worker = w.val

def Generation.wait (g : Generation h) : Event := g.waited.read.event

/-- Project the physical source of the owner's final acquire onto the effective
    mask updates. Several physical resets may project to the same release. -/
def retireRead {g : RawMask ws} (closed : ClosedAfterFirst g) (r : AtomicWord.Read ws)
    (zero : (ws.get r.source).value = 0) (acq : r.acquire = true)
    (markSB : SequencedBefore (ws.get g.mark).event r.event)
    (window : AtomicWord.ReadWindow r (HappensBefore h) g.mark (g.mark.val+g.actions.length))
    (sync : ∀ i, AtomicWord.InReleaseSequence ws i r.source → h.otherSync (ws.get i).event r.event) :
    SplitCompletion.Retirement h (words g) (activation g closed) where
  read := ⟨r.event,rf g r.source,r.acquire⟩
  acquire := acq
  zero := (rf_value g r.source window.bounds).trans zero
  afterOwnerBit := by
    rw [show ((words g).get (activation g closed).start).event = (ws.get g.mark).event from
      congrArg Write.event (word g 0 (by omega))]
    exact markSB
  beforeReuse := by
    intro i pos
    change i.val = 0 + (actions g).length + 1 at pos
    have len := Nat.lt_of_lt_of_eq i.isLt (length_words g)
    rw [length_actions] at pos
    omega
  writeRead := fun i hb => (rf_lower g r.source window.bounds.1 i).2
    (window.writeRead (back g i) (by simpa only [at_back] using hb))
  readWrite := by
    intro i hb
    have upper := window.readWrite (back g i) (by simpa only [at_back] using hb)
    apply Nat.lt_of_not_ge
    exact fun le => Nat.not_le_of_gt upper ((rf_lower g r.source window.bounds.1 i).1 le)
  synchronize := fun i _ seq => by
    simpa only [at_back] using sync (back g i) (lift_sequence g r.source window.bounds i seq)

def lastMask (log : List (SplitSchedulerMir.Guard ws)) : Option (Fin ws.length) :=
  match log.getLast? with
  | some (.mask src) => some src
  | _ => none

private theorem lastZero {g : RawMask ws} (exit : OwnerExit h g) :
    ∃ src, lastMask exit.log = some src ∧ exit.log = exit.log.dropLast ++ [.mask src] ∧
      (ws.get src).value = 0 := by
  obtain ⟨pre,src,post,site,zero⟩ := SplitSchedulerMir.idle_acquired_zero exit.returned exit.poolOpen
  obtain rfl := SplitSchedulerMir.after_zero exit.returned site zero
  exact ⟨src,by simp [lastMask,site],by simp [site],zero⟩

/-- The running idle loop observes zero on its final guard. The load's worker,
    instruction and physical source are taken from the observed exit. -/
def retire {g : RawMask ws} (exit : OwnerExit h g) (closed : ClosedAfterFirst g)
    (mark : (ws.get g.mark).event = exit.clock.atStep 1)
    (sync : ∀ pre src post, exit.log = pre ++ .mask src :: post → ∀ i,
      AtomicWord.InReleaseSequence ws i src →
        h.otherSync (ws.get i).event (exit.clock.atStep (4 + pre.length))) :
    SplitCompletion.Retirement h (words g) (activation g closed) :=
  let src := (lastMask exit.log).get (by obtain ⟨s,ok,_⟩ := lastZero exit; simp [ok])
  have data_ : exit.log = exit.log.dropLast ++ [.mask src] ∧ (ws.get src).value = 0 := by
    obtain ⟨s,ok,rest⟩ := lastZero exit
    simpa [src,ok] using rest
  let read : AtomicWord.Read ws := ⟨exit.clock.atStep (3+exit.log.length),src,true⟩
  have time : 4 + exit.log.dropLast.length = 3 + exit.log.length := by
    have len := congrArg List.length data_.1
    simp only [List.length_append,List.length_cons,List.length_nil] at len
    omega
  retireRead closed read data_.2 rfl
    (by rw [mark]; exact ⟨rfl,exit.clock.advances _ _ (by omega)⟩)
    (by simpa only [time,read] using exit.sampled _ _ [] data_.1)
    (fun i seq => by simpa only [time,read] using sync _ _ [] data_.1 i seq)

theorem ownerRetirement {g : RawMask ws} (exit : OwnerExit h g) (gates : GateHistory h g)
    (mark : (ws.get g.mark).event = exit.clock.atStep 1)
    (sync : ∀ pre src post, exit.log = pre ++ .mask src :: post → ∀ i,
      AtomicWord.InReleaseSequence ws i src →
        h.otherSync (ws.get i).event (exit.clock.atStep (4 + pre.length))) :
    ∃ r : SplitCompletion.Retirement h (words g) (activation g gates.closed),
      r.read.event = exit.clock.atStep (3 + exit.log.length) :=
  ⟨retire exit gates.closed mark sync,rfl⟩

/-- A set participation bit forces a later effective reset by this worker.
    The retiring load can precede redundant resets in the physical order. -/
theorem Generation.pass (g : Generation h)
    (causal : ∀ x y, HappensBefore h x y → clock x < clock y)
    (c : CaptureCut g.mask clock inner) (w : Worker) (who : inner.worker = w.val)
    (on : (g.writes.get c.cut).value.getLsbD w.val = true)
    (held : RetainedThrough h access inner) : RetainedThrough h access g.wait := by
  let a := activation g.mask g.closed
  let m := maskOfActivation a
  let lo := rf g.mask c.cut
  let hi := g.waited.read.source
  have top := (SplitCompletion.emptyReadContract_proven _ _ _ g.waited).1
  change hi.val = m.mark.val + m.actions.length at top
  have loBound : lo.val ≤ hi.val := by
    have lim := Nat.lt_of_lt_of_eq lo.isLt (length_words g.mask)
    change hi.val = 0 + (actions g.mask).length at top
    rw [length_actions] at top
    omega
  have bit : ((words g.mask).get lo).value.getLsbD w.val = true := by
    rw [show ((words g.mask).get lo).value = (g.writes.get c.cut).value from
      rf_value _ _ ⟨c.base,c.endAt⟩]
    exact on
  obtain ⟨i,e,op,later,_⟩ := m.release_between lo hi w (Nat.zero_le _) loBound (Nat.le_of_eq top) bit g.waited.zero
  let j : Fin (hits g.mask).length := ⟨i.val, by
    have lt := i.isLt
    change _ < (actions g.mask).length at lt
    rwa [length_actions] at lt⟩
  let k := (hits g.mask).get j
  have physical : g.mask.actions.get k = .finish w e := (action_at g.mask j).symm.trans op
  have rankLater : (cut g.mask c.cut c.base).rank.val < j.succ.val := by
    rw [← rf_val g.mask c.cut c.base]
    change lo.val < 0 + i.val + 1 at later
    change lo.val < i.val + 1
    omega
  have passed := (cut g.mask c.cut c.base).later j.succ rankLater
  have stamped := c.after k passed
  rw [physical] at stamped
  have sb := local_order causal (who.trans (g.actors k w e physical).symm) stamped
  apply RetainedThrough.wait g.waited i _ held
  · change SequencedBefore inner (m.actions.get i).event
    rw [op]; exact sb
  · change (m.actions.get i).release = true
    rw [op]; rfl

/-- Project the machine's reverse-chronological ticks on one reusable slot.
    Each generation is named by the owner-bit initialization event. -/
structure MaskLog where
  birth : Option Event
  updates : List Action

def MaskLog.add (m : MaskLog) (a : Action) : MaskLog := { m with updates := m.updates ++ [a] }

def MaskLog.step (s : Slot) (m : MaskLog) (e : Event) : Instruction → MaskLog
  | .openSlot k => if s = k then ⟨some e,[]⟩ else m
  | .reserve k w => if s = k then m.add (.book w e) else m
  | .release k w => if s = k then m.add (.finish w e) else m
  | .abandon (some k) w => if s = k then m.add (.finish w e) else m
  | _ => m

def maskLog (s : Slot) : List Tick → MaskLog
  | [] => ⟨none,[]⟩
  | (e,op)::past => (maskLog s past).step s e op

def MaskLog.bits (m : MaskLog) (s : Slot) :=
  m.updates.foldl (fun bits action => action.updateBits bits)
    (if m.birth.isSome then Bitboard.squareMask s.1 else 0)

private theorem log_bits (s : Slot) (past : List Tick) :
    (maskLog s past).bits s = (machine past).mask s := by
  induction past with
  | nil => rfl
  | cons tick past ih =>
    obtain ⟨e,op⟩ := tick
    cases op with
    | openSlot k =>
      change ((maskLog s past).step s e (.openSlot k)).bits s =
        if s = k then Bitboard.squareMask k.1 else (machine past).mask s
      by_cases same : s = k <;> simp [MaskLog.step,MaskLog.bits,same] at *
      exact ih
    | reserve k w =>
      change ((maskLog s past).step s e (.reserve k w)).bits s =
        if s = k then (Action.book w e).updateBits ((machine past).mask k) else (machine past).mask s
      by_cases same : s = k
      · subst k; simpa [MaskLog.step,MaskLog.add,MaskLog.bits,List.foldl_append] using congrArg (Action.book w e).updateBits ih
      · simpa [MaskLog.step,same] using ih
    | release k w =>
      change ((maskLog s past).step s e (.release k w)).bits s =
        if s = k then (Action.finish w e).updateBits ((machine past).mask k) else (machine past).mask s
      by_cases same : s = k
      · subst k; simpa [MaskLog.step,MaskLog.add,MaskLog.bits,List.foldl_append] using congrArg (Action.finish w e).updateBits ih
      · simpa [MaskLog.step,same] using ih
    | abandon taken w =>
      cases taken with
      | none => exact ih
      | some k =>
        change ((maskLog s past).step s e (.abandon (some k) w)).bits s =
          if s = k then (Action.finish w e).updateBits ((machine past).mask k) else (machine past).mask s
        by_cases same : s = k
        · subst k; simpa [MaskLog.step,MaskLog.add,MaskLog.bits,List.foldl_append] using congrArg (Action.finish w e).updateBits ih
        · simpa [MaskLog.step,same] using ih
    | capture | restore => exact ih

private theorem raw_prefix (g : RawMask ws) (n : Nat) (bound : n ≤ g.actions.length) :
    (g.actions.take n).foldl (fun bits a => a.updateBits bits) (Bitboard.squareMask g.owner) =
      (ws.get ⟨g.mark.val+n,by have := g.bounded; omega⟩).value := by
  induction n with
  | zero => exact g.initial.symm
  | succ n ih =>
    have lt : n < g.actions.length := by omega
    rw [List.take_succ_eq_append_getElem lt,List.foldl_append,List.foldl_cons,List.foldl_nil,ih (by omega)]
    exact (g.rmw ⟨n,lt⟩).2.2.2.symm

/-- Record the address taken by search_active_split_point and the birth of the
    slot it entered. An idle-loop restore changes the active pointer while the
    enclosing dispatch stays on this stack. -/
structure Loan where
  slot : Slot
  birth : Option Event
  entered : Event
  deriving DecidableEq

def loans (worker : Worker) : List Tick → List Loan
  | [] => []
  | (e,op)::past =>
    let below := loans worker past
    match op with
    | .capture s w => if worker = w then ⟨s,(maskLog s past).birth,e⟩ :: below else below
    | .release _ w => if worker = w then below.tail else below
    | .abandon _ w => if worker = w then below.filter (fun loan => loan.slot.1 != w) else below
    | _ => below

theorem loanSlots (w : Worker) (past : List Tick) :
    (loans w past).map Loan.slot = (machine past).frames w := by
  induction past with
  | nil => rfl
  | cons tick past ih =>
    obtain ⟨e,op⟩ := tick
    cases op with
    | openSlot | reserve | restore => exact ih
    | capture s k =>
      change (if w=k then _ :: loans w past else loans w past).map Loan.slot =
        if w=k then s :: (machine past).frames k else (machine past).frames w
      by_cases eq : w=k <;> simp_all
    | release s k =>
      change (if w=k then (loans w past).tail else loans w past).map Loan.slot =
        if w=k then (machine past).frames k |>.tail else (machine past).frames w
      by_cases eq : w=k <;> simp_all [List.map_tail]
    | abandon taken k =>
      change (if w=k then _ else loans w past).map Loan.slot =
        if w=k then ((machine past).frames k).filter _ else (machine past).frames w
      by_cases eq : w=k
      · subst k
        simp only [ite_true,←ih,List.filter_map]
        congr 1
      · simpa [eq] using ih

/-- Projection of the exact slot initialization and mask operations through the
    queried instruction. Matching the generation's birth distinguishes reused slots
    with identical occupancy. The frames are recomputed from all addresses. -/
structure AtCut (g : Generation h) (clock : Event → Nat) (inner : Event) where
  past : List Tick
  scheduled : Scheduled h clock past
  cut : CaptureCut g.mask clock inner
  worker : Worker
  who : inner.worker = worker.val
  owner : g.mask.owner = g.slot.1
  log : maskLog g.slot past = ⟨some (g.writes.get g.mask.mark).event,
    g.mask.actions.take (cut.cut.val - g.mask.mark.val)⟩
  entered : Event
  frame : ⟨g.slot,some (g.writes.get g.mask.mark).event,entered⟩ ∈ loans worker past

theorem AtCut.held {g : Generation h} (c : AtCut g clock inner)
    (causal : ∀ x y, HappensBefore h x y → clock x < clock y)
    (before : RetainedThrough h access inner) : RetainedThrough h access g.wait := by
  have frame : g.slot ∈ (machine c.past).frames c.worker := by
    rw [← loanSlots]; exact List.mem_map_of_mem c.frame
  have live := (c.scheduled.retains causal).2.1 g.slot c.worker frame
  rw [← log_bits, c.log] at live
  change ((g.mask.actions.take (c.cut.cut.val-g.mask.mark.val)).foldl _ (Bitboard.squareMask g.slot.1)).getLsbD c.worker.val = true at live
  rw [← c.owner] at live
  have base := c.cut.base
  have upper := c.cut.endAt
  have value := raw_prefix g.mask (c.cut.cut.val-g.mask.mark.val) (by omega)
  have idx : g.mask.mark.val+(c.cut.cut.val-g.mask.mark.val) = c.cut.cut.val := by omega
  have value' : (g.mask.actions.take (c.cut.cut.val-g.mask.mark.val)).foldl (fun bits a => a.updateBits bits) (Bitboard.squareMask g.mask.owner) =
      (g.writes.get c.cut.cut).value := by simpa only [idx] using value
  exact g.pass causal c.cut c.worker c.who (value' ▸ live) before

/-- Position after the last physical update at or before the queried event. -/
def pastUpdates (clock : Event → Nat) (e : Event) : List Action → Nat
  | [] => 0
  | a::as =>
    let n := pastUpdates clock e as
    if n > 0 ∨ clock a.event ≤ clock e then n+1 else 0

theorem pastUpdates_bound (clock : Event → Nat) (e : Event) (as : List Action) :
    pastUpdates clock e as ≤ as.length := by
  induction as with
  | nil => exact Nat.le_refl _
  | cons a as ih => simp only [pastUpdates,List.length_cons]; split <;> omega

theorem pastUpdates_after (clock : Event → Nat) (e : Event) (as : List Action)
    (i : Fin as.length) (later : pastUpdates clock e as ≤ i.val) : clock e < clock (as.get i).event := by
  induction as with
  | nil => exact Fin.elim0 i
  | cons a as ih =>
    simp only [pastUpdates] at later
    split at later
    · cases i using Fin.cases with
      | zero => simp at later
      | succ j => exact ih j (by change _+1 ≤ j.val+1 at later; omega)
    · rename_i neither
      cases i using Fin.cases with
      | zero => exact Nat.lt_of_not_ge (fun le => neither (Or.inr le))
      | succ j => exact ih j (by omega)

def currentCut (g : RawMask ws) (clock : Event → Nat) (e : Event) : CaptureCut g clock e where
  cut := ⟨g.mark.val + pastUpdates clock e g.actions, by have := g.bounded; have := pastUpdates_bound clock e g.actions; omega⟩
  base := Nat.le_add_right ..
  endAt := Nat.add_le_add_left (pastUpdates_bound clock e g.actions) _
  after i later := pastUpdates_after clock e g.actions i (by change g.mark.val + _ < g.mark.val + i.val + 1 at later; omega)

structure Tape (h : History) (clock : Event → Nat) where
  ticks : List Tick
  scheduled : Scheduled h clock ticks

def through (clock : Event → Nat) (e : Event) (ticks : List Tick) : List Tick :=
  ticks.dropWhile (fun t => decide (clock e < clock t.1))

theorem through_scheduled (s : Scheduled h clock ticks) (e : Event) :
    Scheduled h clock (through clock e ticks) := by
  induction s with
  | empty => exact .empty
  | @step past stamp op older src order ih =>
    by_cases later : clock e < clock stamp
    · simpa [through,later] using ih
    · simpa [through,later] using Scheduled.step older src order

/-- Both stack and mask checks use the same finite scheduling trace at the
    physical worker instruction. Neither worker release nor HB is an operand. -/
def AtCut.ofTape (g : Generation h) (t : Tape h clock) (inner captured : Event)
    (w : Worker) (who : inner.worker = w.val) (owner : g.mask.owner = g.slot.1)
    (log : maskLog g.slot (through clock inner t.ticks) =
      ⟨some (g.writes.get g.mask.mark).event, g.mask.actions.take (pastUpdates clock inner g.mask.actions)⟩)
    (frame : ⟨g.slot,some (g.writes.get g.mask.mark).event,captured⟩ ∈ loans w (through clock inner t.ticks)) :
    AtCut g clock inner where
  past := through clock inner t.ticks
  scheduled := through_scheduled t.scheduled inner
  cut := currentCut g.mask clock inner
  worker := w
  who := who
  owner := owner
  log := by simpa only [currentCut,Nat.add_sub_cancel_left] using log
  entered := captured
  frame := frame

/-- Newer local search_active_split_point frames above one suspended dispatch.
    Detach pops a newer work frame. Unwind removes this worker's own slot loans,
    as in Resources.exec; it can leave foreign loans on the stack. -/
def nestedStep (w : Worker) (s : Slot) (above : List Slot) : Instruction → Option (List Slot)
  | .capture k who => some (if w = who then k :: above else above)
  | .release _ who => if w = who then above.tail? else some above
  | .abandon _ who =>
    if w ≠ who then some above
    else if s.1 = w then none else some (above.filter (fun k => k.1 != w))
  | _ => some above

def nested (w : Worker) (s : Slot) : List Tick → Option (List Slot)
  | [] => some []
  | (_,op)::rest => (nested w s rest).bind (fun above => nestedStep w s above op)

private theorem above_capture (w : Worker) (s : Slot) (captured : Event) (older span : List Tick)
    {above : List Slot} (balanced : nested w s span = some above) :
    ∃ newer tail : List Loan, newer.map Loan.slot = above ∧
      loans w (span ++ (captured,.capture s w)::older) =
        newer ++ ⟨s,(maskLog s older).birth,captured⟩ :: tail := by
  induction span generalizing above with
  | nil =>
    cases balanced
    exact ⟨[],loans w older,rfl,by simp [loans]⟩
  | cons tick span ih =>
    obtain ⟨e,op⟩ := tick
    obtain ⟨xs,prior,changeAt⟩ := Option.bind_eq_some_iff.mp balanced
    obtain ⟨ns,ts,rfl,stack⟩ := ih prior
    cases op with
    | openSlot | reserve | restore =>
      cases changeAt; exact ⟨ns,ts,rfl,stack⟩
    | capture k who =>
      by_cases eq : w = who
      · subst who
        simp only [nestedStep,ite_true,Option.some.injEq] at changeAt
        exact ⟨⟨k,(maskLog k (span ++ (captured,.capture s w)::older)).birth,e⟩::ns,ts,changeAt,by simp [loans,stack]⟩
      · simp only [nestedStep,eq,ite_false,Option.some.injEq] at changeAt
        exact ⟨ns,ts,changeAt,by simpa [loans,eq] using stack⟩
    | release k who =>
      by_cases eq : w = who
      · subst who
        cases ns with
        | nil => simp [nestedStep] at changeAt
        | cons n ns =>
          simp [nestedStep] at changeAt
          exact ⟨ns,ts,changeAt,by simp [loans,stack]⟩
      · simp only [nestedStep,eq,ite_false,Option.some.injEq] at changeAt
        exact ⟨ns,ts,changeAt,by simpa [loans,eq] using stack⟩
    | abandon taken who =>
      by_cases eq : w = who
      · subst who
        by_cases own : s.1 = w
        · simp [nestedStep,own] at changeAt
        · simp [nestedStep,own] at changeAt
          exact ⟨ns.filter (fun a => a.slot.1 != w), ts.filter (fun a => a.slot.1 != w),
            by simpa [List.filter_map,Function.comp_def] using changeAt,by simp [loans,stack,own]⟩
      · simp [nestedStep,eq] at changeAt
        exact ⟨ns,ts,changeAt,by simpa [loans,eq] using stack⟩

/-- Construct the live ancestor from the native dispatch capture and intervening
    bookings, captures, detaches and pointer restores. No release index is supplied. -/
theorem loan_of_nested (w : Worker) (s : Slot) (captured : Event) (older span : List Tick)
    (balanced : (nested w s span).isSome) :
    ⟨s,(maskLog s older).birth,captured⟩ ∈ loans w (span ++ (captured,.capture s w)::older) := by
  obtain ⟨above,ok⟩ := Option.isSome_iff_exists.mp balanced
  obtain ⟨_,_,_,stack⟩ := above_capture w s captured older span ok
  rw [stack]; simp

def AtCut.ofDispatch (g : Generation h) (t : Tape h clock) (inner captured : Event)
    (w : Worker) (who : inner.worker = w.val) (owner : g.mask.owner = g.slot.1)
    (older span : List Tick)
    (atSite : through clock inner t.ticks = span ++ (captured,.capture g.slot w)::older)
    (birth : (maskLog g.slot older).birth = some (g.writes.get g.mask.mark).event)
    (log : maskLog g.slot (through clock inner t.ticks) =
      ⟨some (g.writes.get g.mask.mark).event, g.mask.actions.take (pastUpdates clock inner g.mask.actions)⟩)
    (nestedFrame : (nested w g.slot span).isSome) : AtCut g clock inner :=
  AtCut.ofTape g t inner captured w who owner log (by
    rw [atSite,← birth]
    exact loan_of_nested w g.slot captured older span nestedFrame)

/-- Ancestors in immediate-parent-first order. Each later observation is made
    by the preceding generation's owner at its actual last idle-loop acquire. -/
inductive Route (h : History) (clock : Event → Nat) : Event → List (Generation h) → Event → Type
  | done (e) : Route h clock e [] e
  | hop {g gs inner last} : AtCut g clock inner → Route h clock g.wait gs last →
      Route h clock inner (g::gs) last

theorem Route.retains (route : Route h clock inner gs last)
    (causal : ∀ x y, HappensBefore h x y → clock x < clock y)
    (held : RetainedThrough h access inner) : RetainedThrough h access last := by
  induction route with
  | done => exact held
  | hop c _ ih => exact ih (c.held causal held)

def parent {h : History} {scan : SplitReturn.Observation.AncestorScan h}
    {r : CutoffVisibility.FlagRead h.flags} (era : StopSchedule.Flags h r.slot)
    (initialized : HappensBefore h era.clear r.event) (endAt : Event)
    (gs : List (Generation h))
    (route : ∀ x ∈ scan.reads, x.slot = r.slot → Route h clock x.event gs endAt)
    (causal : ∀ x y, HappensBefore h x y → clock x < clock y)
    (next : ∀ x ∈ era.future, x.value = false → SequencedBefore endAt x.event) :
    StopSchedule.Parent h scan r :=
  ⟨era, initialized, endAt, fun x mem same => (route x mem same).retains causal .here, next⟩

namespace Regression

def parentSlot : Slot := (0,0)
def childSlot : Slot := (1,0)
def outerTicks : List Tick :=
  [(⟨1,2⟩,.capture parentSlot 1),(⟨0,1⟩,.reserve parentSlot 1),(⟨0,0⟩,.openSlot parentSlot)]
def innerTicks : List Tick :=
  [(⟨1,6⟩,.restore (some parentSlot) 1),(⟨1,5⟩,.release childSlot 1),
    (⟨1,4⟩,.capture childSlot 1),(⟨1,3⟩,.openSlot childSlot)]

/-- Finalizing a nested split restores the pointer. An extra detach of the
    parent or a later reuse at the same address cannot supply the old capture. -/
theorem nested_restore_and_reuse :
    (nested 1 parentSlot innerTicks).isSome = true ∧
    ⟨parentSlot,some ⟨0,0⟩,⟨1,2⟩⟩ ∈ loans 1 (innerTicks ++ outerTicks) ∧
    nested 1 parentSlot ((⟨1,7⟩,.release parentSlot 1)::innerTicks) = none ∧
    ⟨parentSlot,some ⟨0,0⟩,⟨1,2⟩⟩ ∉ loans 1
      ((⟨1,10⟩,.capture parentSlot 1)::(⟨0,9⟩,.reserve parentSlot 1)::
       (⟨0,8⟩,.openSlot parentSlot)::(⟨1,7⟩,.release parentSlot 1)::innerTicks ++ outerTicks) := by
  decide +kernel

/-- The scheduling abstraction alone admits restoring an unbooked pointer.
    The call/parent-link decoder must identify the original dispatch capture. -/
theorem restore_alone (h : History) (clock : Event → Nat) (e : Event) (s : Slot) (w : Worker) :
    Scheduled h clock [(e,.restore (some s) w)] ∧
    (machine [(e,.restore (some s) w)]).active w = .used s ∧
    loans w [(e,.restore (some s) w)] = [] := by
  refine ⟨.step .empty (.restore _ _) (by simp),?_,rfl⟩
  change (if w=w then _ else _) = _
  simp

end Regression

end NeuralReversi.PvsParallel.Interleaving.NestedRetention

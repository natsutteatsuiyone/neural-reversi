module
public import NeuralReversi.PvsParallel.SlotFrameMir.Roster

@[expose] public section

namespace NeuralReversi.PvsParallel.SlotFrameMir

open Threading
open AtomicWord (Event Read Write SequencedBefore ReadWindow)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (Activation Retirement Action)

def ownerWord (u : Worker) (e : Event) : Write := ⟨e,BitVec.ofNat 64 u.val,false,false⟩

/-- The first protected initialization and reuses in the same thread's physical
    bank publish the same owner ID. Earlier word contents may be arbitrary. -/
structure OwnerRead (hist : History) (slotOwner running : Worker) where
  before : List Write
  init : Event
  reuse : List Event
  word : Read (before ++ (init::reuse).map (ownerWord slotOwner))
  published : HappensBefore hist init word.event
  coherence : SplitRebookingMir.Coherence hist _ word
  observed : Worker
  decoded : ((before ++ (init::reuse).map (ownerWord slotOwner)).get word.source).value.toNat = observed.val
  didNotAbort : unwindAborts running observed = false

theorem OwnerRead.same (r : OwnerRead hist owner w) : owner = w := by
  let ws := r.before ++ (r.init::r.reuse).map (ownerWord owner)
  let first : Fin ws.length := ⟨r.before.length,by simp [ws]⟩
  have birth : (ws.get first).event = r.init := by simp [ws,first,ownerWord]
  have prior : HappensBefore hist (ws.get first).event r.word.event :=
    Eq.mpr (congrArg (fun x => HappensBefore hist x r.word.event) birth) r.published
  have lower := r.coherence.writeRead first prior
  change r.before.length ≤ r.word.source.val at lower
  have get : ws[r.word.source.val]? = some (ws.get r.word.source) := List.getElem?_eq_getElem _
  dsimp only [ws] at get
  rw [List.getElem?_append_right lower,List.getElem?_map] at get
  obtain ⟨e,_,actual⟩ := Option.map_eq_some_iff.mp get
  have decoded := r.decoded
  rw [← actual] at decoded
  have within : owner.val < 2^64 := by have := owner.isLt; omega
  have same : owner = r.observed := Fin.ext (by simpa [ownerWord,Nat.mod_eq_of_lt within] using decoded)
  exact same.trans (by simpa [unwindAborts] using r.didNotAbort : r.observed = w)

/-- This is the skipped-assignment case: while this prior idle call is still
    active it never captures the offered physical slot for a new work invocation.
    All attempted unwind resets expose the protected owner ID load and its branch. -/
inductive Unentered (hist : History) (u : Worker) (base : Nat) (a : Slot) :
    State → List Occurrence → Prop
  | done : Unentered hist u base a s []
  | next (ok : advance u base s item.op = some t)
      (skipped : item.op = .captureWork → s.active ≠ some a)
      (unwind : item.op = .finishUnwind → s.taken = some a → Nonempty (OwnerRead hist a.1 u))
      (tail : Unentered hist u base a t rest) : Unentered hist u base a s (item::rest)

private theorem advance_work (next : advance u base s op = some t) (old : ∀ x ∈ s.work, x ≠ some a)
    (skipped : op=.captureWork → s.active ≠ some a)
    (unwind : op=.finishUnwind → s.taken = some a → False) :
    (∀ x ∈ t.work, x ≠ some a) ∧ reset s op ≠ some a := by
  cases op with
  | book v => cases next; exact ⟨old,by simp [reset]⟩
  | captureParent => cases next; exact ⟨old,by simp [reset]⟩
  | enter => cases next; exact ⟨old,by simp [reset]⟩
  | exit =>
    cases eq : s.nested with
    | zero => simp [advance,eq] at next
    | succ n => simp [advance,eq] at next; subst t; exact ⟨old,by simp [reset]⟩
  | take => cases next; exact ⟨old,by simp [reset]⟩
  | captureWork =>
    cases next
    exact ⟨fun x loc => (List.mem_cons.mp loc).elim (fun eq => eq ▸ skipped rfl) (old x),by simp [reset]⟩
  | finishWork =>
    cases eq : s.work with
    | nil => simp [advance,eq] at next
    | cons q qs =>
      simp [advance,eq] at next; subst t
      have h : ∀ x ∈ q :: qs, x ≠ some a := eq ▸ old
      exact ⟨fun x mem => h x (List.mem_cons_of_mem _ mem),by simpa [reset,eq,workerReset] using h q (by simp)⟩
  | finishUnwind => cases next; exact ⟨old,unwind rfl⟩

theorem Unentered.no_reset (gap : Unentered hist u base a s ops) (run : Run u base s ops resets)
    (distinct : a.1 ≠ u) (old : some a ∉ s.work) : ∀ e, (a,e) ∉ resets := by
  induction gap generalizing resets with
  | done => cases run; simp
  | @next start end_ tail item ok skipped unwind _ ih =>
    cases run with
    | next same rest =>
      have eq := Option.some.inj (ok.symm.trans same)
      subst_vars
      have result := advance_work same (fun x member eq => old (eq ▸ member)) skipped
        (fun op eq => (unwind op eq).elim fun sample => distinct sample.same)
      intro e found
      rcases List.mem_append.mp found with before | after
      · cases actual : reset start item.op with
        | none => simp [actual] at before
        | some slot =>
          have same : (a,e) = (slot,item.atTime) := by simpa [actual] using before
          have equal : a = slot := congrArg Prod.fst same
          exact result.2 (equal ▸ actual)
      · exact ih rest (fun h => result.1 _ h rfl) e after

private theorem initial_before_reset (a : Activation mask) (raw : MaskEvents mask a) (early : List Reservation)
    (rest : List Action) (parts : a.actions = early.map Reservation.action ++ rest)
    (initial : .book w opened ∈ early.map Reservation.action) (j : Fin a.actions.length)
    (done : a.actions.get j = .finish w e) : raw.time opened < raw.time e := by
  have different : early.length ≤ j.val := by
    by_cases inEarly : j.val < early.length
    · have before : a.actions.get j = early[j.val].action := by
        change a.actions[j.val] = _
        simp [parts,List.getElem_append_left (by simpa using inEarly : j.val < (early.map Reservation.action).length)]
      have mistaken := before.symm.trans done
      cases mistaken
    · omega
  obtain ⟨k,pos⟩ := List.get_of_mem initial
  have len : a.actions.length = early.length + rest.length := (congrArg List.length parts).trans (by simp)
  have kBound : k.val < early.length := by simpa using k.isLt
  let i : Fin a.actions.length := ⟨k.val,by omega⟩
  have same : a.actions.get i = .book w opened := by
    refine Eq.trans ?_ pos
    change a.actions[k.val] = _
    simp [parts,List.getElem_append_left k.isLt]
  have ord : (a.updateIndex i).val < (a.updateIndex j).val := by
    change a.start.val + k.val+1 < a.start.val+j.val+1; omega
  have less := raw.ordered _ _ ord
  have x : (mask.get (a.updateIndex i)).event = opened := (a.updates i).1.trans (by rw [same]; rfl)
  have y : (mask.get (a.updateIndex j)).event = e := (a.updates j).1.trans (by rw [done]; rfl)
  rw [x,y] at less; exact less

structure WaitSegment (hist : History) (raw : MaskEvents mask a) (w : Worker) (slot : Nat)
    (opened bEmpty : Event) where
  start : State
  base : Nat
  ops : List Occurrence
  before : List (Slot × Event)
  during : List (Slot × Event)
  after : List (Slot × Event)
  executed : Run w base start ops during
  skipped : Unentered hist w base (a.owner,slot) start ops
  clockBefore : ∀ s e, (s,e) ∈ before → raw.time e ≤ raw.time opened
  clockAfter : ∀ s e, (s,e) ∈ after → SequencedBefore bEmpty e
  complete : ∀ j : Fin a.actions.length, ∀ e, a.actions.get j = .finish w e →
    ((a.owner,slot),e) ∈ before ++ during ++ after

/-- From two actual worker-clock intervals: a normal reset preceding B's empty
    read would have to use an A Arc already captured before A's initial booking.
    An unwind of a foreign slot would have requested cancellation before its reset. -/
theorem WaitSegment.orders {a : Activation mask} {raw : MaskEvents mask a}
    (span : WaitSegment hist raw w slot opened bEmpty) (distinct : a.owner ≠ w)
    (live : some (a.owner,slot) ∉ span.start.work)
    (early : List Reservation) (rest : List Action) (parts : a.actions = early.map Reservation.action ++ rest)
    (initial : .book w opened ∈ early.map Reservation.action) :
    ∀ e, Action.finish w e ∈ a.actions → SequencedBefore bEmpty e := by
  intro e found
  obtain ⟨j,done⟩ := List.get_of_mem found
  rcases List.mem_append.mp (span.complete j e done) with earlier | after
  · rcases List.mem_append.mp earlier with old | inside
    · have before := span.clockBefore _ e old
      have after := initial_before_reset a raw early rest parts initial j done; omega
    · exact (span.skipped.no_reset span.executed distinct live e inside).elim
  · exact span.clockAfter _ e after

/-- All four clock slices are over physical operations. A skipped initial assignment
    with two normal retirements implies either an A work Arc already live on the
    waiting worker when A reserved it, or a causally earlier duplicate booking. -/
theorem skipped_initial_requires_earlier_fault (hist : History) (a : Activation aMask) (b : Activation bMask)
    (aEmpty : Retirement hist aMask a) (bEmpty : Retirement hist bMask b) (helpful : Read bMask)
    (enroll : Event) (earlyA : List Reservation) (restA : List Action)
    (parts : a.actions = earlyA.map Reservation.action ++ restA)
    (posted : .book b.owner enroll ∈ earlyA.map Reservation.action)
    (range : ReadWindow helpful (HappensBefore hist) b.start (b.start.val+b.actions.length))
    (accepted : WorkerFrameMir.canJoin searching failed true deepest (AtomicBitSet.test (bMask.get helpful.source).value a.owner) = true)
    (different : a.owner ≠ b.owner) (slotA slotB : Nat) (rawA : MaskEvents aMask a) (rawB : MaskEvents bMask b)
    (waiter : WaitSegment hist rawA b.owner slotA enroll bEmpty.read.event)
    (owner : OwnerSegment a.owner b (b.owner,slotB) helpful.event aEmpty.read.event)
    (bookings : PostedWithin hist rawB a.owner (b.owner,slotB) helpful.event owner.events)
    (clock : ∀ x y, HappensBefore hist x y → rawB.time x < rawB.time y)
    (earlyB : List Reservation) (restB : List AfterStartup) (cursor : Scan b.owner (poolIndices size) earlyB)
    (bParts : b.actions = earlyB.map Reservation.action ++ restB.map AfterStartup.action) :
    some (a.owner,slotA) ∈ waiter.start.work ∨
    ∃ i j : Fin b.actions.length, ∃ earlier repeated, b.actions.get i = .book a.owner earlier ∧
      b.actions.get j = .book a.owner repeated ∧ i.val < j.val ∧ earlyB.length ≤ j.val ∧
      repeated.worker = a.owner.val ∧ rawB.time repeated < rawB.time bEmpty.read.event := by
  by_cases prior : some (a.owner,slotA) ∈ waiter.start.work
  · exact Or.inl prior
  · have inAll : .book b.owner enroll ∈ a.actions := parts ▸ List.mem_append_left _ posted
    exact Or.inr (crossed_wait_has_earlier_self_booking hist a b aEmpty bEmpty helpful enroll inAll range
      accepted (waiter.orders different prior earlyA restA parts posted) different slotB owner rawB bookings clock
      earlyB restB cursor bParts)

end NeuralReversi.PvsParallel.SlotFrameMir

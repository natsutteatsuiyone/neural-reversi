import NeuralReversi.PvsParallel.SlotFrameMir.Pool
import NeuralReversi.PvsParallel.SlotFrameMir.Gate

namespace NeuralReversi.PvsParallel.SlotFrameMir

open Threading.AtomicWord (Event)

structure Reservation where
  actor : Worker
  target : Worker
  time : Event

/-- The initial slot-lock hold sees at most one book of the just-yielded cursor.
    Guards may skip entries and the capacity test may terminate the scan early. -/
inductive Scan (owner : Worker) : List Nat → List Reservation → Prop
  | stop : Scan owner ids []
  | skip : Scan owner ids loans → Scan owner (i::ids) loans
  | book (w : Worker) (tick : Nat) : Scan owner ids loans →
      Scan owner (w.val::ids) (⟨owner,w,⟨owner.val,tick⟩⟩::loans)

theorem Scan.selected (r : Scan owner ids loans) : (loans.map (fun b => b.target.val)).Sublist ids := by
  induction r with
  | stop => simp
  | skip _ ih => exact ih.cons _
  | book w _ _ ih => exact ih.cons_cons _

theorem Scan.unique (r : Scan owner (poolIndices size) loans) : (loans.map (fun b => b.target.val)).Nodup := by
  have source : (poolIndices size).Nodup := by
    simp [poolIndices,List.nodup_cons,List.mem_range',List.nodup_range']
    omega
  exact r.selected.nodup source

open Threading.SplitCompletion (Action)

def Reservation.action (r : Reservation) : Action := .book r.target r.time

inductive AfterStartup
  | reserve (thread : Worker) (tick : Nat) | finish (thread : Worker) (tick : Nat)

def AfterStartup.action : AfterStartup → Action
  | .reserve w t => .book (lateTarget w) ⟨w.val,t⟩
  | .finish w t => .finish w ⟨w.val,t⟩

/-- The caller that performs the second set for a repeated bit is the target
    worker itself. Initial reservations use the owning thread's clock. -/
theorem duplicate_books_itself (r : Scan owner (poolIndices size) early) (late : List AfterStartup)
    (acts : List Action) (join : acts = early.map Reservation.action ++ late.map AfterStartup.action)
    (i j : Fin acts.length) (ordered : i.val < j.val)
    (first : acts.get i = .book w old) (second : acts.get j = .book w next) :
    early.length ≤ j.val ∧ next.worker = w.val := by
  have inLate : early.length ≤ j.val := by
    by_cases later : early.length ≤ j.val
    · exact later
    · have left : i.val < early.length := by omega
      have right : j.val < early.length := by omega
      have earlier : acts.get i = early[i.val].action := by
        change acts[i.val] = _; simp [join,List.getElem_append_left (by simpa using left : i.val < (early.map Reservation.action).length)]
      have laterEq : acts.get j = early[j.val].action := by
        change acts[j.val] = _; simp [join,List.getElem_append_left (by simpa using right : j.val < (early.map Reservation.action).length)]
      have one : early[i.val].target = w := (Action.book.inj (earlier.symm.trans first)).1
      have two : early[j.val].target = w := (Action.book.inj (laterEq.symm.trans second)).1
      have collision : (early.map (fun b => b.target.val))[i.val]'(by simp [left]) =
          (early.map (fun b => b.target.val))[j.val]'(by simp [right]) := by
        simpa using congrArg Fin.val (one.trans two.symm)
      have equal := r.unique.eq_of_getElem_eq (by simp [left]) (by simp [right]) collision
      omega
  refine ⟨inLate,?_⟩
  obtain ⟨k,atK⟩ : ∃ k : Fin late.length, acts.get j = (late.get k).action := by
    have size : acts.length = early.length+late.length := (congrArg List.length join).trans (by simp)
    have room := j.isLt
    let k : Fin late.length := ⟨j.val - early.length,by omega⟩
    refine ⟨k,?_⟩
    change acts[j.val] = _
    simp [join,List.getElem_append_right (by simpa using inLate : (early.map Reservation.action).length≤j.val),k]
  rw [atK] at second
  cases eq : late.get k with
  | finish who tick => rw [eq] at second; cases second
  | reserve who tick => rw [eq] at second; cases second; rfl

open Threading.AtomicWord (Read ReadWindow SequencedBefore)
open Threading.CutoffVisibility (History HappensBefore)
open Threading.SplitCompletion (Activation Retirement)

/-- Apply the startup cursor to the earlier duplicate forced by a crossed wait.
    Both its physical writer and its position relative to the actual B empty load
    are recovered; no reset-origin disjunction is required. -/
theorem crossed_wait_has_earlier_self_booking (hist : History) (a : Activation aMask)
    (b : Activation bMask) (aEmpty : Retirement hist aMask a) (bEmpty : Retirement hist bMask b)
    (helpful : Read bMask) (enroll : Event) (postedA : .book b.owner enroll ∈ a.actions)
    (range : ReadWindow helpful (HappensBefore hist) b.start (b.start.val+b.actions.length))
    (accepted : WorkerFrameMir.canJoin searching failed true deepest (Threading.AtomicBitSet.test (bMask.get helpful.source).value a.owner) = true)
    (waiting : ∀ e, Action.finish b.owner e ∈ a.actions → SequencedBefore bEmpty.read.event e)
    (other : a.owner ≠ b.owner) (slot : Nat)
    (segment : OwnerSegment a.owner b (b.owner,slot) helpful.event aEmpty.read.event)
    (raw : MaskEvents bMask b) (physical : PostedWithin hist raw a.owner (b.owner,slot) helpful.event segment.events)
    (causalClock : ∀ x y, HappensBefore hist x y → raw.time x < raw.time y)
    (early : List Reservation) (late : List AfterStartup) (cursor : Scan b.owner (poolIndices size) early)
    (actions : b.actions = early.map Reservation.action ++ late.map AfterStartup.action) :
    ∃ i j : Fin b.actions.length, ∃ earlier repeated, b.actions.get i = .book a.owner earlier ∧
      b.actions.get j = .book a.owner repeated ∧ i.val < j.val ∧ early.length ≤ j.val ∧
      repeated.worker = a.owner.val ∧ raw.time repeated < raw.time bEmpty.read.event := by
  obtain ⟨i,j,_,one,two,_,first,second,_,ordered,_,before⟩ := crossed_wait_gives_prior_duplicate
    hist a b aEmpty bEmpty helpful enroll postedA range accepted waiting other slot segment raw physical causalClock
  obtain ⟨lateAt,self⟩ := duplicate_books_itself cursor late b.actions actions i j ordered first second
  exact ⟨i,j,one,two,first,second,ordered,lateAt,self,before⟩

end NeuralReversi.PvsParallel.SlotFrameMir

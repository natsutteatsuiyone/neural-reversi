import NeuralReversi.PvsParallel.SlotFrameMir.ReceiverProgram

namespace NeuralReversi.PvsParallel.SlotFrameMir.Receiver

open Threading.AtomicWord (Event)

def adjacent (parent child : Site) := (incoming child).elim true (fun sites => sites.contains parent)

private def quiet : Site → Bool
  | .dispatch | .pvs | .scout | .bookJob | .other => true
  | _ => false

private theorem closed (parent child : Site) (safe : quiet parent=true)
    (call : adjacent parent child=true) (first : child≠.split) : quiet child=true := by
  revert safe call first;cases parent <;> cases child <;> decide

inductive Effect
  | replace (value : Option Slot) | searching (value : Bool)
  deriving DecidableEq

def emits (site : Site) : Effect → Bool
  | .replace _ => pointerWriters.contains site
  | .searching bit => searchingWriters.contains (site,bit)

private theorem quiet_writers (site : Site) (safe : quiet site=true) :
    (∀ p, emits site (.replace p)=false) ∧ emits site (.searching false)=false := by
  cases site <;> simp_all [quiet,emits,pointerWriters,searchingWriters]

inductive Op
  | call (s : Site) | ret | effect (op : Effect)
  deriving DecidableEq

abbrev EventOp := Event × Op

def top (s : List Site) := s.headD .dispatch

/-- Normal return and propagation of an unwind both pop the callee. The
    destructor, if any, executes its own emitted call before the surrounding
    frame disappears. The root is the outstanding dispatch_search invocation. -/
inductive Calls : List Site → List EventOp → Prop
  | done : Calls frames []
  | call : adjacent (top frames) s=true → Calls (s::frames) rest → Calls frames ((e,.call s)::rest)
  | ret : Calls frames rest → Calls (s::frames) ((e,.ret)::rest)
  | action : emits (top frames) op=true → Calls frames rest → Calls frames ((e,.effect op)::rest)

def untilSplit : List EventOp → List EventOp
  | [] => []
  | (e,op)::xs => if op=.call .split then [] else (e,op)::untilSplit xs

/-- The caller that enters the first chronological nested split is selected
    before applying any argument about which Arc any particular guard read. -/
def firstSplit : List EventOp → Option Event
  | [] => none
  | (e,op)::xs => if op=.call .split then some e else firstSplit xs

private theorem before_first (calls : Calls fs xs) (safe : ∀ f ∈ fs, quiet f=true) :
    (∀ e val, (e,.effect (.replace val)) ∉ untilSplit xs) ∧
      ∀ e, (e,.effect (.searching false)) ∉ untilSplit xs := by
  have current (fs : List Site) (safe : ∀ f ∈ fs,quiet f=true) : quiet (top fs)=true := by
    cases fs with | nil => rfl | cons f rest => exact safe f (by simp)
  induction calls with
  | done => simp [untilSplit]
  | @call fs s rest e edge _ ih =>
    by_cases next : s=.split
    · subst s;simp [untilSplit]
    · have nested:=ih (fun f mem => (List.mem_cons.mp mem).elim
        (fun eq => eq ▸ closed _ _ (current fs safe) edge next) (safe _))
      simpa [untilSplit,next] using nested
  | ret _ ih =>
    have rest:=ih (fun f mem => safe f (List.mem_cons_of_mem _ mem))
    simpa [untilSplit] using rest
  | @action fs op rest e writes _ ih =>
    have original:=ih safe
    have ⟨noPointer,noIdle⟩:=quiet_writers _ (current fs safe)
    cases op with
    | replace val => rw [noPointer] at writes;cases writes
    | searching bit =>
      cases bit
      · rw [noIdle] at writes;cases writes
      · simpa [untilSplit] using original

theorem Calls.untouched (calls : Calls [] xs) :
    (∀ e val, (e,.effect (.replace val)) ∉ untilSplit xs) ∧
      ∀ e, (e,.effect (.searching false)) ∉ untilSplit xs :=
  before_first calls (by simp)

theorem first_before (xs : List EventOp) (clock : Event → Nat)
    (ordered : List.Pairwise (fun (x y : EventOp) => clock x.1 < clock y.1) xs)
    (inStack : (e,.call .split) ∈ xs) :
    ∃ first, firstSplit xs=some first ∧ clock first ≤ clock e := by
  induction xs with
  | nil => cases inStack
  | cons x rest ih =>
    obtain ⟨stamp,op⟩:=x
    obtain ⟨before,tail⟩:=List.pairwise_cons.mp ordered
    rcases List.mem_cons.mp inStack with here | later
    · cases here;exact ⟨_,by simp [firstSplit],Nat.le_refl _⟩
    · by_cases split : op=.call .split
      · exact ⟨stamp,by simp [firstSplit,split],Nat.le_of_lt (before (e,.call .split) later)⟩
      · obtain ⟨first,found,le⟩:=ih tail later
        exact ⟨first,by simpa [firstSplit,split] using found,le⟩

def pushPop (frames : List (Event × Site)) : EventOp → List (Event × Site)
  | (e,.call site) => (e,site)::frames
  | (_,.ret) => frames.tail
  | (_,.effect _) => frames

def asWorkFrame (item : Event × Site) : WorkerFrameMir.Frame := ⟨workKind item.2,item.1⟩

private theorem split_on_stack (past : List EventOp) (frames : List (Event × Site))
    (active : (e,Site.split) ∈ past.foldl pushPop frames) :
    (e,Site.split) ∈ frames ∨ (e,Op.call .split) ∈ past := by
  induction past generalizing frames with
  | nil => exact Or.inl active
  | cons tick rest ih =>
    obtain ⟨stamp,op⟩:=tick
    rcases ih _ active with now | old
    · cases op with
      | call site =>
        rcases List.mem_cons.mp now with eq | before
        · cases eq;exact Or.inr (List.mem_cons_self ..)
        · exact Or.inl before
      | ret =>
        cases frames with
        | nil => cases now
        | cons h t => exact Or.inl (List.mem_cons_of_mem _ now)
      | effect => exact Or.inl now
    · exact Or.inr (List.mem_cons_of_mem _ old)

/-- A snapshot of the dynamic call stack relative to search_active. The
    dispatch frame directly beneath its callees is not a split. -/
structure StackCut (ops : List EventOp) (frames : List WorkerFrameMir.Frame) where
  past : List EventOp
  future : List EventOp
  pieces : ops=past++future
  dispatch : Event
  registers : frames=((past.foldl pushPop []).map asWorkFrame)++[⟨workKind .dispatch,dispatch⟩]

theorem StackCut.contains (c : StackCut ops frames) (first : WorkerFrameMir.FirstInner frames f) :
    (f.entered,Op.call .split) ∈ ops := by
  have present:=first.mem
  rw [c.registers] at present
  rcases List.mem_append.mp present with inStack | tail
  · obtain ⟨⟨e,site⟩,loc,eq⟩:=List.mem_map.mp inStack
    have kind:=congrArg WorkerFrameMir.Frame.kind eq
    change workKind site=f.kind at kind
    have only : workKind site=.split → site=.split := by cases site <;> decide
    have siteEq:=only (kind.trans first.1)
    subst site
    have event:=congrArg WorkerFrameMir.Frame.entered eq
    change e=f.entered at event
    have called:=(split_on_stack c.past [] loc).resolve_left (by simp)
    rw [c.pieces,←event]
    exact List.mem_append_left _ called
  · have eq : f=⟨workKind .dispatch,c.dispatch⟩ := by simpa using tail
    have bad:=first.1
    rw [eq] at bad;cases bad

end NeuralReversi.PvsParallel.SlotFrameMir.Receiver

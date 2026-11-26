import NeuralReversi.PvsParallel.Interleaving.SlotSynchronization

namespace NeuralReversi.PvsParallel.Interleaving.ActivePointer

open Threading
open AtomicWord (Event SequencedBefore)
open CutoffVisibility (History HappensBefore)
open SpinLock (Operation)
open TaskAncestry (Generation Path)

/-- Every non-atomic Option write to the selected worker's cell. The initial
    None belongs to the construction of the worker, before any operation here. -/
inductive Kind where
  | book | ownerActive | detach | restore
  deriving DecidableEq

structure Store where
  at_ : Event
  kind : Kind
  word : Option Nat
  deriving DecidableEq

def stores (target : Nat) (es : Joint.Tape Operand) : List Store :=
  es.filterMap fun row =>
    let emit (actor step worker : Nat) (kind : Kind) (word : Option Nat) :=
      if target = worker then some ⟨⟨actor,step⟩,kind,word⟩ else none
    match row with
    | .taskWord actor t (.book helper slot) => emit actor t helper .book (some slot)
    | .taskWord w t (.ownerActive slot) => emit w t w .ownerActive (some slot)
    | .taskWord w t (.detach _) | .taskWord w t .detachEmpty => emit w t w .detach none
    | .taskWord w t (.restore _ parent) => emit w t w .restore parent
    | _ => none

/-- A plain load observes the last HB-preceding write. Every other write must
    lie earlier than that writer or after the load. Source identity includes the
    instruction, even for repeated assignments of exactly the same Arc. -/
def Visible (hist : History) (ws : List Store) (load : Event) : Option Store → Prop
  | none => ∀ x ∈ ws, HappensBefore hist load x.at_
  | some s => s ∈ ws ∧ HappensBefore hist s.at_ load ∧
      ∀ x ∈ ws, x = s ∨ HappensBefore hist x.at_ s.at_ ∨ HappensBefore hist load x.at_

private theorem candidate (h : History) (ws : List Store) (load : Event)
    (dr : ∀ x ∈ ws, HappensBefore h x.at_ load ∨ HappensBefore h load x.at_)
    (ww : ∀ x ∈ ws, ∀ y ∈ ws, x ≠ y → HappensBefore h x.at_ y.at_ ∨ HappensBefore h y.at_ x.at_) :
    ∃ s, Visible h ws load s := by
  induction ws with
  | nil => exact ⟨none,by simp [Visible]⟩
  | cons x xs ih =>
    obtain ⟨s,vis⟩ := ih (fun y mem => dr y (List.mem_cons_of_mem _ mem))
      (fun y mem z mz => ww y (List.mem_cons_of_mem _ mem) z (List.mem_cons_of_mem _ mz))
    rcases dr x (by simp) with precedes | follows
    · cases s with
      | none =>
        refine ⟨some x,List.mem_cons_self ..,precedes,?_⟩
        intro z mem
        exact (List.mem_cons.mp mem).elim Or.inl (fun mz => Or.inr (Or.inr (vis z mz)))
      | some y =>
        obtain ⟨memY,old,dominates⟩ := vis
        by_cases eq : x=y
        · subst y
          exact ⟨some x,List.mem_cons_self ..,precedes,fun z m => (List.mem_cons.mp m).elim Or.inl (dominates z)⟩
        rcases ww x (by simp) y (List.mem_cons_of_mem _ memY) eq with before | after
        · refine ⟨some y,List.mem_cons_of_mem _ memY,old,?_⟩
          intro z m
          exact (List.mem_cons.mp m).elim (fun e => e ▸ Or.inr (Or.inl before)) (dominates z)
        · refine ⟨some x,List.mem_cons_self ..,precedes,?_⟩
          intro z m
          rcases List.mem_cons.mp m with eq | tail
          · exact Or.inl eq
          · rcases dominates z tail with rfl | lt | later
            · exact Or.inr (Or.inl after)
            · exact Or.inr (Or.inl (lt.trans after))
            · exact Or.inr (Or.inr later)
    · cases s with
      | none =>
        exact ⟨none,fun y m => (List.mem_cons.mp m).elim (fun e => e ▸ follows) (vis y)⟩
      | some y =>
        exact ⟨some y,List.mem_cons_of_mem _ vis.1,vis.2.1,fun z m =>
          (List.mem_cons.mp m).elim (fun e => e ▸ Or.inr (Or.inr follows)) (vis.2.2 z)⟩

theorem Visible.unique {read : Event} {s t : Option Store} (first : Visible h ws read s) (other : Visible h ws read t)
    (acyclic : ∀ e, ¬ HappensBefore h e e) : s=t := by
  cases s with
  | none =>
    cases t with
    | none => rfl
    | some t => exact (acyclic _ ((first t other.1).trans other.2.1)).elim
  | some s =>
    cases t with
    | none => exact (acyclic _ ((other s first.1).trans first.2.1)).elim
    | some t =>
      by_cases eq : s=t
      · exact congrArg some eq
      rcases (first.2.2 t other.1).resolve_left (Ne.symm eq) with prior | late
      · rcases (other.2.2 s first.1).resolve_left eq with reversed | tooLate
        · exact (acyclic _ (prior.trans reversed)).elim
        · exact (acyclic _ (first.2.1.trans tooLate)).elim
      · exact (acyclic _ (other.2.1.trans late)).elim

/-- Existence and uniqueness for arbitrary many stores in any order in the
    input list. DRF orders conflicts on this address; equal values play no role. -/
theorem unique_source (h : History) (ws : List Store) (r : Event)
    (acyclic : ∀ e, ¬ HappensBefore h e e)
    (rw : ∀ x ∈ ws, HappensBefore h x.at_ r ∨ HappensBefore h r x.at_)
    (ww : ∀ x ∈ ws, ∀ y ∈ ws, x ≠ y → HappensBefore h x.at_ y.at_ ∨ HappensBefore h y.at_ x.at_) :
    ∃ s, Visible h ws r s ∧ ∀ t, Visible h ws r t → t=s := by
  obtain ⟨s,ok⟩ := candidate h ws r rw ww
  exact ⟨s,ok,fun _ also => Visible.unique also ok acyclic⟩

private def hb (c : MaskClock.Capture h) (a b : Event) :=
  if a.worker = b.worker then decide (a.step < b.step) else (FlagSegment.published c.transfers a b).isSome

private theorem hb_iff (c : MaskClock.Capture h) (a b : Event) :
    hb c a b = true ↔ HappensBefore h a b := by
  unfold hb
  split
  · rename_i same
    simp only [decide_eq_true_eq]
    refine ⟨fun lt => .sequenced ⟨same,lt⟩,fun edge => ?_⟩
    apply Nat.lt_of_not_ge
    intro rev
    rcases Nat.eq_or_lt_of_le rev with eq | lt
    · have ident : a = b := by cases a; cases b; simp_all
      subst b
      exact Nat.lt_irrefl _ (c.before a a edge)
    · exact Nat.lt_asymm (c.before a b edge) (c.before b a (.sequenced ⟨same.symm,lt⟩))
  · exact (MaskClock.published h c a b).symm

private def fits (c : MaskClock.Capture h) (ws : List Store) (read : Event) : Option Store → Bool
  | none => ws.all (fun w => hb c read w.at_)
  | some s => ws.contains s && hb c s.at_ read &&
    ws.all fun x => x == s || hb c x.at_ s.at_ || hb c read x.at_

private theorem fits_iff (c : MaskClock.Capture h) (ws : List Store) (r : Event) (s : Option Store) :
    fits c ws r s = true ↔ Visible h ws r s := by
  cases s with
  | none => simp [fits,Visible,hb_iff]
  | some s => simp [fits,Visible,hb_iff,and_assoc,or_assoc]

def source (c : MaskClock.Capture h) (ws : List Store) (r : Event) : Option (Option Store) :=
  (none :: ws.map some).find? (fits c ws r)

theorem source_visible {c : MaskClock.Capture h} (ok : source c ws r = some s) : Visible h ws r s :=
  (fits_iff c ws r s).mp (List.find?_some (p := fits c ws r) (l := none :: ws.map some) ok)

theorem source_complete (c : MaskClock.Capture h) (ws : List Store) (r : Event)
    (rw : ∀ x ∈ ws, HappensBefore h x.at_ r ∨ HappensBefore h r x.at_)
    (ww : ∀ x ∈ ws, ∀ y ∈ ws, x ≠ y → HappensBefore h x.at_ y.at_ ∨ HappensBefore h y.at_ x.at_) :
    ∃ s, source c ws r = some s ∧ ∀ t, Visible h ws r t → t=s := by
  have acyclic := fun e edge => Nat.lt_irrefl _ (c.before e e edge)
  obtain ⟨s,visible,_⟩ := unique_source h ws r acyclic rw ww
  cases eq : source c ws r with
  | some t => exact ⟨t,rfl,fun _ ok => Visible.unique ok (source_visible eq) acyclic⟩
  | none =>
    have absent := List.find?_eq_none.mp (show (none :: ws.map some).find? (fits c ws r) = none from eq)
    have present : s ∈ none :: ws.map some := by
      cases s with
      | none => simp
      | some t => exact List.mem_cons_of_mem _ (List.mem_map_of_mem visible.1)
    have off := absent s present
    exact (off ((fits_iff c ws r s).mpr visible)).elim

structure Guard where
  acquire : Event
  release : Option Event
  deriving DecidableEq

/-- Successful CAS and guard drops of a *single* physical lock, including the
    open last critical section in an observation ending before its destructor. -/
def guards : List Operation → Option (List Guard)
  | [] => some []
  | [.acquire a] => some [⟨a,none⟩]
  | .acquire a :: .release r :: rest =>
    if a.worker = r.worker ∧ a.step < r.step then (guards rest).map (⟨a,some r⟩ :: ·) else none
  | _ => none

private def encloses (g : Guard) (e : Event) :=
  g.acquire.worker == e.worker && g.acquire.step < e.step &&
    g.release.all (fun r => r.worker == e.worker && e.step < r.step)

structure Section (ops : List Operation) (e : Event) where
  guard : Guard
  sections : List Guard
  paired : guards ops = some sections
  member : guard ∈ sections
  entered : SequencedBefore guard.acquire e
  beforeDrop : ∀ r, guard.release = some r → SequencedBefore e r

def guardAt (ops : List Operation) (e : Event) : Option (Section ops e) := do
  match found : guards ops with
  | none => none
  | some gs =>
    match one : gs.filter (fun g => encloses g e) with
    | [g] =>
      have picked : g ∈ gs.filter (fun z => encloses z e) := by rw [one]; simp
      have ok := (List.mem_filter.mp picked).2
      have ok : (g.acquire.worker=e.worker ∧ g.acquire.step<e.step) ∧ _ :=
        (by simpa only [encloses,Bool.and_eq_true_iff,beq_iff_eq,decide_eq_true_eq] using ok)
      return ⟨g,gs,found,(List.mem_filter.mp picked).1,ok.1,fun r hr => by
        have last := ok.2
        simp [hr] at last
        exact ⟨last.1.symm,last.2⟩⟩
    | _ => none

private def sectionsValid (c : MaskClock.Capture h) (ops : List Operation) :=
  (SlotSynchronization.lockEdges ops).all c.sync.edges.contains

private def word (path : Path) := path.head?.map Generation.slot

/-- Semantic statement of `storeSite`: every recorded store matches the task journal
    word and holds the locks required by its kind. Note that `ownerActive` holds only
    the split-point lock, not the target worker's state lock. -/
def ValidStore (address : TaskRetention.Address) (es : Joint.Tape Operand)
    (j : TaskAncestry.Journal) (target : Nat) (s : Store) : Prop :=
  ∃ path, j.assignments s.at_ = some path ∧ word path = s.word ∧
    match s.kind with
    | .detach | .restore =>
      s.at_.worker = target ∧ (guardAt (SlotSynchronization.workerOps target es) s.at_).isSome = true
    | .book =>
      ∃ gen owner bank, path.head? = some gen ∧ MaskGeneration.locate address es gen = some (owner,bank) ∧
        (guardAt (SlotSynchronization.operations owner bank es) s.at_).isSome = true ∧
        (guardAt (SlotSynchronization.workerOps target es) s.at_).isSome = true
    | .ownerActive =>
      ∃ gen owner bank, path.head? = some gen ∧ MaskGeneration.locate address es gen = some (owner,bank) ∧
        (guardAt (SlotSynchronization.operations owner bank es) s.at_).isSome = true ∧
        s.at_.worker = target ∧ owner.val = target

/-- Only ownerActive initializes the owner's pointer without taking the target
    worker mutex. This store still holds the split slot lock. book_into holds
    both locks. detach_split and finalize_split_point hold the worker lock. -/
def storeSite (c : MaskClock.Capture h) (address : TaskRetention.Address) (es : Joint.Tape Operand)
    (j : TaskAncestry.Journal) (target : Nat) (s : Store) : Bool := Id.run do
  let some path := j.assignments s.at_ | return false
  if word path != s.word then return false
  let workerHeld := (guardAt (SlotSynchronization.workerOps target es) s.at_).isSome
  match s.kind with
  | .detach | .restore => return s.at_.worker == target && workerHeld
  | .book | .ownerActive =>
    let some gen := path.head? | return false
    let some (owner,bank) := MaskGeneration.locate address es gen | return false
    let slotOps := SlotSynchronization.operations owner bank es
    if !(sectionsValid c slotOps && (guardAt slotOps s.at_).isSome) then return false
    if s.kind == .book then return workerHeld
    else return s.at_.worker == target && owner.val == target

private theorem storeSite_valid {c : MaskClock.Capture h}
    (ok : storeSite c address es j target s = true) : ValidStore address es j target s := by
  unfold storeSite at ok
  cases assign : j.assignments s.at_ with
  | none => simp [assign] at ok
  | some path =>
    simp only [assign] at ok
    by_cases sameWord : word path = s.word
    · simp only [sameWord,bne_self_eq_false,Bool.false_eq_true,↓reduceIte] at ok
      refine ⟨path,assign,sameWord,?_⟩
      cases kind : s.kind with
      | detach | restore =>
        simp only [kind,Id.run,pure,Bool.and_eq_true,beq_iff_eq] at ok
        exact ok
      | book =>
        simp only [kind] at ok
        cases head : path.head? with
        | none => simp [head] at ok
        | some gen =>
          simp only [head] at ok
          cases loc : MaskGeneration.locate address es gen with
          | none => simp [loc] at ok
          | some pair =>
            obtain ⟨owner,bank⟩ := pair
            simp only [loc] at ok
            by_cases slotOk : sectionsValid c (SlotSynchronization.operations owner bank es) = true ∧
                (guardAt (SlotSynchronization.operations owner bank es) s.at_).isSome = true
            · simp only [slotOk.1,slotOk.2,Bool.and_self,Bool.not_true,Bool.false_eq_true,↓reduceIte,
                beq_self_eq_true,Id.run,pure] at ok
              exact ⟨gen,owner,bank,rfl,loc,slotOk.2,ok⟩
            · have fail : (sectionsValid c (SlotSynchronization.operations owner bank es) &&
                  (guardAt (SlotSynchronization.operations owner bank es) s.at_).isSome) = false := by
                cases h1 : sectionsValid c (SlotSynchronization.operations owner bank es) <;>
                  cases h2 : (guardAt (SlotSynchronization.operations owner bank es) s.at_).isSome <;> simp_all
              simp [fail] at ok
      | ownerActive =>
        simp only [kind] at ok
        cases head : path.head? with
        | none => simp [head] at ok
        | some gen =>
          simp only [head] at ok
          cases loc : MaskGeneration.locate address es gen with
          | none => simp [loc] at ok
          | some pair =>
            obtain ⟨owner,bank⟩ := pair
            simp only [loc] at ok
            by_cases slotOk : sectionsValid c (SlotSynchronization.operations owner bank es) = true ∧
                (guardAt (SlotSynchronization.operations owner bank es) s.at_).isSome = true
            · simp only [slotOk.1,slotOk.2,Bool.and_self,Bool.not_true,Bool.false_eq_true,↓reduceIte,
                reduceCtorEq,Id.run,pure,Bool.and_eq_true,beq_iff_eq] at ok
              exact ⟨gen,owner,bank,rfl,loc,slotOk.2,ok.1,ok.2⟩
            · have fail : (sectionsValid c (SlotSynchronization.operations owner bank es) &&
                  (guardAt (SlotSynchronization.operations owner bank es) s.at_).isSome) = false := by
                cases h1 : sectionsValid c (SlotSynchronization.operations owner bank es) <;>
                  cases h2 : (guardAt (SlotSynchronization.operations owner bank es) s.at_).isSome <;> simp_all
              simp [fail] at ok
    · have ne : (word path != s.word) = true := bne_iff_ne.mpr sameWord
      simp [ne] at ok

private def assignment (j : TaskAncestry.Journal) : Option Store → Option Path
  | none => some []
  | some s => j.assignments s.at_

structure Clone (h : History) (address : TaskRetention.Address) (es : Joint.Tape Operand) (e : Event) where
  path : Path
  raw : Option Nat
  writer : Option Store
  journal : TaskAncestry.Journal
  replayed : TaskAncestry.capture address es = some journal
  viewed : TaskAncestry.View
  atRead : journal.clones e = some viewed
  samePath : path = viewed.path
  atWrite : assignment journal writer = some path
  guarded : Section (SlotSynchronization.workerOps e.worker es) e
  visible : Visible h (stores e.worker es) e writer
  validStores : ∀ x ∈ stores e.worker es, ValidStore address es journal e.worker x
  writeOrder : ∀ x ∈ stores e.worker es, ∀ y ∈ stores e.worker es,
    x ≠ y → HappensBefore h x.at_ y.at_ ∨ HappensBefore h y.at_ x.at_
  matchesWord : (writer.bind Store.word) = raw
  addresses : word path = raw

private def cloneWord (es : Joint.Tape Operand) (e : Event) : Option (Option Nat) :=
  match es.filterMap fun row => match row with
    | .taskWord w t (.cloneActive value) => if e == ⟨w,t⟩ then some value else none
    | _ => none with
  | [raw] => some raw
  | _ => none

def clone (c : MaskClock.Capture h) (address : TaskRetention.Address) (es : Joint.Tape Operand)
    (e : Event) : Option (Clone h address es e) := do
  let captured ← (TaskAncestry.capture address es).attach
  let j := captured.val
  let viewed ← (j.clones e).attach
  let raw ← cloneWord es e
  let ops := SlotSynchronization.workerOps e.worker es
  guard (sectionsValid c ops)
  let held ← guardAt ops e
  let ws := stores e.worker es
  guard ((ws.map Store.at_).Nodup)
  if sites : ws.all (storeSite c address es j e.worker) then
    let comparable (x y : Store) := x == y || hb c x.at_ y.at_ || hb c y.at_ x.at_
    if drf : ws.all (fun x => ws.all (comparable x)) then
      match found : source c ws e with
      | none => none
      | some writer =>
        let saved ← (assignment j writer).attach
        let path := saved.val
        if aligned : path = viewed.val.path then
          if observed : writer.bind Store.word = raw then
            if named : word path = raw then
              have allValid : ∀ x ∈ ws, ValidStore address es j e.worker x :=
                fun x mem => storeSite_valid (List.all_eq_true.mp sites x mem)
              have ordered : ∀ x ∈ ws, ∀ y ∈ ws, x ≠ y → HappensBefore h x.at_ y.at_ ∨ HappensBefore h y.at_ x.at_ := by
                intro x mx y my ne
                have pair := List.all_eq_true.mp (List.all_eq_true.mp drf x mx) y my
                simp only [comparable,Bool.or_eq_true,beq_iff_eq,hb_iff] at pair
                exact pair.elim (fun h => h.elim (fun eq => (ne eq).elim) Or.inl) Or.inr
              return ⟨path,raw,writer,j,captured.property,viewed.val,viewed.property,aligned,saved.property,held,
                source_visible found,allValid,ordered,observed,named⟩
        none
    else none
  else none

theorem Clone.unique (o : Clone h address es e) (c : MaskClock.Capture h)
    (other : Visible h (stores e.worker es) e s) : s = o.writer :=
  Visible.unique other o.visible (fun ev cyc => Nat.lt_irrefl _ (c.before ev ev cyc))

private abbrev Node := EndgameSearch.CacheOwner.Node

private structure Frame where
  kind : Node
  opening : Nat
  operand : Option (Event × Option Nat) := none
  issued : Bool := false

structure Call where
  activation : Event
  read : Event
  raw : Option Nat
  deriving DecidableEq

private inductive Local where
  | enter (step site : Nat) (callee : Node)
  | leave (step opening : Nat) (callee : Node)
  | load (step : Nat) (value : Option Nat)

private def sample (target : Nat) (row : Joint.Event Operand) : Option Local := do
  match row with
  | .taskWord w t (.cloneActive raw) => if target = w then some (.load t raw) else none
  | _ =>
    let s ← row.frameSample
    match s.record with
    | .localCall w ev =>
      guard (w = target)
      match ev with
      | .entry t site node .. => some (.enter t site node)
      | .leave t opening node .. => some (.leave t opening node)
      | .discarded t opening _ => some (.leave t opening .pvs)
      | .scored .. => none
    | _ => none

private def lookup (w t : Nat) (stack : List Frame) : List Local → Option Call
  | [] => none
  | row :: rest => do
    match row with
    | .enter step site .dispatch =>
      let f :: below := stack | none
      let (e,raw) ← f.operand
      guard (f.kind == .active && !f.issued && site == 0 && f.opening < e.step && e.step < step)
      if step = t then return ⟨⟨w,f.opening⟩,e,raw⟩
      lookup w t (⟨.dispatch,step,none,false⟩ :: {f with issued := true} :: below) rest
    | .enter step _ node =>
      if node == .active || !stack.isEmpty then
        lookup w t (⟨node,step,none,false⟩ :: stack) rest
      else lookup w t stack rest
    | .load step value =>
      let f :: below := stack | none
      guard (f.kind == .active && f.operand.isNone && !f.issued)
      lookup w t ({f with operand := some (⟨w,step⟩,value)} :: below) rest
    | .leave _ opening node =>
      match stack with
      | [] => lookup w t [] rest
      | f :: below =>
        guard (opening == f.opening && node == f.kind)
        lookup w t below rest

private def Local.step : Local → Nat
  | .enter t .. | .leave t .. | .load t _ => t

def call (es : Joint.Tape Operand) (entry : Event) : Option Call := do
  let thread := es.filterMap (sample entry.worker)
  guard ((thread.map Local.step).Pairwise (· < ·))
  lookup entry.worker entry.step [] thread

structure DispatchOrigin (h : History) (address : TaskRetention.Address) (es : Joint.Tape Operand) (entry : Event) extends Call where
  fromCall : call es entry = some toCall
  generation : Generation
  store : Store
  operand : Clone h address es read
  binding : operand.writer = some store
  assigned : operand.path.head? = some generation
  addressWord : raw = some generation.slot
  drop : Event
  dropped : operand.guarded.guard.release = some drop
  beforeDispatch : SequencedBefore drop entry

def dispatch (c : MaskClock.Capture h) (address : TaskRetention.Address) (es : Joint.Tape Operand)
    (entry : Event) : Option (DispatchOrigin h address es entry) := do
  match native : call es entry with
  | none => none
  | some d =>
    let o ← clone c address es d.read
    match hit : o.writer, current : o.path.head? with
    | some s, some g =>
      if actual : d.raw = some g.slot then
        guard (o.raw == d.raw)
        let finished ← o.guarded.guard.release.attach
        let drop := finished.val
        if order : drop.worker = entry.worker ∧ drop.step < entry.step then
          return ⟨d,native,g,s,o,hit,current,actual,drop,finished.property,order⟩
        none
      none
    | _,_ => none

theorem DispatchOrigin.visible (d : DispatchOrigin h address es entry) :
    Visible h (stores d.read.worker es) d.read (some d.store) := d.binding ▸ d.operand.visible

theorem DispatchOrigin.unique (d : DispatchOrigin h address es entry) (c : MaskClock.Capture h)
    (other : Visible h (stores d.read.worker es) d.read s) : s = some d.store :=
  (d.operand.unique c other).trans d.binding

private def lookupSplit (w t : Nat) (stack : List Frame) : List Local → Option Event
  | [] => none
  | row :: rest => do
    match row with
    | .enter step site .splitSearch =>
      let f :: below := stack | none
      guard (f.kind == .dispatch && !f.issued && site ≤ 2 && f.opening < step)
      if step = t then return ⟨w,f.opening⟩
      lookupSplit w t (⟨.splitSearch,step,none,false⟩ :: {f with issued := true} :: below) rest
    | .enter step site .dispatch =>
      let f :: below := stack | none
      let (e,_) ← f.operand
      guard (f.kind == .active && !f.issued && site == 0 && f.opening < e.step && e.step < step)
      lookupSplit w t (⟨.dispatch,step,none,false⟩ :: {f with issued := true} :: below) rest
    | .enter step _ node =>
      if node == .active || !stack.isEmpty then
        lookupSplit w t (⟨node,step,none,false⟩ :: stack) rest
      else lookupSplit w t stack rest
    | .load step value =>
      let f :: below := stack | none
      guard (f.kind == .active && f.operand.isNone && !f.issued)
      lookupSplit w t ({f with operand := some (⟨w,step⟩,value)} :: below) rest
    | .leave _ opening node =>
      match stack with
      | [] => lookupSplit w t [] rest
      | f :: below =>
        guard (opening == f.opening && node == f.kind)
        lookupSplit w t below rest

private theorem lookupSplit_before (w t : Nat) (stack : List Frame) (rows : List Local) (e : Event)
    (found : lookupSplit w t stack rows = some e) : SequencedBefore e ⟨w,t⟩ := by
  induction rows generalizing stack with
  | nil => contradiction
  | cons row rest ih =>
    unfold lookupSplit at found
    cases row with
    | enter step site node =>
      cases node with
      | splitSearch =>
        cases stack with
        | nil => contradiction
        | cons f below =>
          by_cases ok : (f.kind == .dispatch && !f.issued && site ≤ 2 && f.opening < step) = true
          · simp only [ok,guard,↓reduceIte,Bind.bind,Option.bind,Pure.pure] at found
            split at found
            · rename_i eq
              cases found
              simp only [Bool.and_eq_true,decide_eq_true_eq] at ok
              exact ⟨rfl,eq ▸ ok.2⟩
            · exact ih _ found
          · simp [ok,guard,failure] at found
      | dispatch =>
        cases stack with
        | nil => contradiction
        | cons f below =>
          cases hOp : f.operand with
          | none => simp [hOp] at found
          | some pair =>
            simp only [hOp,Bind.bind,Option.bind,guard,Pure.pure] at found
            split at found <;> try contradiction
            exact ih _ found
      | _ =>
        dsimp only at found
        split at found <;> exact ih _ found
    | load step value =>
      cases stack with
      | nil => contradiction
      | cons f below =>
        simp only [Bind.bind,Option.bind,guard,Pure.pure] at found
        split at found <;> try contradiction
        exact ih _ found
    | leave step opening node =>
      cases stack with
      | nil => exact ih _ found
      | cons f below =>
        simp only [Bind.bind,Option.bind,guard,Pure.pure] at found
        split at found <;> try contradiction
        exact ih _ found

/-- Locate the enclosing `dispatch_search` activation for a concrete `search_split_point`
    entry on the same worker, checking that the `active -> clone -> dispatch -> splitSearch`
    nesting is well formed. -/
def dispatchForSplit (es : Joint.Tape Operand) (splitEntry : Event) : Option Event := do
  let thread := es.filterMap (sample splitEntry.worker)
  guard ((thread.map Local.step).Pairwise (· < ·))
  lookupSplit splitEntry.worker splitEntry.step [] thread

theorem dispatchForSplit_before {es : Joint.Tape Operand} {splitEntry dispatchEntry : Event}
    (found : dispatchForSplit es splitEntry = some dispatchEntry) :
    SequencedBefore dispatchEntry splitEntry := by
  unfold dispatchForSplit at found
  simp only [Bind.bind,Option.bind,guard,Pure.pure] at found
  split at found <;> try contradiction
  exact lookupSplit_before _ _ _ _ _ found

/-- Provenance of a worker's `search_split_point` activation: its cloned active pointer
    came from a unique visible write belonging to `target`, that write holds its required
    locks and is HB-ordered against every other store to the worker's cell (including
    `ownerActive` under the split lock vs foreign `book` under the worker lock), and the
    worker state lock was released strictly before search began. -/
structure SplitDispatch (h : History) (address : TaskRetention.Address) (es : Joint.Tape Operand)
    (target : Generation) (splitEntry : Event) where
  dispatchEntry : Event
  enclosing : dispatchForSplit es splitEntry = some dispatchEntry
  beforeSplit : SequencedBefore dispatchEntry splitEntry
  origin : DispatchOrigin h address es dispatchEntry
  sameGeneration : origin.generation = target
  kindValid : origin.store.kind = .book ∨
    (origin.store.kind = .ownerActive ∧ splitEntry.worker = target.mark.worker)

theorem SplitDispatch.afterUnlock (sd : SplitDispatch h address es target splitEntry) :
    SequencedBefore sd.origin.drop splitEntry :=
  ⟨sd.origin.beforeDispatch.1.trans sd.beforeSplit.1,
   Nat.lt_trans sd.origin.beforeDispatch.2 sd.beforeSplit.2⟩

theorem SplitDispatch.visible (sd : SplitDispatch h address es target splitEntry) :
    Visible h (stores splitEntry.worker es) sd.origin.read (some sd.origin.store) := by
  have sameWorker : sd.origin.read.worker = splitEntry.worker := by
    have w1 : sd.origin.drop.worker = sd.origin.read.worker := by
      have before := sd.origin.operand.guarded.beforeDrop sd.origin.drop sd.origin.dropped
      exact before.1.symm
    exact w1.symm.trans sd.afterUnlock.1
  rw [←sameWorker]
  exact sd.origin.visible

theorem SplitDispatch.unique (sd : SplitDispatch h address es target splitEntry) (c : MaskClock.Capture h)
    (other : Visible h (stores splitEntry.worker es) sd.origin.read s) : s = some sd.origin.store :=
  Visible.unique other sd.visible (fun ev cyc => Nat.lt_irrefl _ (c.before ev ev cyc))

def splitDispatch (c : MaskClock.Capture h) (address : TaskRetention.Address) (es : Joint.Tape Operand)
    (target : Generation) (splitEntry : Event) : Option (SplitDispatch h address es target splitEntry) := do
  match found : dispatchForSplit es splitEntry with
  | none => none
  | some dispatchEntry =>
    let o ← dispatch c address es dispatchEntry
    if same : o.generation = target then
      if kind : o.store.kind = .book ∨
          (o.store.kind = .ownerActive ∧ splitEntry.worker = target.mark.worker) then
        return ⟨dispatchEntry,found,dispatchForSplit_before found,o,same,kind⟩
      else none
    else none

def UniqueSourceContract : Prop :=
  ∀ (h : History) (ws : List Store) (r : Event),
    (∀ e, ¬ HappensBefore h e e) →
    (∀ x ∈ ws, HappensBefore h x.at_ r ∨ HappensBefore h r x.at_) →
    (∀ x ∈ ws, ∀ y ∈ ws, x ≠ y → HappensBefore h x.at_ y.at_ ∨ HappensBefore h y.at_ x.at_) →
    ∃ s, Visible h ws r s ∧ ∀ t, Visible h ws r t → t = s

theorem uniqueSourceContract_proven : UniqueSourceContract := unique_source

def SplitDispatchContract : Prop :=
  ∀ {Operand : Type} (h : History) (c : MaskClock.Capture h) (address : TaskRetention.Address)
    (es : Joint.Tape Operand) (target : Generation) (splitEntry : Event)
    (sd : SplitDispatch h address es target splitEntry),
    splitDispatch c address es target splitEntry = some sd →
    sd.origin.generation = target ∧
    Visible h (stores splitEntry.worker es) sd.origin.read (some sd.origin.store) ∧
    (∀ s, Visible h (stores splitEntry.worker es) sd.origin.read s → s = some sd.origin.store) ∧
    SequencedBefore sd.origin.drop splitEntry ∧
    (sd.origin.store.kind = .book ∨
      (sd.origin.store.kind = .ownerActive ∧ splitEntry.worker = target.mark.worker)) ∧
    (∀ x ∈ stores sd.origin.read.worker es, ∀ y ∈ stores sd.origin.read.worker es,
      x ≠ y → HappensBefore h x.at_ y.at_ ∨ HappensBefore h y.at_ x.at_)

theorem splitDispatchContract_proven : SplitDispatchContract := by
  intro _ h c address es target splitEntry sd _
  exact ⟨sd.sameGeneration,sd.visible,fun _ other => sd.unique c other,sd.afterUnlock,
    sd.kindValid,sd.origin.operand.writeOrder⟩

namespace Regression

private def base : History := ⟨[],fun _ => [],[],[],fun _ _ => False⟩
private def addr : TaskRetention.Address := fun w b => some (4*w+b)
private abbrev Tape := Joint.Tape Unit

private def ptr (w t : Nat) (op : Joint.TaskWord) : Joint.Event Unit := .taskWord w t op
private def owner (baseStep bank : Nat) (parent : Option Nat) : Tape := [
  ptr 0 (baseStep+1) (.copyParent bank parent),
  .splitWord 0 (baseStep+2) (.lock 0 bank),
  .splitWord 0 (baseStep+3) (.mark bank),
  ptr 0 (baseStep+4) (.install bank parent),
  ptr 0 (baseStep+5) (.ownerActive bank)]

private def book (baseStep : Nat) : Tape := owner baseStep 0 none ++ [
  .workerLock 0 (baseStep+6) 1 true, ptr 0 (baseStep+7) (.book 1 0),
  .workerLock 0 (baseStep+8) 1 false, .splitWord 0 (baseStep+9) (.unlock 0 0)]

private def entering (w t : Nat) (n : Node) : Joint.Event Unit :=
  .engine (.localCall w (.entry t 0 n .none none) none)
private def enteringAt (w t site : Nat) (n : Node) : Joint.Event Unit :=
  .engine (.localCall w (.entry t site n .none none) none)
private def leaving (w t o : Nat) (n : Node) : Joint.Event Unit :=
  .engine (.localCall w (.leave t o n none none) none)

private def reading (k : Nat) : Tape := [entering 1 (k+10) .active, .workerLock 1 (k+11) 1 true,
  ptr 1 (k+12) (.cloneActive (some 0)), .workerLock 1 (k+13) 1 false, entering 1 (k+14) .dispatch]

private def splitReading (k site : Nat) : Tape :=
  reading k ++ [enteringAt 1 (k+15) site .splitSearch]

private def result (es : Tape) (d : Event) := do
  let ⟨_,c⟩ ← SlotSynchronization.capture base es [] []
  let o ← dispatch c addr es d
  return (o.store.at_,o.generation.mark)

private def splitResult (es : Tape) (target : Generation) (splitEntry : Event) := do
  let ⟨_,c⟩ ← SlotSynchronization.capture base es [] []
  let o ← splitDispatch c addr es target splitEntry
  return (o.origin.store.at_,o.origin.store.kind,o.origin.generation.mark,o.origin.drop)

private def read (es : Tape) (e : Event) := do
  let ⟨_,c⟩ ← SlotSynchronization.capture base es [] []
  let o ← clone c addr es e
  return (o.writer.map Store.at_,o.path.head?.map Generation.mark)

/-- The worker mutex handoff orders the foreign booking with the helper's
    clone. Prefixes can end before guard drop, but entering dispatch cannot. -/
example : result (book 0 ++ reading 0) ⟨1,14⟩ = some (⟨0,7⟩,⟨0,3⟩) := by decide +kernel
example : read (book 0 ++ (reading 0).take 3) ⟨1,12⟩ = some (some ⟨0,7⟩,some ⟨0,3⟩) := by decide +kernel
example : result (book 0 ++ (reading 0).take 3 ++ [entering 1 14 .dispatch]) ⟨1,14⟩ = none := by decide +kernel

private def dispose : Tape := [leaving 1 15 14 .dispatch, leaving 1 16 10 .active,
  .workerLock 1 17 1 true, ptr 1 18 (.detach 0), .workerLock 1 19 1 false,
  .workerLock 0 20 0 true, ptr 0 21 (.detach 0), .workerLock 0 22 0 false,
  .workerLock 0 23 0 true, ptr 0 24 (.restore 0 none), .workerLock 0 25 0 false, ptr 0 26 (.clearTask 0)]

/-- Two generations use slot zero. Both reads see the same pointer word.
    The second load has the later booking and the later split mark. -/
private def reuse := book 0 ++ reading 0 ++ dispose ++ book 30 ++ reading 30
#guard result reuse ⟨1,14⟩ == some (⟨0,7⟩,⟨0,3⟩)
#guard result reuse ⟨1,44⟩ == some (⟨0,37⟩,⟨0,33⟩)

private def splitDispose : Tape := [leaving 1 16 15 .splitSearch, leaving 1 17 14 .dispatch, leaving 1 18 10 .active,
  .workerLock 1 19 1 true, ptr 1 20 (.detach 0), .workerLock 1 21 1 false,
  .workerLock 0 22 0 true, ptr 0 23 (.detach 0), .workerLock 0 24 0 false,
  .workerLock 0 25 0 true, ptr 0 26 (.restore 0 none), .workerLock 0 27 0 false, ptr 0 28 (.clearTask 0)]

/-- Both root (site 0) and nested (site 1/2) split searches resolve to their exact generation
    mark. Claiming the old generation at the same reused address is rejected. -/
private def splitReuse (site : Nat) := book 0 ++ splitReading 0 site ++ splitDispose ++ book 30 ++ splitReading 30 site
#guard splitResult (splitReuse 0) ⟨0,⟨0,33⟩⟩ ⟨1,45⟩ == some (⟨0,37⟩,.book,⟨0,33⟩,⟨1,43⟩)
#guard splitResult (splitReuse 1) ⟨0,⟨0,33⟩⟩ ⟨1,45⟩ == some (⟨0,37⟩,.book,⟨0,33⟩,⟨1,43⟩)
#guard splitResult (splitReuse 2) ⟨0,⟨0,33⟩⟩ ⟨1,45⟩ == some (⟨0,37⟩,.book,⟨0,33⟩,⟨1,43⟩)
#guard splitResult (splitReuse 0) ⟨0,⟨0,3⟩⟩ ⟨1,45⟩ == none
#guard splitResult (book 0 ++ (reading 0).take 3 ++ [entering 1 14 .dispatch, enteringAt 1 15 0 .splitSearch])
  ⟨0,⟨0,3⟩⟩ ⟨1,15⟩ == none

/-- Wrong worker lock on booking (lock target 2 instead of helper 1) is rejected. -/
private def wrongLockBook : Tape := owner 0 0 none ++ [
  .workerLock 0 6 2 true, ptr 0 7 (.book 1 0),
  .workerLock 0 8 2 false, .splitWord 0 9 (.unlock 0 0)] ++ splitReading 0 0
#guard splitResult wrongLockBook ⟨0,⟨0,3⟩⟩ ⟨1,15⟩ == none

/-- OwnerActive takes only the split lock while a foreign book (helpful owner) takes
    worker 0's state lock. When synchronized via worker 0's detach/unlock -> worker 1's
    lock/book, both dispatches pass. Without that handoff, the race is rejected. -/
private def ownerThenForeign (synchronized : Bool) : Tape :=
  owner 0 0 none ++ [.splitWord 0 6 (.unlock 0 0),
   entering 0 7 .active, .workerLock 0 8 0 true, ptr 0 9 (.cloneActive (some 0)), .workerLock 0 10 0 false,
   entering 0 11 .dispatch, enteringAt 0 12 0 .splitSearch,
   leaving 0 13 12 .splitSearch, leaving 0 14 11 .dispatch, leaving 0 15 7 .active] ++
  (if synchronized then [.workerLock 0 16 0 true, ptr 0 17 (.detach 0), .workerLock 0 18 0 false] else []) ++
  [ptr 1 20 (.copyParent 4 none), .splitWord 1 21 (.lock 1 0), .splitWord 1 22 (.mark 0),
   ptr 1 23 (.install 4 none), ptr 1 24 (.ownerActive 4),
   .workerLock 1 25 0 true, ptr 1 26 (.book 0 4), .workerLock 1 27 0 false, .splitWord 1 28 (.unlock 1 0),
   entering 0 30 .active, .workerLock 0 31 0 true, ptr 0 32 (.cloneActive (some 4)), .workerLock 0 33 0 false,
   entering 0 34 .dispatch, enteringAt 0 35 1 .splitSearch]

#guard splitResult (ownerThenForeign true) ⟨0,⟨0,3⟩⟩ ⟨0,12⟩ == some (⟨0,5⟩,.ownerActive,⟨0,3⟩,⟨0,10⟩)
#guard splitResult (ownerThenForeign true) ⟨4,⟨1,22⟩⟩ ⟨0,35⟩ == some (⟨1,26⟩,.book,⟨1,22⟩,⟨0,33⟩)
#guard splitResult (ownerThenForeign false) ⟨0,⟨0,3⟩⟩ ⟨0,12⟩ == none
#guard splitResult (ownerThenForeign false) ⟨4,⟨1,22⟩⟩ ⟨0,35⟩ == none

private def restored : Tape := owner 0 0 none ++ [.splitWord 0 6 (.unlock 0 0)] ++
  owner 10 1 (some 0) ++ [.splitWord 0 16 (.unlock 0 1),
  .workerLock 0 17 0 true, ptr 0 18 (.detach 1), .workerLock 0 19 0 false,
  .workerLock 0 20 0 true, ptr 0 21 (.restore 1 (some 0)), .workerLock 0 22 0 false,
  ptr 0 23 (.clearTask 1), .workerLock 0 24 0 true, ptr 0 25 (.cloneActive (some 0))]

/-- Reinstalling the original address supplies the new load, even though the
    first ownerActive already wrote that word under the split lock alone. -/
example : read restored ⟨0,25⟩ = some (some ⟨0,21⟩,some ⟨0,3⟩) := by decide +kernel

example : read [.workerLock 0 1 0 true,ptr 0 2 .detachEmpty,ptr 0 3 (.cloneActive none)] ⟨0,3⟩ =
    some (some ⟨0,2⟩,none) := by decide +kernel
example : read [.workerLock 0 1 0 true,ptr 0 3 (.cloneActive none)] ⟨0,3⟩ =
    some (none,none) := by decide +kernel
example : read [.workerLock 1 1 0 true,ptr 0 3 (.cloneActive none)] ⟨0,3⟩ = none := by decide +kernel

end Regression

end NeuralReversi.PvsParallel.Interleaving.ActivePointer

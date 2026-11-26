import NeuralReversi.PvsParallel.Interleaving.Joint
import NeuralReversi.PvsParallel.Interleaving.StopSchedule

namespace NeuralReversi.PvsParallel.Interleaving.TaskAncestry

open Threading.AtomicWord (Event)

structure Generation where
  slot : Nat
  mark : Event
  deriving DecidableEq

abbrev Path := List Generation

private def head (p : Path) := p.head?.map Generation.slot

structure Task where
  gen : Generation
  parents : Path
  deriving DecidableEq

abbrev Slots := Nat → Option Task

private def linked (slots : Slots) : Path → Bool
  | [] => true
  | g :: tail => slots g.slot == some ⟨g,tail⟩ && linked slots tail

private def parent (slots : Slots) (slot : Nat) := (slots slot).bind fun task => head task.parents

private theorem walk (slots : Slots) (p : Path) (ok : linked slots p = true) :
    StopSchedule.Chain (parent slots) (head p) (p.map Generation.slot) := by
  induction p with
  | nil => exact .root
  | cons g tail ih =>
    obtain ⟨field,rest⟩ := Bool.and_eq_true_iff.mp ok
    have stored := beq_iff_eq.mp field
    apply StopSchedule.Chain.active
    simpa [parent,stored] using ih rest

structure View where
  position : Nat
  slots : Slots
  path : Path
  reset : Option Event

structure Read where
  field : Generation
  word : Bool
  source : Event
  view : View

structure Restore where
  position : Nat
  task : Task
  at_ : Event

structure Walk where
  start : Event
  view : View
  remaining : Path
  found : Bool
  reverseReads : List Event

structure Journal where
  polls : Event → Option View
  after : Event → Option (Restore × View)
  loads : Event → Option Read
  stores : Event → Option (Generation × Bool)
  walks : Event → Option Walk
  clones : Event → Option View
  assignments : Event → Option Path

private def put [DecidableEq α] (f : α → β) (where_ : α) (value : β) (key : α) :=
  if key = where_ then value else f key

private structure Building where
  slot : Nat
  parents : Path
  mark : Option Event

private structure State where
  position : Nat
  slots : Slots
  active : Nat → Path
  building : Nat → Option Building
  restored : Nat → Option Restore
  resets : Nat → Option Event
  previous : Nat → Option Nat
  scanning : Nat → Option Walk
  journal : Journal

private def empty : State :=
  ⟨0, fun _ => none, fun _ => [], fun _ => none, fun _ => none, fun _ => none, fun _ => none, fun _ => none,
   ⟨fun _ => none,fun _ => none,fun _ => none,fun _ => none,fun _ => none,fun _ => none,fun _ => none⟩⟩

private def State.view (s : State) (w : Nat) : Option View := do
  guard (linked s.slots (s.active w))
  return ⟨s.position,s.slots,s.active w,s.resets w⟩

private def State.advance (s : State) (e : Event) : Option State := do
  guard ((s.previous e.worker).all (· < e.step))
  return {s with position := s.position+1, previous := put s.previous e.worker (some e.step)}

private def mark (s : State) (e : Event) (addr : Nat) : Option State := do
  let b ← s.building e.worker
  guard (s.slots addr |>.isNone)
  guard (b.slot = addr ∧ b.mark = none)
  return {s with building := put s.building e.worker (some {b with mark := some e})}

private def taskWord (s : State) (e : Event) : Joint.TaskWord → Option State
  | .cloneActive loaded => do
    let view ← s.view e.worker
    guard (loaded = head view.path)
    return {s with journal.clones := put s.journal.clones e (some view)}
  | .copyParent slot loaded => do
    let view ← s.view e.worker
    guard (loaded = head view.path ∧ s.building e.worker = none)
    guard (!(view.path.map Generation.slot).contains slot)
    return {s with building := put s.building e.worker (some ⟨slot,view.path,none⟩),
                   restored := put s.restored e.worker none}
  | .install slot stored => do
    let b ← s.building e.worker
    let atMark ← b.mark
    guard (slot = b.slot ∧ stored = head b.parents ∧ b.parents = s.active e.worker)
    guard (linked s.slots b.parents && (s.slots slot).isNone)
    let task := ⟨⟨slot,atMark⟩,b.parents⟩
    return {s with slots := put s.slots slot (some task), building := put s.building e.worker none}
  | .ownerActive slot => do
    let task ← s.slots slot
    guard (e.worker = task.gen.mark.worker ∧ s.active e.worker = task.parents)
    let path := task.gen :: task.parents
    return {s with active := put s.active e.worker path,
                   journal.assignments := put s.journal.assignments e (some path)}
  | .book helper slot => do
    let task ← s.slots slot
    guard ((s.active helper).isEmpty && linked s.slots (task.gen :: task.parents))
    let path := task.gen :: task.parents
    return {s with active := put s.active helper path, restored := put s.restored helper none,
                   journal.assignments := put s.journal.assignments e (some path)}
  | .detach old => do
    let view ← s.view e.worker
    guard (head view.path = some old)
    return {s with active := put s.active e.worker [], restored := put s.restored e.worker none,
                   journal.assignments := put s.journal.assignments e (some [])}
  | .detachEmpty => do
    guard ((s.active e.worker).isEmpty)
    return {s with restored := put s.restored e.worker none, journal.assignments := put s.journal.assignments e (some [])}
  | .restore slot loaded => do
    let task ← s.slots slot
    guard (task.gen.mark.worker = e.worker ∧ loaded = head task.parents)
    guard ((s.active e.worker).isEmpty && linked s.slots (task.gen :: task.parents))
    return {s with active := put s.active e.worker task.parents,
                   restored := put s.restored e.worker (some ⟨s.position,task,e⟩),
                   journal.assignments := put s.journal.assignments e (some task.parents)}
  | .clearTask slot => do
    let last ← s.restored e.worker
    guard (last.task.gen.slot = slot ∧ s.slots slot = some last.task)
    return {s with slots := put s.slots slot none}
  | .latch _ | .epoch .. => some s
  | .resetCache .. => return {s with resets := put s.resets e.worker (some e)}
  | .poll | .checkStop _ => do
    let view ← s.view e.worker
    return {s with journal.polls := put s.journal.polls e (some view)}
  | .afterSplit loaded => do
    let last ← s.restored e.worker
    let view ← s.view e.worker
    guard ((s.scanning e.worker).isNone)
    guard (head view.path = loaded ∧ view.path = last.task.parents ∧ s.slots last.task.gen.slot = none)
    return {s with journal.after := put s.journal.after e (some (last,view)),
                   scanning := put s.scanning e.worker (some ⟨e,view,view.path,false,[]⟩),
                   restored := put s.restored e.worker none}
  | .beginWalk loaded => do
    let view ← s.view e.worker
    guard ((s.scanning e.worker).isNone && head view.path == loaded)
    return {s with scanning := put s.scanning e.worker (some ⟨e,view,view.path,false,[]⟩)}
  | .endWalk stopped => do
    let scan ← s.scanning e.worker
    guard (scan.remaining.isEmpty && scan.found == stopped)
    return {s with scanning := put s.scanning e.worker none,
                   journal.walks := put s.journal.walks scan.start (some scan)}
  | .storeFlag slot word => do
    let task ← s.slots slot
    let actor ← s.view e.worker
    if word then guard (actor.path = task.gen :: task.parents)
    else guard (e.worker = task.gen.mark.worker ∧ actor.path = task.parents)
    return {s with journal.stores := put s.journal.stores e (some (task.gen,word))}
  | .loadFlag slot word src => do
    let view ← s.view e.worker
    let scan ← s.scanning e.worker
    let g :: tail := scan.remaining | none
    guard (g.slot = slot ∧ view.path = scan.view.path ∧ view.reset = scan.view.reset)
    let next := {scan with remaining := if word then [] else tail, found := word, reverseReads := e :: scan.reverseReads}
    return {s with journal.loads := put s.journal.loads e (some ⟨g,word,src,view⟩),
                   scanning := put s.scanning e.worker (some next)}

private def step (slot : Nat → Nat → Option Nat) (s : State) : Joint.Event Op → Option State
  | .splitWord w t (.mark bank) => do
    let addr ← slot w bank
    let e := ⟨w,t⟩
    let s ← s.advance e
    mark s e addr
  | .taskWord w t op => do
    let e := ⟨w,t⟩
    let s ← s.advance e
    taskWord s e op
  | _ => some s

/-- The task's parent comes from the active pointer copied before acquiring the
    initialization lock. Its generation is identified by the shared split mark. -/
def capture (slot : Nat → Nat → Option Nat) (es : Joint.Tape Op) : Option Journal :=
  (es.foldlM (step slot) empty).map State.journal

structure Window (g : Generation) where
  view : View
  owner : View
  parents : Path
  path : view.path = g :: parents
  current : linked view.slots (g :: parents) = true

/-- The helper was assigned to this generation and the owner restored its
    identical parent generations before starting its uncached post-join scan. -/
def window (j : Journal) (g : Generation) (poll after : Event) : Option (Window g) := do
  let view ← j.polls poll
  let (closed,owner) ← j.after after
  let _ ← j.walks after
  guard (closed.task.gen = g ∧ after.worker = g.mark.worker ∧ owner.path = closed.task.parents)
  guard (view.position < closed.position ∧ closed.position < owner.position ∧ closed.at_.worker = after.worker)
  let parents := owner.path
  if shape : view.path = g :: parents then
    if live : linked view.slots (g :: parents) = true then return ⟨view,owner,parents,shape,live⟩
  none

def Window.ancestry (w : Window g) : StopSchedule.Ancestry g.slot (w.parents.map Generation.slot) := by
  let links := parent w.view.slots
  have assigned := walk w.view.slots (g :: w.parents) w.current
  have field := beq_iff_eq.mp (Bool.and_eq_true_iff.mp w.current).1
  have restored := walk w.view.slots w.parents (Bool.and_eq_true_iff.mp w.current).2
  exact ⟨links,(g :: w.parents).map Generation.slot,assigned,by simpa [links,parent,field] using restored⟩

open Threading.CutoffVisibility (FlagRead History)

structure Source (j : Journal) (v : View) (r : FlagRead flags) where
  load : Read
  generation : Generation
  readAt : j.loads r.event = some load
  assignment : load.view.path = v.path ∧ load.view.reset = v.reset
  field : load.field = generation ∧ generation.slot = r.slot ∧ generation ∈ v.path
  register : load.source = ((flags r.slot).get r.source).event ∧ load.word = r.value
  storeAt : j.stores load.source = some (generation,r.value)

/-- Resolve a recorded flag read against its exact write instruction. The two
    lookups inspect the complete journal so a relaxed source may appear later
    than the reader in a linear extension. -/
def source (j : Journal) (v : View) (r : FlagRead flags) : Option (Source j v r) := do
  match atLoad : j.loads r.event with
  | none => none
  | some raw =>
    let g := raw.field
    if same : raw.view.path = v.path ∧ raw.view.reset = v.reset then
      if member : g.slot = r.slot ∧ g ∈ v.path then
        if reg : raw.source = ((flags r.slot).get r.source).event ∧ raw.word = r.value then
          if wrote : j.stores raw.source = some (g,r.value) then
            return ⟨raw,g,atLoad,same,⟨rfl,member⟩,reg,wrote⟩
    none

def checkedScan (j : Journal) (v : View) (scan : Threading.SplitReturn.Observation.AncestorScan h) : Bool :=
  match j.walks scan.event with
  | some walk => walk.view.path == v.path && walk.view.reset == v.reset &&
    walk.reverseReads.reverse == scan.reads.map (·.event) && walk.found == scan.stopped &&
    scan.reads.all (fun r => (source j v r).isSome)
  | none => false

def samples (j : Journal) (g : Generation) (m : NativeStopReturns.Memory)
    (check : Threading.SplitReturn.Observation.OwnerCheck m.history) (wait : Event) (ps : List Tickets.Post) (poll : Event)
    (operands : (chain : List Nat) → Option (StopSchedule.PollOperands m check wait g.slot ps poll chain)) :
    Option (StopSchedule.Samples m check wait g.slot ps poll) := do
  let w ← window j g poll check.ancestors.event
  if shape : w.parents.map Generation.slot = check.ancestors.chain then
    guard (checkedScan j w.owner check.ancestors)
    let lineage : StopSchedule.Ancestry g.slot check.ancestors.chain := shape ▸ w.ancestry
    let args ← operands lineage.cachedChain
    guard (args.reads.all (fun r => (source j w.view r).isSome))
    return ⟨lineage,args⟩
  else none

end NeuralReversi.PvsParallel.Interleaving.TaskAncestry

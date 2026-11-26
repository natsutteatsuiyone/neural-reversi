import NeuralReversi.PvsParallel.Interleaving.MaskGeneration
import NeuralReversi.PvsParallel.SplitSchedulerMir.Mutex

namespace NeuralReversi.PvsParallel.Interleaving.SlotSynchronization

open Threading
open AtomicWord (Event)
open SpinLock (Operation)
open SplitCompletion (Worker)
open CutoffVisibility (History)

def operations (o : Worker) (bank : Nat) (es : Joint.Tape Operand) : List Operation :=
  es.filterMap fun row => match row with
    | .splitWord w t (.lock p b) => if p = o.val ∧ b = bank then some (.acquire ⟨w,t⟩) else none
    | .splitWord w t (.unlock p b) => if p = o.val ∧ b = bank then some (.release ⟨w,t⟩) else none
    | _ => none

private def handoff (ops : List Operation) (n : Nat) : Option (Event × Event) := do
  let .release r ← ops[n]? | none
  let .acquire a ← ops[n+1]? | none
  return (r,a)

def lockEdges (ops : List Operation) := (List.range ops.length).filterMap (handoff ops)

private theorem atRank : handoff ops n = some (r,a) ↔
    ops[n]? = some (.release r) ∧ ops[n+1]? = some (.acquire a) := by
  simp [handoff,Option.bind_eq_some_iff]
  constructor
  · rintro ⟨x,hx,h⟩
    cases x with
    | acquire => contradiction
    | release u =>
      obtain ⟨y,hy,h⟩ := Option.bind_eq_some_iff.mp h
      cases y with
      | release => contradiction
      | acquire v => cases h; exact ⟨hx,hy⟩
  · rintro ⟨hu,hv⟩
    exact ⟨_,hu,by simp [hv]⟩

def LockContract := ∀ (ops : List Operation) (r a : Event), (r,a) ∈ lockEdges ops ↔
  ∃ n, ops[n]? = some (.release r) ∧ ops[n+1]? = some (.acquire a)

theorem adjacent : LockContract := by
  intro ops r a
  constructor
  · intro mem
    obtain ⟨n,_,ok⟩ := List.mem_filterMap.mp mem
    exact ⟨n,atRank.mp ok⟩
  · rintro ⟨n,first,next⟩
    have bound : n < ops.length := List.getElem?_eq_some_iff.mp first |>.choose
    exact List.mem_filterMap.mpr ⟨n,List.mem_range.mpr bound,atRank.mpr ⟨first,next⟩⟩

/-- Handoffs of this slot's lock and synchronization caused by loads of its
    participation mask. Independent locks can interleave in the original tape. -/
def edges (o : Worker) (bank : Nat) (es : Joint.Tape Operand) :=
  lockEdges (operations o bank es) ++ MaskGeneration.releaseEdges o (MaskGeneration.rows o bank es)

private def address : Joint.Event Operand → Option (Nat × Nat)
  | .splitWord _ _ (.lock p b) | .splitWord _ _ (.unlock p b) | .splitWord _ _ (.maskRmw p b ..) => some (p,b)
  | .splitWord p _ (.mark b) | .maskWord p _ (.load b ..) => some (p,b)
  | _ => none

def workerOps (target : Nat) (es : Joint.Tape Operand) : List Operation :=
  es.filterMap fun row => match row with
    | .workerLock w t p acquired =>
        if p = target then some (if acquired then .acquire ⟨w,t⟩ else .release ⟨w,t⟩) else none
    | _ => none

def workers (es : Joint.Tape Operand) : List Nat :=
  (es.filterMap fun row => match row with | .workerLock _ _ p _ => some p | _ => none).eraseDups

def recorded (es : Joint.Tape Operand) : Option (List (Event × Event)) := do
  let slots := (es.filterMap address).eraseDups
  let groups ← slots.mapM fun (p,b) => if ok : p < 64 then some (edges ⟨p,ok⟩ b es) else none
  let stateLocks := workers es
  guard (stateLocks.all (· < 64))
  return groups.flatten ++ stateLocks.flatMap (fun target => lockEdges (workerOps target es))

/-- Unrelated synchronization objects contribute their own transfers. Split
    and worker-state mutexes are projected here. Epoch reads belong to base. -/
def history (base : History) (transfers : List (Event × Event)) : History :=
  {base with otherSync := fun a b => (a,b) ∈ transfers}

def capture (base : History) (es : Joint.Tape Operand) (other : List (Event × Event)) (order : List Event) :
    Option ((transfers : List (Event × Event)) × MaskClock.Capture (history base transfers)) := do
  let transfers := (← recorded es) ++ other
  let s : MaskClock.Sync (history base transfers) := ⟨transfers,fun _ _ => Iff.rfl⟩
  return ⟨transfers,← MaskClock.capture s order⟩

def mutex (h : History) (init : Event) (o : Worker) (bank : Nat) (es : Joint.Tape Operand)
    (includes : ∀ r a, (r,a) ∈ edges o bank es → h.otherSync r a) : SplitSchedulerMir.Mutex.Capture h where
  init := init
  ops := operations o bank es
  sync n r a unlocked acquired := includes r a (List.mem_append_left _ ((adjacent ..).mpr ⟨n,unlocked,acquired⟩))

end NeuralReversi.PvsParallel.Interleaving.SlotSynchronization

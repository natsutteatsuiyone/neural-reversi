module
public import NeuralReversi.EndgameSearch.CacheHistory
public import NeuralReversi.EndgameSearch.CacheHistory.NativeTrace.Sites

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheHistory.NativeTrace

open CacheOwner (Descends Resource EntryPoint Node Edge)

structure Worker (p : Pool) where
  ordinal : Nat
  kind : EntryPoint
  started : (CacheOwner.poolPlan p.size)[ordinal]? = some kind

variable {p : Pool} {w : Worker p}

def Worker.root (w : Worker p) : Resource :=
  .thread (CacheOwner.start (CacheOwner.programFor w.kind) (p.base+3*w.ordinal) p.foreign).running

structure Operands where
  frame : Nat
  input : Input

inductive Raw where
  | enter (step site : Nat) (callee : Node) (received : Resource) (operands : Option Operands)
  | leave (step opening : Nat) (callee : Node) (scoredAt : Option Nat)

def Raw.step : Raw → Nat
  | .enter t .. | .leave t .. => t

structure Activation (w : Worker p) where
  first : Nat
  node : Node
  received : Resource
  path : Descends p.outside (CacheOwner.entry w.kind) w.root node received
  operands : Option Operands

structure Cursor (w : Worker p) where
  clock : Nat
  stack : List (Activation w)
  finished : List (Visit p)
  boundarySites : List (Nat × Node)

def Cursor.initial (w : Worker p) : Cursor w := ⟨0,[],[],[]⟩

def Cursor.parent (s : Cursor w) : Activation w :=
  s.stack.head?.getD ⟨0,CacheOwner.entry w.kind,w.root,.same,none⟩

def Cursor.busy (s : Cursor w) : Bool := s.stack.any (fun f => f.node == .boundary)

def Cursor.enter (s : Cursor w) (step site : Nat) (callee : Node) (received : Resource) (args : Option Operands) :
    Option (Cursor w) := do
  let parent := s.parent
  let found := CacheOwner.transfers.find? (fun e => e.caller == parent.node && e.site == site)
  match h : found with
  | none => none
  | some edge =>
    let member := List.mem_of_find?_eq_some h
    if caller : edge.caller = parent.node then
      if target : edge.callee = callee then
        if actual : edge.flow.apply p.outside parent.received = some received then
          if args.isSome = (callee == .boundary) ∧ (callee = .boundary → s.busy = false) then
            let previous := caller.symm ▸ parent.path
            let routed := Descends.call previous member actual
            let pushed : Activation w := ⟨step,callee,received,target ▸ routed,args⟩
            let sites := if callee = .boundary then [] else
              if parent.node = .boundary then (site,callee) :: s.boundarySites else s.boundarySites
            some {s with clock := step+1, stack := pushed :: s.stack, boundarySites := sites}
          else none
        else none
      else none
    else none

def Cursor.leave (s : Cursor w) (step opening : Nat) (callee : Node) (score : Option Nat) : Option (Cursor w) := do
  let top :: rest := s.stack | none
  if top.first ≠ opening ∨ top.node ≠ callee then none else
    match top.operands with
    | none =>
      if score.isSome then none else some {s with clock := step+1, stack := rest}
    | some op =>
      if atBoundary : top.node = .boundary then
        if ordered : top.first ≤ step then
          if stamp : (score.map (fun k => decide (step ≤ k))).getD true then
            have after : ∀ k, score = some k → step ≤ k := by
              intro k h
              simpa [h] using stamp
            let event : Visit p := ⟨w.ordinal,w.kind,w.started,top.received,atBoundary ▸ top.path,
              op.frame,top.first,step,ordered,score,after,op.input⟩
            if expectedSites op.input.arm = some s.boundarySites.reverse then
              some ⟨step+1,rest,event :: s.finished,[]⟩
            else none
          else none
        else none
      else none

def Cursor.advance (s : Cursor w) (event : Raw) : Option (Cursor w) :=
  if s.clock ≤ event.step then
    match event with
    | .enter t site callee receiver args => s.enter t site callee receiver args
    | .leave t begin callee score => s.leave t begin callee score
  else none

inductive Event (Shared : Type) where
  | worker (ordinal : Nat) (event : Raw)
  | shared (event : Shared)

def workerEvents (ordinal : Nat) (stream : List (Event S)) : List Raw :=
  stream.filterMap fun | .worker n ev => if n=ordinal then some ev else none | .shared _ => none

def sharedEvents (stream : List (Event S)) : List S :=
  stream.filterMap fun | .shared e => some e | .worker .. => none

def run (w : Worker p) (stream : List (Event S)) : Option (Cursor w) :=
  (workerEvents w.ordinal stream).foldlM Cursor.advance (.initial w)

theorem backwards_time (s : Cursor w) (e : Raw) (old : e.step < s.clock) : s.advance e = none := by
  simp [Cursor.advance,Nat.not_le_of_lt old]

theorem checked_entry (s out : Cursor w)
    (accepted : s.enter step site callee received args = some out) :
    ∃ top, out.stack = top :: s.stack ∧ out.finished = s.finished ∧
      top.first = step ∧ top.node = callee ∧ top.received = received ∧ top.operands = args ∧
      (args.isSome = (callee == .boundary)) ∧ (callee = .boundary → s.busy = false) := by
  unfold Cursor.enter at accepted
  dsimp only at accepted
  split at accepted <;> try contradiction
  split at accepted <;> try contradiction
  split at accepted <;> try contradiction
  split at accepted <;> try contradiction
  split at accepted <;> try contradiction
  rename_i metadata
  cases accepted
  exact ⟨_,rfl,rfl,rfl,rfl,rfl,rfl,metadata⟩

theorem boundary_reserves (s out : Cursor w) (accepted : s.enter step site .boundary received args = some out) :
    args.isSome = true ∧ s.busy = false ∧ out.busy = true := by
  obtain ⟨top,stack,_,_,kind,_,_,required,noOther⟩ := checked_entry s out accepted
  exact ⟨required,noOther rfl,by simp [Cursor.busy,stack,kind]⟩

theorem wrong_return (s : Cursor w) (top : Activation w) (stack : s.stack = top :: rest)
    (different : top.first ≠ opening ∨ top.node ≠ callee) :
    s.leave step opening callee score = none := by
  simp [Cursor.leave,stack,different]

theorem matched_boundary (s out : Cursor w) (top : Activation w) (stack : s.stack = top :: rest)
    (metadata : top.operands = some args) (accepted : s.leave step opening callee score = some out) :
    ∃ v, out.finished = v :: s.finished ∧ out.stack = rest ∧ v.worker = w.ordinal ∧
      v.first = opening ∧ v.last = step ∧ v.frame = args.frame ∧ v.input = args.input ∧ v.scoredAt = score ∧
      expectedSites args.input.arm = some s.boundarySites.reverse := by
  unfold Cursor.leave at accepted
  simp only [stack,metadata] at accepted
  split at accepted
  · contradiction
  · rename_i valid
    split at accepted <;> try contradiction
    split at accepted <;> try contradiction
    split at accepted <;> try contradiction
    split at accepted <;> try contradiction
    rename_i observed
    cases accepted
    have same : top.first = opening := by simpa [not_or] using (not_or.mp valid).1
    exact ⟨_,rfl,rfl,rfl,same,rfl,rfl,rfl,rfl,observed⟩

theorem propagation (s out : Cursor w) (top : Activation w) (stack : s.stack = top :: rest)
    (other : top.operands = none) (accepted : s.leave step opening callee score = some out) :
    out.finished = s.finished ∧ out.stack = rest := by
  unfold Cursor.leave at accepted
  simp only [stack,other] at accepted
  split at accepted <;> try contradiction
  split at accepted <;> try contradiction
  cases accepted
  exact ⟨rfl,rfl⟩

theorem other_worker (w : Worker p) (stream : List (Event S)) (e : Raw) (another : n ≠ w.ordinal) :
    run w (stream ++ [.worker n e]) = run w stream := by
  simp [run,workerEvents,another]

theorem shared_event (w : Worker p) (stream : List (Event S)) (e : S) :
    run w (stream ++ [.shared e]) = run w stream ∧ sharedEvents (stream ++ [.shared e]) = sharedEvents stream ++ [e] := by
  simp [run,workerEvents,sharedEvents]

theorem worker_keeps_shared (stream : List (Event S)) (n : Nat) (e : Raw) :
    sharedEvents (stream ++ [.worker n e]) = sharedEvents stream := by simp [sharedEvents]

/-- A finite suffix from other native stacks, including unfinished calls, leaves this worker's
    entire cursor unchanged. All shared publications from the suffix remain in program order. -/
theorem foreign_prefix (w : Worker p) (stream suffix : List (Event S))
    (other : ∀ n e, Event.worker n e ∈ suffix → n ≠ w.ordinal) :
    run w (stream ++ suffix) = run w stream ∧
      sharedEvents (stream ++ suffix) = sharedEvents stream ++ sharedEvents suffix := by
  have absent : workerEvents w.ordinal suffix = [] := by
    induction suffix with
    | nil => rfl
    | cons ev rest ih =>
      have more := ih (fun n e found => other n e (List.mem_cons_of_mem _ found))
      cases ev with
      | shared => exact more
      | worker n e =>
        have different := other n e (by simp)
        simpa [workerEvents,different] using more
  have projected : workerEvents w.ordinal (stream ++ suffix) = workerEvents w.ordinal stream ++ workerEvents w.ordinal suffix :=
    List.filterMap_append
  exact ⟨by simp [run,projected,absent], List.filterMap_append⟩

def Cursor.completed (s : Cursor w) : List (Visit p) := s.finished.reverse

instance {p : Pool} : DecidableRel (Earlier (p:=p)) := fun a b => inferInstanceAs (Decidable (a.worker = b.worker → a.last < b.first))

/-- Unfinished boundary activations remain on the stack. The retained timeline contains all
    matching returns already seen, independently of publication or cancellation. -/
def Cursor.timeline (s : Cursor w) : Option (Timeline p) :=
  if sorted : s.completed.Pairwise Earlier then some ⟨s.completed,sorted⟩ else none

def Cursor.visible (s : Cursor w) (mc : Bitboard.Moves.X86Config) (fc : FlipConfig) : Option Local := do
  if s.busy then none else
    let t ← s.timeline
    return replay mc fc p t.visits

theorem pending_private (s : Cursor w) (openCall : s.busy = true) :
    s.visible mc fc = none := by simp [Cursor.visible,openCall]

theorem closed_sound (s : Cursor w) (atRest : s.visible mc fc = some state) :
    Good mc fc p.width p.sizes p.key state := by
  unfold Cursor.visible at atRest
  split at atRest
  · contradiction
  · cases h : s.timeline with
    | none => simp [h] at atRest
    | some t =>
      simp [h] at atRest
      exact atRest ▸ replay_good mc fc p t.visits

def workerAt (p : Pool) (ordinal : Nat) : Option (Worker p) :=
  match h : (CacheOwner.poolPlan p.size)[ordinal]? with
  | some kind => some ⟨ordinal,kind,h⟩
  | none => none

def readWorker (p : Pool) (stream : List (Event S)) (ordinal : Nat) : Option ((w : Worker p) × Cursor w) := do
  let w ← workerAt p ordinal
  let parsed ← run w stream
  return ⟨w,parsed⟩

structure Capture (p : Pool) (S : Type) where
  workers : List ((w : Worker p) × Cursor w)
  timeline : Timeline p
  shared : List S

def decode (p : Pool) (stream : List (Event S)) : Option (Capture p S) := do
  let count := (CacheOwner.poolPlan p.size).length
  if !(stream.all fun | .worker n _ => decide (n < count) | .shared _ => true) then none else
    let workers ← (List.range count).mapM (readWorker p stream)
    let visits := workers.flatMap (fun w => w.2.completed)
    if ord : visits.Pairwise Earlier then
      some ⟨workers,⟨visits,ord⟩,sharedEvents stream⟩
    else none

theorem decode_keeps (p : Pool) (stream : List (Event S)) {capture : Capture p S}
    (success : decode p stream = some capture) :
    capture.shared = sharedEvents stream ∧
      capture.timeline.visits = capture.workers.flatMap (fun w => w.2.completed) ∧
      (List.range (CacheOwner.poolPlan p.size).length).mapM (readWorker p stream) = some capture.workers := by
  unfold decode at success
  dsimp only at success
  split at success
  · contradiction
  · cases all : (List.range (CacheOwner.poolPlan p.size).length).mapM (readWorker p stream) with
    | none => simp [all] at success
    | some workers =>
      simp [all] at success
      obtain ⟨_,rfl⟩ := success
      exact ⟨rfl,rfl,rfl⟩

end NeuralReversi.EndgameSearch.CacheHistory.NativeTrace

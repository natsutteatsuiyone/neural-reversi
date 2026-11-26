import NeuralReversi.PvsParallel.Interleaving.AlphaOrigin
import NeuralReversi.PvsParallel.Interleaving.SlotSynchronization
import NeuralReversi.PvsParallel.Interleaving.SplitOwnerReadback

namespace NeuralReversi.PvsParallel.Interleaving.AlphaReplay

open Threading
open AtomicWord (Event Write)
open SplitSchedulerMir
open SlotFrameMir.RawProjection (subsequent)
open AlphaOrigin (OwnerOperands)

abbrev Slot := SlotFrameMir.Slot

def accesses (slot : Slot) (es : Joint.Tape Operand) : List (Event × Joint.AlphaWord) :=
  es.filterMap fun row => match row with
    | .alphaWord w t owner bank op =>
        if owner = slot.1.val ∧ bank = slot.2 then some (⟨w,t⟩,op) else none
    | _ => none

private def image (w : Write) := (w.event,w.value.setWidth 32)

def stores (rows : List (Event × Joint.AlphaWord)) : List Write :=
  rows.filterMap fun (e,op) => match op with
    | .store v => some ⟨e,v.zeroExtend 64,false,false⟩
    | _ => none

/-- Identify the seed of this particular use and check every store from the seed
    onward, including a warm slot's later reports and reinitializations. -/
def prefixStores (all_ current later : List Write) (seed : Event) : Option (List Write) := do
  let pre := all_.takeWhile (fun x => x.event != seed)
  guard ((all_.map Write.event).Nodup && current.head?.any (fun w => w.event == seed))
  guard (all_.map image == (pre ++ current ++ later).map image)
  return pre

structure Sample where
  event : Event
  source : Event
  word : BitVec 32

def PrefixContract := ∀ (pre : List Write) (seed : Write) (updates later : List Write),
  ((pre ++ (seed :: updates) ++ later).map Write.event).Nodup →
    prefixStores (pre ++ (seed :: updates) ++ later) (seed :: updates) later seed.event = some pre

theorem recoverPrefix : PrefixContract := by
  intro pre seed updates later distinct
  have absent (w : Write) (mem : w ∈ pre) : w.event ≠ seed.event := by
    have disjoint := (List.pairwise_append.mp (by
      simpa only [List.map_append,List.append_assoc] using distinct :
        ((pre.map Write.event) ++ ((seed :: updates).map Write.event ++ later.map Write.event)).Nodup)).2.2
    exact disjoint _ (List.mem_map_of_mem mem) _ (by simp)
  have split : (pre ++ (seed :: updates) ++ later).takeWhile (fun w => w.event != seed.event) = pre := by
    rw [List.append_assoc,List.takeWhile_append_of_pos (fun w mem => by simpa using absent w mem)]
    simp
  unfold prefixStores
  dsimp only
  rw [split]
  rw [decide_eq_true distinct]
  simp [guard]

def Sample.decode (r : Sample) (slot : Slot) (ws : List Write) : Option AlphaOrigin.Register := do
  let i ← OwnerMaskGuards.source ws r.source
  if (ws.get i).value.setWidth 32 = r.word then return ⟨slot,.alpha,r.event,i.val,r.word,false⟩
  none

def SourceContract := ∀ (r : Sample) (slot : Slot) (ws : List Write) (reg : AlphaOrigin.Register),
  r.decode slot ws = some reg → ∃ i : Fin ws.length,
    reg.source = i.val ∧ (ws.get i).event = r.source ∧
    reg.event = r.event ∧ (ws.get i).value.setWidth 32 = r.word

theorem readSource : SourceContract := by
  intro r slot ws reg hit
  obtain ⟨i,found,rest⟩ := Option.bind_eq_some_iff.mp hit
  split at rest <;> try contradiction
  cases rest
  exact ⟨i,rfl,OwnerMaskGuards.source_event found,rfl,by assumption⟩

def LoadContract := ∀ (site : Event) (slot : Slot) (ws : List Write) (i : Fin ws.length),
  (ws.map Write.event).Nodup →
  (Sample.mk site (ws.get i).event ((ws.get i).value.setWidth 32)).decode slot ws =
    some ⟨slot,.alpha,site,i.val,(ws.get i).value.setWidth 32,false⟩

theorem recoverLoad : LoadContract := by
  intro site slot ws i distinct
  unfold Sample.decode
  dsimp only
  rw [OwnerMaskGuards.source_of_distinct distinct i]
  simp

def samples (xs : List (Event × Joint.AlphaWord)) := xs.filterMap fun (e,op) => match op with
  | .load src value => some (⟨e,src,value⟩ : Sample)
  | _ => none

def inOrder (c : MaskClock.Capture h) (ws : List Write) (site : Event) (rank : Nat) : Bool :=
  let transfers := c.transfers
  decide (∀ i : Fin ws.length,
    ((FlagSegment.published transfers (ws.get i).event site).isSome = true → i.val ≤ rank) ∧
    ((FlagSegment.published transfers site (ws.get i).event).isSome = true → rank < i.val))

def coherent (c : MaskClock.Capture h) (ws : List Write) (r : AlphaOrigin.Register) :=
  inOrder c ws r.event r.source

def CoherenceContract := ∀ (h : CutoffVisibility.History) (c : MaskClock.Capture h)
  (ws : List Write) (r : AlphaOrigin.Register), coherent c ws r = true ↔ r.Coherent h ws

theorem atomicOrder : CoherenceContract := by
  intro h c ws r
  simp only [coherent,inOrder,decide_eq_true_eq,← MaskClock.published h c]
  exact ⟨fun f => ⟨fun i => (f i).1,fun i => (f i).2⟩,fun f i => ⟨f.1 i,f.2 i⟩⟩

/-- These registers have already been projected to one physical alpha. If two
    of its reads happen in order, the later one cannot return to an older write. -/
def readOrder (c : MaskClock.Capture h) (rows : List AlphaOrigin.Register) :=
  let transfers := c.transfers
  rows.all (fun a => rows.all fun b =>
    decide ((FlagSegment.published transfers a.event b.event).isSome = true → a.source ≤ b.source))

def ReadOrderContract := ∀ (h : CutoffVisibility.History) (c : MaskClock.Capture h)
  (rows : List AlphaOrigin.Register),
  readOrder c rows = true ↔ ∀ a ∈ rows, ∀ b ∈ rows,
    CutoffVisibility.HappensBefore h a.event b.event → a.source ≤ b.source

theorem readOrder_iff : ReadOrderContract := by
  intro h c rows
  simp only [readOrder,List.all_eq_true,decide_eq_true_eq,← MaskClock.published h c]

private def mutex (c : MaskClock.Capture h) (init : Event) (ops : List SpinLock.Operation) :
    Option (Mutex.Capture h) := do
  if ok : (SlotSynchronization.lockEdges ops).all (fun p => c.sync.edges.contains p) then
    return ⟨init,ops,fun n a b first next => (c.sync.exact ..).mp (by
      have mem := List.all_eq_true.mp ok (a,b) ((SlotSynchronization.adjacent ..).mpr ⟨n,first,next⟩)
      exact List.contains_iff_mem.mp mem)⟩
  none

/-- The initial atomic writes of the physical spinlocks. All subsequent CAS
    and guard-drop records are obtained from the common tape. -/
structure Initialization where
  slot : Slot → Event
  worker : Nat → Event

def workerMutexes (c : MaskClock.Capture h) (es : Joint.Tape Operand) (init : Nat → Event) :
    Option (Startup.StateLocks h) :=
  (SlotSynchronization.workers es).mapM fun w => do
    return (w,← mutex c (init w) (SlotSynchronization.workerOps w es))

def addressed {t : NativeSplit.Tickets rc pv beta} (b : OwnerOperands t) (c : MaskClock.Capture b.memory.history)
    (es : Joint.Tape Operand) (init : Initialization) : Option (AlphaOrigin.Addressed b.owner) := do
  let slot := (b.generation.owner,b.bankIndex)
  let ops := accesses slot es
  let later := subsequent b.owner.exit slot .alpha b.owner.later
  let pre ← prefixStores (stores ops) t.final.alpha later t.a.event
  let ws := pre ++ t.final.alpha ++ later
  let rows ← (samples ops).mapM fun r => r.decode slot ws
  guard ((rows.map AlphaOrigin.Register.event).Nodup && readOrder c rows)
  if order : rows.all (coherent c ws) then
    let co r member _ _ := (atomicOrder _ c ws r).mp (List.all_eq_true.mp order r member)
    let slotLock ← mutex c (init.slot slot) (SlotSynchronization.operations slot.1 slot.2 es)
    let locks ← workerMutexes c es init.worker
    if covers : Mutex.covers slotLock.ops (t.journal.map fun row => (row.acquired,row.published)) then
      if start : t.dispatches.all fun d => !(d.sections.any SplitSchedulerMir.WorkerFields.Section.needsReset) ||
          Startup.ready slotLock.ops (Startup.stateOps locks)
            (b.owner.exit.timeline.atStep (SplitSchedulerMir.initPhases.idxOf .assign)) d.entry then
        if regs : t.state.writes.all (NativeSplit.AlphaLoads.ticket (AlphaOrigin.memory slot ws rows co).reads t.journal t.stream) then
          return ⟨pre,slotLock,covers,locks,start,rows,co,regs⟩
  none

namespace Regression

private def slot : Slot := (0,0)
private def equalWord : BitVec 32 := BitVec.ofInt 32 512

/-- Two distinct stores write the exact same alpha value `512` on worker 0. -/
private def equalTape (sameWorker : Bool) (syncReaders : Bool) (firstSrc secondSrc : Event) : Joint.Tape Unit :=
  [.alphaWord 0 1 0 0 (.store equalWord),
   .alphaWord 0 5 0 0 (.store equalWord),
   .alphaWord 1 10 0 0 (.load firstSrc equalWord)] ++
  (if !sameWorker && syncReaders then [.workerLock 1 11 2 false, .workerLock 2 19 2 true] else []) ++
  [.alphaWord (if sameWorker then 1 else 2) 20 0 0 (.load secondSrc equalWord)]

private def evalEqual (sameWorker syncReaders : Bool) (firstSrc secondSrc : Event) : Option (Bool × Bool × Bool) := do
  let es := equalTape sameWorker syncReaders firstSrc secondSrc
  let base : CutoffVisibility.History := ⟨[],fun _ => [],[],[],fun _ _ => False⟩
  let edges := if !sameWorker && syncReaders then [(⟨1,11⟩,⟨2,19⟩)] else []
  let ⟨_,c⟩ ← SlotSynchronization.capture base es edges []
  let ops := accesses slot es
  let ws := stores ops
  let rows ← (samples ops).mapM fun r => r.decode slot ws
  let [r1,r2] := rows | none
  return (coherent c ws r1, coherent c ws r2, readOrder c rows)

/-- Both loads read an identical alpha word and each load is individually coherent
    against write-read and read-write order. Returning to the older write on the
    later load is rejected whenever the two loads are HB-ordered (same worker or
    via cross-worker synchronization), and permitted only when unordered. -/
example :
    evalEqual true false ⟨0,5⟩ ⟨0,1⟩ = some (true,true,false) ∧
    evalEqual true false ⟨0,5⟩ ⟨0,5⟩ = some (true,true,true) ∧
    evalEqual true false ⟨0,1⟩ ⟨0,5⟩ = some (true,true,true) ∧
    evalEqual false false ⟨0,5⟩ ⟨0,1⟩ = some (true,true,true) ∧
    evalEqual false true ⟨0,5⟩ ⟨0,1⟩ = some (true,true,false) := by
  cbv

end Regression

end NeuralReversi.PvsParallel.Interleaving.AlphaReplay

import NeuralReversi.PvsParallel.Interleaving.CutoffPollReplay
import NeuralReversi.PvsParallel.Interleaving.SlotSynchronization

namespace NeuralReversi.PvsParallel.Interleaving.SplitTapeReplay

open Threading
open AtomicWord (Event Write Read SequencedBefore InReleaseSequence)
open CutoffVisibility (History FlagWrite FlagRead HappensBefore)
open TaskAncestry (Journal Generation)
open SlotFrameMir.RawProjection (OwnerSetup subsequent)

local instance : Decidable (SequencedBefore a b) :=
  inferInstanceAs (Decidable (a.worker = b.worker ∧ a.step < b.step))

local instance {ws : List Write} {i j : Fin ws.length} : Decidable (InReleaseSequence ws i j) :=
  inferInstanceAs (Decidable ((ws.get i).release = true ∧ i.val ≤ j.val ∧
    ∀ k : Fin ws.length, i.val < k.val → k.val ≤ j.val → (ws.get k).rmw = true))

local instance (word : BitVec 64) : Decidable (SearchAbort.Stopped word) :=
  decidable_of_iff (SearchAbort.generation word = SearchAbort.requested word)
    (SearchAbort.stopped_iff word).symm

local instance : DecidableEq Write := fun ⟨e1,v1,r1,m1⟩ ⟨e2,v2,r2,m2⟩ =>
  if h : e1 = e2 ∧ v1 = v2 ∧ r1 = r2 ∧ m1 = m2 then
    isTrue (by rcases h with ⟨rfl,rfl,rfl,rfl⟩; rfl)
  else isFalse (fun eq => by cases eq; exact h ⟨rfl,rfl,rfl,rfl⟩)

local instance {ws : List Write} : DecidableEq (Read ws) := fun ⟨e1,s1,a1⟩ ⟨e2,s2,a2⟩ =>
  if h : e1 = e2 ∧ s1 = s2 ∧ a1 = a2 then
    isTrue (by rcases h with ⟨rfl,rfl,rfl⟩; rfl)
  else isFalse (fun eq => by cases eq; exact h ⟨rfl,rfl,rfl⟩)

/-- Locate the unique physical write at `src` in `ws`. -/
def findWrite (ws : List Write) (src : Event) : Option (Fin ws.length) :=
  match (List.finRange ws.length).filter (fun i => (ws.get i).event == src) with
  | [i] => some i
  | _ => none

def findFlag (ws : List FlagWrite) (src : Event) : Option (Fin ws.length) :=
  match (List.finRange ws.length).filter (fun i => (ws.get i).event == src) with
  | [i] => some i
  | _ => none

def findRead {ws : List Write} (rs : List (Read ws)) (site : Event) : Option {r // r ∈ rs} :=
  match rs.attach.filter (fun r => r.val.event == site) with
  | [r] => some r
  | _ => none

/-- Extract the physical cutoff_epoch modification order from the unified tape. -/
def epochWrites (es : Joint.Tape Operand) : List Write :=
  es.filterMap fun
    | .epochWord w t (.init v) => some ⟨⟨w,t⟩,v,false,false⟩
    | .epochWord w t (.bump v) => some ⟨⟨w,t⟩,v,true,true⟩
    | _ => none

/-- Extract the physical cutoff flag modification order per slot address. -/
def flagWrites (es : Joint.Tape Operand) (slot : Nat) : List FlagWrite :=
  es.filterMap fun
    | .taskWord w t (.storeFlag s v) => if s == slot then some ⟨⟨w,t⟩,v⟩ else none
    | _ => none

/-- Extract all cutoff_epoch loads (both reset Relaxed loads and poll Acquire loads). -/
def epochReads (es : Joint.Tape Operand) (ws : List Write) : Option (List (Read ws)) := do
  let raw := es.filterMap fun
    | .taskWord _ _ (.resetCache load src val) => some (load,src,val,false)
    | .taskWord _ _ (.epoch load src val _) => some (load,src,val,true)
    | _ => none
  raw.mapM fun (load,src,val,acq) => do
    let idx ← findWrite ws src
    guard ((ws.get idx).value == val)
    return ⟨load,idx,acq⟩

/-- Extract all cutoff flag loads with verified write sources. -/
def flagReads (es : Joint.Tape Operand) (fl : Nat → List FlagWrite) : Option (List (FlagRead fl)) := do
  let raw := es.filterMap fun
    | .taskWord w t (.loadFlag slot val src) => some (slot,⟨w,t⟩,val,src)
    | _ => none
  raw.mapM fun (slot,ev,val,src) => do
    let idx ← findFlag (fl slot) src
    guard (((fl slot).get idx).value == val)
    return ⟨slot,ev,idx⟩

/-- Extract the shared AbortState modification order from the unified tape. -/
def abortWrites (es : Joint.Tape Operand) : List Write :=
  es.filterMap fun
    | .abortWord w t (.init v) => some ⟨⟨w,t⟩,v,false,false⟩
    | .abortWord w t (.begin v) => some ⟨⟨w,t⟩,v,true,true⟩
    | .abortWord w t (.request v) => some ⟨⟨w,t⟩,v,true,true⟩
    | _ => none

/-- Extract all shared AbortState loads with verified write sources. -/
def abortReads (es : Joint.Tape Operand) (ws : List Write) : Option (List (Read ws)) := do
  let raw := es.filterMap fun
    | .abortWord w t (.load src val acq) => some (⟨w,t⟩,src,val,acq)
    | _ => none
  raw.mapM fun (ev,src,val,acq) => do
    let idx ← findWrite ws src
    guard ((ws.get idx).value == val)
    return ⟨ev,idx,acq⟩

/-- Extract per-worker helper_job_aborted writes from the unified tape. -/
def helperWrites (es : Joint.Tape Operand) (target : Nat) : List Write :=
  es.filterMap fun
    | .helperAbortWord w t (.store tgt val rel) => if tgt == target then some ⟨⟨w,t⟩,val,rel,false⟩ else none
    | _ => none

/-- Workers that perform helper_job_aborted loads on the tape. -/
def helperLoadWorkers (es : Joint.Tape Operand) : List Nat :=
  (es.filterMap fun
    | .helperAbortWord w _ (.load ..) => some w
    | _ => none).eraseDups

/-- Extract verified helper_job_aborted loads for one worker. -/
def helperReadsFor (es : Joint.Tape Operand) (ws : List Write) (target : Nat) : Option (List (Read ws)) := do
  let raw := es.filterMap fun
    | .helperAbortWord w t (.load src val acq) => if w == target then some (⟨w,t⟩,src,val,acq) else none
    | _ => none
  raw.mapM fun (ev,src,val,acq) => do
    let idx ← findWrite ws src
    guard ((ws.get idx).value == val)
    return ⟨ev,idx,acq⟩

/-- Extract physical best_score writes on `(owner, bank)` from the unified tape. -/
def bestStores (owner bank : Nat) (es : Joint.Tape Operand) : List Write :=
  es.filterMap fun
    | .ownerStore w t p b (.best v) => if p == owner && b == bank then some ⟨⟨w,t⟩,v.zeroExtend 64,false,false⟩ else none
    | _ => none

/-- Extract physical best_move writes on `(owner, bank)` from the unified tape. -/
def moveStores (owner bank : Nat) (es : Joint.Tape Operand) : List Write :=
  es.filterMap fun
    | .ownerStore w t p b (.picked v) => if p == owner && b == bank then some ⟨⟨w,t⟩,v.zeroExtend 64,false,false⟩ else none
    | _ => none

/-- Release-sequence synchronization edges induced by acquire loads on an atomic word. -/
def releaseEdges (ws : List Write) (rs : List (Read ws)) : List (Event × Event) :=
  rs.flatMap fun r =>
    if r.acquire then
      ((List.finRange ws.length).filter (fun j => decide (InReleaseSequence ws j r.source))).map
        fun j => ((ws.get j).event,r.event)
    else []

/-- All atomic histories, reads, and causal synchronization extracted from ONE common tape. -/
structure Extracted where
  memory : NativeStopReturns.Memory
  transfers : List (Event × Event)
  chronology : MaskClock.Capture memory.history

/-- Derive `NativeStopReturns.Memory` (cutoff_epoch, flags, AbortState, helper_job_aborted)
    and the causal clock `MaskClock.Capture` from one instruction tape. -/
def extract (es : Joint.Tape Operand) (other : List (Event × Event)) (order : List Event) : Option Extracted := do
  let ep := epochWrites es
  let fl := flagWrites es
  let epReads ← epochReads es ep
  let flReads ← flagReads es fl
  let ab := abortWrites es
  let abReads ← abortReads es ab
  let hw := helperWrites es
  let loadWs := helperLoadWorkers es
  guard (loadWs.all fun w => (helperReadsFor es (hw w) w).isSome)
  let hr : (w : Nat) → List (Read (hw w)) := fun w => (helperReadsFor es (hw w) w).getD []
  let hEdges := loadWs.flatMap fun w => releaseEdges (hw w) (hr w)
  let base : History := ⟨ep,fl,epReads,flReads,fun _ _ => False⟩
  let extra := releaseEdges ab abReads ++ hEdges ++ other
  let ⟨transfers,clock⟩ ← SlotSynchronization.capture base es extra order
  let mem : NativeStopReturns.Memory := ⟨SlotSynchronization.history base transfers,ab,abReads,hw,hr⟩
  return ⟨mem,transfers,clock⟩

/-- Full WR, RW, and RR coherence for shared AbortState reads. -/
structure AbortMemory (m : NativeStopReturns.Memory) : Prop where
  coherent : ∀ r ∈ m.abortReads, SplitRebookingMir.Coherence m.history m.abort r
  readRead : ∀ a ∈ m.abortReads, ∀ b ∈ m.abortReads,
    HappensBefore m.history a.event b.event → a.source.val ≤ b.source.val

def abortMemoryOK (m : NativeStopReturns.Memory) (c : MaskClock.Capture m.history) : Bool :=
  let transfers := c.transfers
  let before (a b : Event) := (FlagSegment.published transfers a b).isSome = true
  m.abortReads.all fun r =>
    decide (∀ w : Fin m.abort.length,
      (before (m.abort.get w).event r.event → w.val ≤ r.source.val) ∧
      (before r.event (m.abort.get w).event → r.source.val < w.val)) &&
    m.abortReads.all fun b => decide (before r.event b.event → r.source.val ≤ b.source.val)

def AbortMemoryContract : Prop :=
  ∀ (m : NativeStopReturns.Memory) (c : MaskClock.Capture m.history),
    abortMemoryOK m c = true ↔ AbortMemory m

theorem checkedAbort : AbortMemoryContract := by
  intro m c
  simp only [abortMemoryOK,List.all_eq_true,Bool.and_eq_true,decide_eq_true_eq,← MaskClock.published m.history c]
  constructor
  · intro ok
    exact ⟨fun r hr => ⟨fun w => ((ok r hr).1 w).1,fun w => ((ok r hr).1 w).2⟩,
      fun a ha b hb => (ok a ha).2 b hb⟩
  · intro h r hr
    exact ⟨fun w => ⟨(h.coherent r hr).writeRead w,(h.coherent r hr).readWrite w⟩,
      fun b hb => h.readRead r hr b hb⟩

/-- Full WR and RW coherence for a worker's helper_job_aborted loads. -/
structure HelperMemory (m : NativeStopReturns.Memory) (w : Nat) : Prop where
  coherent : ∀ r ∈ m.helperReads w, SplitRebookingMir.Coherence m.history (m.helper w) r

def helperMemoryOK (m : NativeStopReturns.Memory) (c : MaskClock.Capture m.history) (w : Nat) : Bool :=
  let transfers := c.transfers
  let before (a b : Event) := (FlagSegment.published transfers a b).isSome = true
  (m.helperReads w).all fun r =>
    decide (∀ i : Fin (m.helper w).length,
      (before ((m.helper w).get i).event r.event → i.val ≤ r.source.val) ∧
      (before r.event ((m.helper w).get i).event → r.source.val < i.val))

def HelperMemoryContract : Prop :=
  ∀ (m : NativeStopReturns.Memory) (c : MaskClock.Capture m.history) (w : Nat),
    helperMemoryOK m c w = true ↔ HelperMemory m w

theorem checkedHelper : HelperMemoryContract := by
  intro m c w
  simp only [helperMemoryOK,List.all_eq_true,decide_eq_true_eq,← MaskClock.published m.history c]
  constructor
  · intro ok
    exact ⟨fun r hr => ⟨fun i => (ok r hr i).1,fun i => (ok r hr i).2⟩⟩
  · intro h r hr i
    exact ⟨(h.coherent r hr).writeRead i,(h.coherent r hr).readWrite i⟩

/-- Reconstruct one search generation of `AbortState` from the physical write log,
    explicitly verifying `begin_search` clears `Stopped` (including across the
    32-bit `0xFFFFFFFF -> 0` generation wrap) and every subsequent write in the
    era is a `request` RMW. -/
def decodeAbortEra (m : NativeStopReturns.Memory) (beginEvent finalEvent : Event) :
    Option (StopSchedule.AbortEra m.abort) := do
  let bIdx ← findWrite m.abort beginEvent
  let fIdx ← findWrite m.abort finalEvent
  guard (bIdx.val ≤ fIdx.val)
  if prevCheck : 0 < bIdx.val then
    let prev : Fin m.abort.length := ⟨bIdx.val - 1, by omega⟩
    guard ((m.abort.get bIdx).value == SearchAbort.begin (m.abort.get prev).value)
  else
    guard (!decide (SearchAbort.Stopped (m.abort.get bIdx).value))
  if rmw : ∀ i : Fin m.abort.length, bIdx.val < i.val → i.val ≤ fIdx.val →
      (m.abort.get i).value = SearchAbort.request (m.abort.get ⟨i.val-1,by have := i.isLt; omega⟩).value then
    return ⟨bIdx,fIdx,rmw⟩
  else none

/-- Reconstruct `OwnerCheck` directly from the owner's post-wait ancestor scan
    and optional post-wait `AbortState` load on the same tape. -/
def decodeOwnerCheck (m : NativeStopReturns.Memory) (j : Journal) (afterSplit : Event)
    (ownerAbort : Option Event) (helperJob : Bool) : Option (SplitReturn.Observation.OwnerCheck m.history) := do
  let scan ← CutoffPollReplay.ownerScan m.history j afterSplit
  match ownerAbort with
  | none =>
    if stopped : scan.stopped = true then
      return ⟨scan,none,fun _ => stopped,helperJob⟩
    else none
  | some ev => do
    let ⟨q,_⟩ ← findRead m.abortReads ev
    guard q.acquire
    return ⟨scan,some ⟨m.abort,q⟩,by simp,helperJob⟩

/-- Reconstruct `StopSchedule.Shared` directly from the extracted abort history
    and causal clock, with no external abort oracle. -/
def decodeShared (m : NativeStopReturns.Memory) (c : MaskClock.Capture m.history)
    (check : SplitReturn.Observation.OwnerCheck m.history) (wait beginEvent finalEvent : Event) :
    Option (StopSchedule.Shared m check wait) := do
  let era ← decodeAbortEra m beginEvent finalEvent
  if abOk : abortMemoryOK m c = true then
    have abMem := (checkedAbort m c).mp abOk
    match hShared : check.shared with
    | none =>
      return ⟨none,hShared,era,abMem.coherent,abMem.readRead,fun _ h => by cases h⟩
    | some ⟨ws,load⟩ => do
      if sameWs : ws = m.abort then
        let q0 : Read m.abort := sameWs ▸ load
        let ⟨q,qMem⟩ ← findRead m.abortReads q0.event
        if sameLoad : q0 = q then
          if afterWait : SequencedBefore wait q.event then
            if beforeNext : ∀ i : Fin m.abort.length, i.val = era.final.val + 1 →
                (FlagSegment.published c.transfers q.event (m.abort.get i).event).isSome = true then
              have nextHB : ∀ i : Fin m.abort.length, i.val = era.final.val + 1 →
                  HappensBefore m.history q.event (m.abort.get i).event :=
                fun i eq => (MaskClock.published m.history c _ _).mpr (beforeNext i eq)
              have obsEq : (⟨ws,load⟩ : SplitReturn.Observation.AbortObservation) = ⟨m.abort,q⟩ := by
                subst sameWs
                exact congrArg (SplitReturn.Observation.AbortObservation.mk m.abort) sameLoad
              return ⟨some q,by simp [hShared,obsEq],era,abMem.coherent,abMem.readRead,
                fun q' eq => by cases eq; exact ⟨qMem,afterWait,nextHB⟩⟩
      none
  else none

/-- Reconstruct the non-flag loads executed inside a single `should_stop` poll
    in `[started, poll.step]` directly from the unified tape. -/
def decodePollLoads (m : NativeStopReturns.Memory) (es : Joint.Tape Operand) (poll : Event) (started : Nat) :
    Option (NativeStopReturns.Loads m poll) := do
  if interval : started ≤ poll.step then
    let abortSites := es.filterMap fun
      | .abortWord w t (.load _ _ true) =>
        if w == poll.worker && started ≤ t && t ≤ poll.step then some ⟨w,t⟩ else none
      | _ => none
    let helperSites := es.filterMap fun
      | .helperAbortWord w t (.load ..) =>
        if w == poll.worker && started ≤ t && t ≤ poll.step then some ⟨w,t⟩ else none
      | _ => none
    let abortOpt ← match abortSites with
      | [] => some none
      | [ev] => do
        let ⟨r,mem⟩ ← findRead m.abortReads ev
        if ok : r.acquire = true ∧ NativeStopReturns.InWorker poll r.event ∧ started ≤ r.event.step then
          some (some (⟨r,mem,ok⟩ : {r : Read m.abort // r ∈ m.abortReads ∧ r.acquire = true ∧
            NativeStopReturns.InWorker poll r.event ∧ started ≤ r.event.step}))
        else none
      | _ => none
    let helperOpt ← match helperSites with
      | [] => some none
      | [ev] => do
        let ⟨r,mem⟩ ← findRead (m.helperReads poll.worker) ev
        if ok : NativeStopReturns.InWorker poll r.event ∧ started ≤ r.event.step then
          some (some (⟨r,mem,ok⟩ : {r : Read (m.helper poll.worker) // r ∈ m.helperReads poll.worker ∧
            NativeStopReturns.InWorker poll r.event ∧ started ≤ r.event.step}))
        else none
      | _ => none
    return {
      started := started
      abort := abortOpt.map Subtype.val
      helper := helperOpt.map Subtype.val
      interval := interval
      abortRecorded := fun r hr => by
        cases abortOpt with
        | none => cases hr
        | some x => cases hr; exact x.property
      helperRecorded := fun r hr => by
        cases helperOpt with
        | none => cases hr
        | some x => cases hr; exact x.property }
  else none

/-- Reconstruct a `HelperJobRead` proof directly from the worker's `helper_job_aborted`
    write log when the worker's clear precedes its poll and no non-zero write occurs
    after that clear in this session. -/
def decodeHelperClear (m : NativeStopReturns.Memory) (c : MaskClock.Capture m.history)
    (w : Nat) (clearEvent : Event) (r : Read (m.helper w)) (mem : r ∈ m.helperReads w) :
    Option (Σ' clr : Fin (m.helper w).length, SplitReturn.Observation.HelperJobRead m.history r clr) := do
  if hOk : helperMemoryOK m c w = true then
    have hMem := (checkedHelper m c w).mp hOk
    let clr ← findWrite (m.helper w) clearEvent
    if zero : ((m.helper w).get clr).value = 0 then
      if own : SequencedBefore ((m.helper w).get clr).event r.event then
        if noLater : ∀ i : Fin (m.helper w).length, clr.val < i.val → ((m.helper w).get i).value = 0 then
          have coh := hMem.coherent r mem
          return ⟨clr,⟨zero,.own own,coh.writeRead,coh.readWrite,
            fun i after nz => False.elim (nz (noLater i after))⟩⟩
  none

/-- Spec for reconstructing a non-flag poll reason directly from the tape
    (replaces the opaque `ctx.memory` oracle in `CutoffPollReplay.Context`). -/
structure PollSpec where
  started : Event → Nat
  stamp : Nat → Event
  abortBegin : Event
  abortFinal : Event
  helperClear : Nat → Option Event

/-- Derive `StopSchedule.PollMemory` directly from the unified instruction tape `es`,
    without passing an opaque per-poll memory oracle. -/
def pollMemoryFromTape (m : NativeStopReturns.Memory) (c : MaskClock.Capture m.history)
    (j : Journal) (es : Joint.Tape Operand) (g : Generation)
    (check : SplitReturn.Observation.OwnerCheck m.history) (wait : Event)
    (spec : PollSpec) (ps : List Tickets.Post) (poll : Event) (reads : List (FlagRead m.history.flags)) :
    Option (StopSchedule.PollMemory m check wait g.slot ps poll reads) := do
  if flagOk : CutoffPollReplay.flagMemoryOK c = true then
    if epochOk : CutoffPollReplay.epochMemoryOK c = true then
      have _ := (CutoffPollReplay.checkedEpochs m.history c).mp epochOk
      have fm := (CutoffPollReplay.checkedFlags m.history c).mp flagOk
      if recheck : SequencedBefore wait check.ancestors.event then
        let era ← FlagSegment.recover m.history j g
        if pub : era.marks = StopSchedule.marks spec.stamp ps then
          let shared ← decodeShared m c check wait spec.abortBegin spec.abortFinal
          let loads ← decodePollLoads m es poll (spec.started poll)
          if localStart : ∀ r ∈ reads, r.slot = g.slot → NativeStopReturns.InWorker poll r.event →
              (FlagSegment.published c.transfers era.clear r.event).isSome = true then
            if localEnd : ∀ r ∈ reads, r.slot = g.slot → NativeStopReturns.InWorker poll r.event →
                ∀ i : Fin (m.history.flags g.slot).length, i.val = era.pre.length + 1 + era.marks.length →
                  (FlagSegment.published c.transfers r.event ((m.history.flags g.slot).get i).event).isSome = true then
              if abortStart : ∀ r, loads.abort = some r →
                  (FlagSegment.published c.transfers (m.abort.get shared.era.begin).event r.event).isSome = true then
                let helperOk : Option (PLift (∀ r, loads.helper = some r →
                    ∃ clr, SplitReturn.Observation.HelperJobRead m.history r clr)) :=
                  match hH : loads.helper with
                  | none => some ⟨fun _ h => by cases h⟩
                  | some hr => do
                    let clrEv ← spec.helperClear poll.worker
                    let mem := (loads.helperRecorded hr hH).1
                    let ⟨clr,proof⟩ ← decodeHelperClear m c poll.worker clrEv hr mem
                    return ⟨fun r' eq => by cases eq; exact ⟨clr,proof⟩⟩
                let helperProof ← helperOk
                return {
                  loads := loads
                  recheck := recheck
                  coherent := fm.coherent
                  readRead := fm.readRead
                  currentEra := era.toFlags
                  stamp := spec.stamp
                  published := pub
                  localStart := fun r hr slot local_ =>
                    (MaskClock.published m.history c _ _).mpr (localStart r hr slot local_)
                  localEnd := fun r hr slot local_ i pos =>
                    (MaskClock.published m.history c _ _).mpr (localEnd r hr slot local_ i pos)
                  sharedState := shared
                  abortStart := fun r hr =>
                    (MaskClock.published m.history c _ _).mpr (abortStart r hr)
                  helper := helperProof.down }
  none

/-- Check that `OwnerSetup`'s `best_score` and `best_move` write arrays match the
    physical `.ownerStore` instructions on the same unified tape. -/
def ownerStoresMatch {t : NativeSplit.Tickets rc pv beta} {g : SlotFrameMir.RawMask mask}
    (b : OwnerSetup g h time slot t) (es : Joint.Tape Operand) : Bool :=
  bestStores slot.1.val slot.2 es == b.bPre ++ t.final.best ++ subsequent b.exit slot .score b.later &&
  moveStores slot.1.val slot.2 es == b.mPre ++ t.final.moves ++ subsequent b.exit slot .move b.later

/-- Check that a split's `OwnerSetup` (mask, owner exit, `best_score`/`best_move`
    stores, `OwnerCheck`) and non-flag stop polls all originate from ONE tape `es`,
    and run `CutoffPollReplay.fromTape` (which also verifies Task 1 `ActivePointer`,
    `OwnerTape`, `AlphaReplay`, `EpochHit`, and `SchedulerTrace`). -/
def replayFromTape {t : NativeSplit.Tickets rc pv beta}
    (ext : Extracted) (b : AlphaOrigin.SetupOperands t)
    (sameMem : b.memory = ext.memory)
    (root cut : Bool) (depth : Nat) (es : Joint.Tape Operand)
    (lockInit : AlphaReplay.Initialization) (spec : PollSpec)
    (afterSplit : Event) (ownerAbort : Option Event) (helperJob : Bool)
    (before : Option (Event → Event → Bool) := none) :
    Option (AlphaOrigin.Capture t root cut depth (fun _ _ _ _ _ => True)) := do
  let c : MaskClock.Capture b.memory.history := sameMem ▸ ext.chronology
  guard (ownerStoresMatch b.owner es)
  guard (MaskGeneration.memoryOK c b.generation.owner (MaskGeneration.rows b.generation.owner b.bankIndex es))
  guard (CutoffPollReplay.flagMemoryOK c && CutoffPollReplay.epochMemoryOK c && abortMemoryOK b.memory c)
  let j ← TaskAncestry.capture (CutoffPollReplay.slots b) es
  let check ← decodeOwnerCheck b.memory j afterSplit ownerAbort helperJob
  guard (check.ancestors.event == b.owner.check.ancestors.event &&
    check.ancestors.chain == b.owner.check.ancestors.chain &&
    check.stopped == b.owner.check.stopped)
  let ctx : CutoffPollReplay.Context b es := {
    chronology := c
    lockInit := lockInit
    memory := fun _ ps poll rs =>
      pollMemoryFromTape b.memory c j es (CutoffPollReplay.generation b) b.owner.check
        (b.owner.exit.clock.atStep (3 + b.owner.exit.log.length)) spec ps poll rs }
  CutoffPollReplay.fromTape b root cut depth es ctx before

/-- Contract 1 (32-bit wrap & abort propagation from tape):
    When `AbortState.begin_search` wraps from generation `0xFFFFFFFF` to `0`,
    it clears `Stopped`. Any subsequent `request` RMW in generation `0` observed
    by a worker's poll forces `ExternalStop` and therefore `check.stopped = true`
    when replayed directly from the unified instruction tape. -/
def AbortEraWrapContract : Prop :=
  (∀ prev : BitVec 64, SearchAbort.generation prev = BitVec.allOnes 32 →
    SearchAbort.generation (SearchAbort.begin prev) = 0#32 ∧
    ¬ SearchAbort.Stopped (SearchAbort.begin prev) ∧
    SearchAbort.Stopped (SearchAbort.request (SearchAbort.begin prev))) ∧
  (∀ {Operand : Type} (m : NativeStopReturns.Memory) (c : MaskClock.Capture m.history)
    (j : Journal) (es : Joint.Tape Operand) (g : Generation)
    (check : SplitReturn.Observation.OwnerCheck m.history) (wait : Event)
    (spec : PollSpec) (ps : List Tickets.Post) (poll : Event),
    StopSchedule.Departed m.history poll wait →
    ∀ (s : StopSchedule.Samples m check wait g.slot ps poll),
    CutoffPollReplay.sampled j es g m check wait ps poll (fun rs => do
      let pm ← pollMemoryFromTape m c j es g check wait spec ps poll rs
      let parents ← TaskRetention.parentEra (MaskGeneration.input c es (fun _ _ => none)) es g poll check.ancestors rs (fun _ _ => none) c.before
      return ⟨pm,parents⟩) = some s →
    s.cache.latched = false →
    check.stopped = true)

theorem abortEraWrapContract_proven : AbortEraWrapContract := by
  constructor
  · intro prev wrapped
    have gw := SearchAbort.generationWordContract_proven prev
    have gen0 : SearchAbort.generation (SearchAbort.begin prev) = 0#32 := by
      rw [gw.1,wrapped]; decide
    exact ⟨gen0,gw.2.2.2.2.2,SearchAbort.request_stops _⟩
  · intro _ m c j es g check wait spec ps poll dep s _ unlatched
    have yes := s.positive
    cases ar : s.loads.abort with
    | none => simp [NativeStopReturns.poll,NativeStopReturns.ChainRead.set,unlatched,ar] at yes
    | some r =>
      cases hr : s.loads.helper with
      | none =>
        have stop : (∃ p ∈ ps, p.cut = true) ∨ SplitReturn.Observation.ExternalStop m.history check wait :=
          s.cause dep
        rcases stop with ⟨_,_⟩ | ext
        · -- Even in the general case, the `unlatched` branch of `s.cause` yields `ExternalStop`.
          have ⟨mem,_,localRead,_⟩ := s.loads.abortRecorded r ar
          have stopped : NativeStopReturns.abortBit m r = true := by
            simpa [NativeStopReturns.poll,NativeStopReturns.ChainRead.set,unlatched,ar,hr] using yes
          have ext : SplitReturn.Observation.ExternalStop m.history check wait := by
            refine .abort ?_
            intro obs site
            rw [s.sharedState.observed] at site
            obtain ⟨q,found,rfl⟩ := Option.map_eq_some_iff.mp site
            obtain ⟨logged,retest,next⟩ := s.sharedState.owner q found
            have lower := (s.sharedState.coherent r mem).writeRead s.sharedState.era.begin (s.abortStart r ar)
            refine .seen r s.sharedState.era.final.val
              ⟨dep.keeps localRead,retest,s.sharedState.readRead r mem q logged,next,
               (s.sharedState.coherent q logged).readWrite,?_⟩
              ((SearchAbort.stopped_iff _).2 (beq_iff_eq.mp stopped))
            intro i after upto
            exact ⟨⟨i.val-1,by have := i.isLt; omega⟩,by simp; omega,s.sharedState.era.rmw i (by omega) upto⟩
          exact SplitReturn.Observation.cancellationGateContract_proven m.history check wait ext
        · exact SplitReturn.Observation.cancellationGateContract_proven m.history check wait ext
      | some h =>
        have hot : NativeStopReturns.helperBit m poll.worker h = some true := by
          simp only [NativeStopReturns.poll,NativeStopReturns.ChainRead.set,unlatched,ar,hr] at yes
          split at yes <;> simp_all
        obtain ⟨clear,seen⟩ := s.helper h hr
        unfold NativeStopReturns.helperBit at hot
        rw [SplitReturn.Observation.helperJobReadContract_proven _ _ _ clear seen] at hot
        contradiction

/-- Contract 2 (Unified single-tape split replay):
    An execution accepted by `replayFromTape` derives from ONE common tape `es`:
    (1) `best_score` and `best_move` modification orders matching `.ownerStore`,
    (2) `ActivePointer.splitDispatch` provenance for every dispatch,
    (3) full mask, flag, epoch, and abort memory coherence, and
    (4) the `AlphaOrigin.Capture` certificate without any opaque poll oracle. -/
def SingleTapeReplayContract : Prop :=
  ∀ {Operand : Type} {rc : List Move.Move} {pv : Bool} {beta : Int}
    (t : NativeSplit.Tickets rc pv beta) (ext : Extracted) (b : AlphaOrigin.SetupOperands t)
    (sameMem : b.memory = ext.memory) (root cut : Bool) (depth : Nat)
    (es : Joint.Tape Operand) (lockInit : AlphaReplay.Initialization) (spec : PollSpec)
    (afterSplit : Event) (ownerAbort : Option Event) (helperJob : Bool)
    (before : Option (Event → Event → Bool))
    (cap : AlphaOrigin.Capture t root cut depth (fun _ _ _ _ _ => True)),
    replayFromTape ext b sameMem root cut depth es lockInit spec afterSplit ownerAbort helperJob before = some cap →
    bestStores b.generation.owner.val b.bankIndex es =
      b.owner.bPre ++ t.final.best ++ subsequent b.owner.exit (b.generation.owner,b.bankIndex) .score b.owner.later ∧
    moveStores b.generation.owner.val b.bankIndex es =
      b.owner.mPre ++ t.final.moves ++ subsequent b.owner.exit (b.generation.owner,b.bankIndex) .move b.owner.later ∧
    CutoffPollReplay.FlagMemory b.memory.history ∧
    CutoffPollReplay.EpochMemory b.memory.history ∧
    AbortMemory b.memory ∧
    ∀ d ∈ t.dispatches, ∃ sd : ActivePointer.SplitDispatch b.memory.history (CutoffPollReplay.slots b) es (CutoffPollReplay.generation b) d.entry,
      ActivePointer.splitDispatch (sameMem ▸ ext.chronology) (CutoffPollReplay.slots b) es (CutoffPollReplay.generation b) d.entry = some sd ∧
      sd.origin.generation = CutoffPollReplay.generation b ∧
      SequencedBefore sd.origin.drop d.entry

theorem singleTapeReplayContract_proven : SingleTapeReplayContract := by
  intro _ _ _ _ t ext b sameMem root cut depth es lockInit spec afterSplit ownerAbort helperJob before cap ok
  unfold replayFromTape at ok
  by_cases hStores : ownerStoresMatch b.owner es = true
  · simp only [hStores,guard,↓reduceIte,Bind.bind,Option.bind,Pure.pure] at ok
    by_cases hMask : MaskGeneration.memoryOK (sameMem ▸ ext.chronology) b.generation.owner
        (MaskGeneration.rows b.generation.owner b.bankIndex es) = true
    · simp only [hMask,↓reduceIte] at ok
      by_cases hMem : (CutoffPollReplay.flagMemoryOK (sameMem ▸ ext.chronology) &&
          CutoffPollReplay.epochMemoryOK (sameMem ▸ ext.chronology) &&
          abortMemoryOK b.memory (sameMem ▸ ext.chronology)) = true
      · simp only [hMem,↓reduceIte] at ok
        cases hJ : TaskAncestry.capture (CutoffPollReplay.slots b) es with
        | none => simp [hJ] at ok
        | some j =>
          simp only [hJ] at ok
          cases hChk : decodeOwnerCheck b.memory j afterSplit ownerAbort helperJob with
          | none => simp [hChk] at ok
          | some check =>
            simp only [hChk] at ok
            by_cases hMatch : (check.ancestors.event == b.owner.check.ancestors.event &&
                check.ancestors.chain == b.owner.check.ancestors.chain &&
                check.stopped == b.owner.check.stopped) = true
            · simp only [hMatch,↓reduceIte] at ok
              simp only [ownerStoresMatch,Bool.and_eq_true,beq_iff_eq] at hStores
              simp only [Bool.and_eq_true] at hMem
              refine ⟨hStores.1,hStores.2,
                (CutoffPollReplay.checkedFlags _ _).mp hMem.1.1,
                (CutoffPollReplay.checkedEpochs _ _).mp hMem.1.2,
                (checkedAbort _ _).mp hMem.2,?_⟩
              intro d mem
              obtain ⟨sd,found,gen,_,after⟩ := CutoffPollReplay.fromTape_dispatches t b root cut depth es _ before cap ok d mem
              exact ⟨sd,found,gen,after⟩
            · simp [hMatch,failure] at ok
      · simp [hMem,failure] at ok
    · simp [hMask,failure] at ok
  · simp [hStores,guard,failure] at ok

end NeuralReversi.PvsParallel.Interleaving.SplitTapeReplay

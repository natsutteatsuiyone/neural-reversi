module
public import NeuralReversi.PvsParallel.NativeStopReturns.Program
public import NeuralReversi.Threading.CutoffVisibility.StopSource
public import NeuralReversi.Threading.SplitReturn.Observation

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeStopReturns

open Threading
open CutoffVisibility (FlagRead History)
open AtomicWord (Event Read Write SequencedBefore)
open SplitReturn.Observation (AncestorScan AncestorRead)

structure Memory where
  history : History
  abort : List Write
  abortReads : List (Read abort)
  helper : Nat → List Write
  helperReads : (w : Nat) → List (Read (helper w))

def InWorker (atPoll e : Event) := e.worker = atPoll.worker ∧ e.step ≤ atPoll.step

inductive ChainRead (m : Memory) (atPoll : Event) : Type
  | cached (chain : List Nat) (cache : SplitCutoff.Cache) (reads : List (FlagRead m.history.flags))
      (source : CutoffVisibility.StopSource.Execution m.history chain cache reads)
      (atWorker : ∀ r ∈ reads, InWorker atPoll r.event)
  | postJoin (scan : AncestorScan m.history) (atWorker : ∀ r ∈ scan.reads, InWorker atPoll r.event)

def ChainRead.mode : ChainRead m t → Mode
  | .cached .. => .cached
  | .postJoin .. => .postJoin

def ChainRead.chain : ChainRead m t → List Nat
  | .cached chain .. => chain
  | .postJoin scan _ => scan.chain

def ChainRead.set : ChainRead m t → Bool
  | .cached _ cache .. => cache.latched
  | .postJoin scan _ => scan.stopped

def ChainRead.reads : ChainRead m t → List (FlagRead m.history.flags)
  | .cached _ _ rs .. => rs
  | .postJoin scan _ => scan.reads

def ChainRead.within (start : Nat) : ChainRead m t → Bool
  | .cached .. => true
  | .postJoin scan _ => decide (t.worker = scan.event.worker ∧ start ≤ scan.event.step ∧ scan.event.step ≤ t.step)

theorem ChainRead.flag (p : ChainRead m t) (set : p.set = true) :
    ∃ r ∈ p.reads, r ∈ m.history.flagReads ∧ r.slot ∈ p.chain ∧ InWorker t r.event ∧ r.value = true := by
  cases p with
  | cached chain cache reads source atWorker =>
    obtain ⟨r,member,recorded,inChain,positive⟩ := CutoffVisibility.StopSource.positive source set
    exact ⟨r,member,recorded,inChain,atWorker r member,positive⟩
  | postJoin scan atWorker =>
    obtain ⟨_,member,rfl⟩ := List.any_eq_true.mp set
    obtain ⟨r,inReads,val⟩ := List.mem_map.mp member
    exact ⟨r,inReads,(scan.afterStart r inReads).1,
      CutoffVisibility.StopSource.Walk.slot_of_read scan.walk inReads,atWorker r inReads,val⟩

structure Loads (m : Memory) (t : Event) where
  started : Nat
  abort : Option (Read m.abort)
  helper : Option (Read (m.helper t.worker))
  interval : started ≤ t.step
  abortRecorded : ∀ r, abort = some r → r ∈ m.abortReads ∧ r.acquire = true ∧ InWorker t r.event ∧ started ≤ r.event.step
  helperRecorded : ∀ r, helper = some r → r ∈ m.helperReads t.worker ∧ InWorker t r.event ∧ started ≤ r.event.step

def abortBit (m : Memory) (r : Read m.abort) : Bool :=
  let word := (m.abort.get r.source).value
  SearchAbort.generation word == SearchAbort.requested word

-- Only 0 and 1 are possible for a captured Rust AtomicBool value.
def helperBit (m : Memory) (w : Nat) (r : Read (m.helper w)) : Option Bool :=
  match ((m.helper w).get r.source).value.toNat with
  | 0 => some false
  | 1 => some true
  | _ => none

def poll (chain : ChainRead m t) (loads : Loads m t) : Option Bool :=
  match chain.set, loads.abort, loads.helper with
  | true, none, none => some true
  | false, some r, none => if abortBit m r then some true else none
  | false, some r, some h =>
    if !abortBit m r && (r.event.worker == h.event.worker && decide (r.event.step < h.event.step)) then helperBit m t.worker h else none
  | _, _, _ => none

inductive Cause (m : Memory) (t : Event) (chain : List Nat) : Prop
  | cutoff (r : FlagRead m.history.flags) : r ∈ m.history.flagReads → r.slot ∈ chain →
      InWorker t r.event → r.value = true → Cause m t chain
  | abort (r : Read m.abort) : r ∈ m.abortReads → r.acquire = true → InWorker t r.event →
      SearchAbort.Stopped (m.abort.get r.source).value → Cause m t chain
  | helper (r : Read (m.helper t.worker)) : r ∈ m.helperReads t.worker → InWorker t r.event →
      ((m.helper t.worker).get r.source).value.toNat = 1 → Cause m t chain

open Classical in
theorem cause_of_poll (c : ChainRead m t) (loads : Loads m t) (yes : poll c loads = some true) :
    Cause m t c.chain := by
  cases hc : c.set with
  | true =>
    obtain ⟨r,_,mem,slot,localRead,value⟩ := c.flag hc
    exact .cutoff r mem slot localRead value
  | false =>
    cases ha : loads.abort with
    | none => simp [poll,hc,ha] at yes
    | some r =>
      obtain ⟨recorded,acq,localRead,_⟩ := loads.abortRecorded _ ha
      cases hh : loads.helper with
      | none =>
        have stopped : abortBit m r = true := by simpa [poll,hc,ha,hh] using yes
        exact .abort r recorded acq localRead ((SearchAbort.stopped_iff _).2 (beq_iff_eq.mp stopped))
      | some h =>
        have yes' : helperBit m t.worker h = some true := by
          simp only [poll,hc,ha,hh] at yes
          split at yes <;> simp_all
        obtain ⟨recorded,localRead,_⟩ := loads.helperRecorded _ hh
        apply Cause.helper h recorded localRead
        unfold helperBit at yes'
        split at yes' <;> simp_all

/-- After its owner restores the parent split point, this call scans the actual retained
    ancestor flags regardless of the local epoch or latch of the preceding assignment. -/
theorem retained_postJoin {m : Memory} {old : ChainRead m earlier} (stopped : old.set = true)
    (scan : AncestorScan m.history) (atWorker : ∀ r ∈ scan.reads, InWorker now r.event)
    (coherent : CutoffVisibility.Coherent m.history)
    (kept : ∀ r ∈ old.reads, r ∈ m.history.flagReads → r.slot ∈ old.chain → InWorker earlier r.event → r.value = true →
      ∃ wait, AncestorRead m.history r scan wait)
    (loads : Loads m now) (shortCircuit : loads.abort = none ∧ loads.helper = none) :
    poll (.postJoin scan atWorker) loads = some true := by
  obtain ⟨r,member,recorded,inside,localRead,seen⟩ := old.flag stopped
  obtain ⟨wait,matchRead⟩ := kept r member recorded inside localRead seen
  have found := SplitReturn.Observation.ancestorStopContract_proven m.history coherent r scan wait matchRead
  simp [poll,ChainRead.set,found,shortCircuit.1,shortCircuit.2]

end NeuralReversi.PvsParallel.NativeStopReturns

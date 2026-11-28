module
public import NeuralReversi.PvsParallel.IteratorCutoff
public import NeuralReversi.Threading.CutoffVisibility.StopSource

@[expose] public section

namespace NeuralReversi.PvsParallel.SplitWorkerMir.Stop

open Threading
open CutoffVisibility (History FlagRead FlagWrite Coherent HappensBefore)
open SplitReturn.Observation (ExternalStop OwnerCheck AncestorRead AbortCause)
open Iterator (LockedCall)
open AtomicWord (Event)

def mark (time : Nat → Event) (c : LockedCall) : Option FlagWrite := do
  let body ← search c.ticket.input
  if (commit c.ticket.input c.ticket.locked body.chosen).cut then some ⟨time c.order,true⟩ else none

private theorem marked_exits (h : (commit i l v).cut = true) : (commit i l v).exit = .cutoff := by
  rw [commit_eq] at h ⊢
  split at h <;> try contradiction
  rename_i running
  rw [ite_eq_right running]
  change (if _ then Exit.cutoff else Exit.again) = Exit.cutoff
  rw [ite_eq_left h]

/-- A positive load of the current slot can also come from a worker that finished
    its own search and marked this split before its sibling polled. -/
structure Current (hist : History) (calls : List LockedCall) where
  read : FlagRead hist.flags
  member : read ∈ hist.flagReads
  seen : read.value = true
  coherent : Coherent hist
  time : Nat → Event
  pre : List FlagWrite
  clear : Event
  post : List FlagWrite
  words : hist.flags read.slot = pre ++ (⟨clear,false⟩ :: calls.filterMap (mark time)) ++ post
  initialized : HappensBefore hist clear read.event
  beforeReuse : ∀ i : Fin (hist.flags read.slot).length,
    i.val = pre.length + 1 + (calls.filterMap (mark time)).length →
      HappensBefore hist read.event ((hist.flags read.slot).get i).event

theorem Current.cause (read : Current hist calls) :
    ∃ c ∈ calls, ∃ b, search c.ticket.input = some b ∧ (commit c.ticket.input c.ticket.locked b.chosen).exit = .cutoff := by
  cases entries : calls.filterMap (mark read.time) with
  | cons w tail =>
    have mem : w ∈ calls.filterMap (mark read.time) := by simp [entries]
    obtain ⟨c,existsHere,hit⟩ := List.mem_filterMap.mp mem
    obtain ⟨b,searched,written⟩ := Option.bind_eq_some_iff.mp hit
    split at written <;> try contradiction
    exact ⟨c,existsHere,b,searched,marked_exits (by assumption)⟩
  | nil =>
    have words := read.words
    rw [entries] at words
    let init : Fin (hist.flags read.read.slot).length := ⟨read.pre.length,by rw [words]; simp⟩
    have seed : (hist.flags read.read.slot).get init = ⟨read.clear,false⟩ := by
      have found : (hist.flags read.read.slot)[read.pre.length]? = some ⟨read.clear,false⟩ := by
        rw [words]; simp
      exact (List.getElem?_eq_some_iff.mp found).2
    have lower := read.coherent.flagWriteRead read.read read.member init (seed ▸ read.initialized)
    have eq : read.read.source = init := by
      apply Fin.ext
      change read.read.source.val = read.pre.length
      by_cases later : read.pre.length < read.read.source.val
      · have within := read.read.source.isLt
        let next : Fin (hist.flags read.read.slot).length := ⟨read.pre.length+1,by omega⟩
        have upper := read.coherent.flagReadWrite read.read read.member next (read.beforeReuse next (by simp [entries,next]))
        change read.read.source.val < read.pre.length+1 at upper
        omega
      · exact Nat.le_antisymm (Nat.le_of_not_gt later) lower
    have seen := read.seen
    unfold FlagRead.value at seen
    rw [eq,seed] at seen
    contradiction

inductive Origin (hist : History) (owner : OwnerCheck hist) (wait : Event) (calls : List LockedCall) : Type where
  | current (sample : Current hist calls)
  | ancestor (flag : FlagRead hist.flags) (coherent : Coherent hist) (sample : AncestorRead hist flag owner.ancestors wait)
  | abort (cause : ∀ obs, owner.shared = some obs → AbortCause hist obs wait)

structure Poll {hist : History} (owner : OwnerCheck hist) (wait : Event) (calls : List LockedCall) where
  slot : Nat
  parents : List Nat
  cache : SplitCutoff.Cache
  samples : List (FlagRead hist.flags)
  trace : CutoffVisibility.StopSource.Execution hist (slot :: parents) cache samples
  result : cache.latched = true
  coherent : Coherent hist
  currentRead : ∀ r ∈ samples, r ∈ hist.flagReads → r.slot = slot → r.value = true →
    {current : Current hist calls // current.read = r}
  parentRead : ∀ r ∈ samples, r ∈ hist.flagReads → r.slot ∈ parents → r.value = true →
    AncestorRead hist r owner.ancestors wait

theorem Poll.origin {hist : History} {owner : OwnerCheck hist} (p : Poll owner wait calls) : Nonempty (Origin hist owner wait calls) := by
  obtain ⟨r,sampled,physical,slot,on⟩ := CutoffVisibility.StopSource.active_or_ancestor p.trace p.result
  rcases slot with current | parent
  · exact ⟨.current (p.currentRead r sampled physical current on).val⟩
  · exact ⟨.ancestor r p.coherent (p.parentRead r sampled physical parent on)⟩

theorem Origin.explains (why : Origin hist owner wait (IteratorFields.chronological ops journal stream workers))
    (a : FieldStores.low32 aStore = initial.alpha) (b : FieldStores.low32 bStore = initial.best)
    (m : FieldStores.low8 mStore = encode initial.picked)
    (ok : IteratorFields.replay rc ops journal stream workers traffic pv beta encode
      (FieldStores.initialLog initial aStore bStore mStore) = some final) :
    SplitReturn.LocalCutoff pv beta initial.state (Iterator.lockReports (IteratorFields.results ops journal stream) workers) ∨
      ExternalStop hist owner wait := by
  cases why with
  | current sample =>
    obtain ⟨_,member,_,searched,exit⟩ := sample.cause
    exact .inl (IteratorFields.call_cutoff a b m ok member searched exit)
  | ancestor flag coh sample => exact .inr (.ancestor flag coh sample)
  | abort cause => exact .inr (.abort cause)

private theorem stopped_field (i : Input) (s : Locked) (v : Word) (exit : (commit i s v).exit = .stopped) :
    s.stopped = true := by
  rw [commit_eq] at exit
  split at exit
  · assumption
  · change (if _ then Exit.cutoff else Exit.again) = Exit.stopped at exit
    split at exit <;> contradiction

private theorem retired_stop (xs : List LockedCall) (run : Iterator.retire xs = some result)
    (finished : result.completed.departure = .cancelled) : ∃ c ∈ xs, c.ticket.locked.stopped = true := by
  induction xs generalizing result with
  | nil => cases run; contradiction
  | cons c rest ih =>
    unfold Iterator.retire at run
    obtain ⟨body,_,more⟩ := Option.bind_eq_some_iff.mp run
    dsimp only at more
    cases exit : (commit c.ticket.input c.ticket.locked body.chosen).exit with
    | stopped => exact ⟨c,by simp,stopped_field _ _ _ exit⟩
    | cutoff => cases rest <;> simp [exit] at more; subst result; contradiction
    | again =>
      rw [exit] at more
      obtain ⟨_,done,same⟩ := Option.bind_eq_some_iff.mp more
      cases same
      obtain ⟨actual,member,stop⟩ := ih done finished
      exact ⟨actual,List.mem_cons_of_mem _ member,stop⟩

theorem worker_stop (ok : IteratorFields.replay rc ops journal stream workers traffic pv beta encode initial = some final)
    (i : SplitCompletion.Worker) (here : i ∈ workers) (finished : (IteratorFields.results ops journal stream i).completed.departure = .cancelled) :
    ∃ c ∈ IteratorFields.chronological ops journal stream workers, c.ticket.locked.stopped = true := by
  obtain ⟨cs,done,every⟩ := IteratorFields.worker_calls (IteratorFields.worker_returns ok i here) here
  obtain ⟨c,member,stop⟩ := retired_stop cs done finished
  exact ⟨c,every c member,stop⟩

end NeuralReversi.PvsParallel.SplitWorkerMir.Stop

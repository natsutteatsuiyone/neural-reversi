module
public import NeuralReversi.Threading.CutoffVisibility

@[expose] public section

namespace NeuralReversi.Threading.CutoffVisibility.StopSource

open SplitCutoff (Cache Epoch reset poll)

theorem Walk.slot_of_read (walk : CutoffVisibility.Walk flags chain reads) (member : r ∈ reads) : r.slot ∈ chain := by
  induction walk with
  | empty => contradiction
  | found => simp only [List.mem_singleton] at member; subst r; simp
  | descend _ _ _ ih =>
    rcases List.mem_cons.mp member with rfl | tail
    · exact List.mem_cons_self ..
    · exact List.mem_cons_of_mem _ (ih tail)

/-- `split_point_cutoff_occurred` may return immediately from the local latch
    or reuse an equal epoch. Neither branch reads a split-point flag. -/
inductive Execution (history : History) : List Nat → Cache → List (FlagRead history.flags) → Prop
  | reset (chain : List Nat) (word : Epoch) : Execution history chain (reset word) []
  | cached : Execution history chain cache reads → cache.latched = true → Execution history chain cache reads
  | hit : Execution history chain cache reads → cache.latched = false →
      (load : EpochRead history.epochs) → load ∈ history.epochReads → load.acquire = true →
      (history.epochs.get load.source).value = cache.seen → Execution history chain cache reads
  | miss : Execution history chain cache prior → cache.latched = false →
      (attempt : PollRead history cache) → attempt.flags.observation.epoch ≠ cache.seen → attempt.flags.chain = chain →
      Execution history chain (poll cache attempt.flags.observation) (prior ++ attempt.flags.reads)
  | otherClear : Execution history chain cache prior → cache.latched = false →
      (attempt : PollRead history cache) → attempt.flags.observation.epoch ≠ cache.seen →
      attempt.flags.observation.chain.any id = false →
      Execution history chain ⟨attempt.flags.observation.epoch,false⟩ prior
  | afterSplit : Execution history chain cache prior →
      (walk : CutoffVisibility.Walk history.flags chain reads) →
      (∀ r ∈ reads, r ∈ history.flagReads) →
      Execution history chain ⟨cache.seen,cache.latched || (reads.map FlagRead.value).any id⟩ (prior ++ reads)

private theorem observed {history : History} {reads : List (FlagRead history.flags)} (walk : CutoffVisibility.Walk history.flags chain reads)
    (physical : ∀ r ∈ reads, r ∈ history.flagReads) (found : true ∈ reads.map FlagRead.value) :
    ∃ r ∈ reads, r ∈ history.flagReads ∧ r.slot ∈ chain ∧ r.value = true := by
  obtain ⟨r,member,value⟩ := List.mem_map.mp found
  exact ⟨r,member,physical r member,Walk.slot_of_read walk member,value⟩

theorem positive {h : History} {reads : List (FlagRead h.flags)} (trace : Execution h chain cache reads) (stop : cache.latched = true) :
    ∃ r ∈ reads, r ∈ h.flagReads ∧ r.slot ∈ chain ∧ r.value = true := by
  induction trace with
  | reset => contradiction
  | cached _ _ ih => exact ih stop
  | hit _ _ _ _ _ _ ih => exact ih stop
  | otherClear => contradiction
  | miss _ clear attempt changed chain _ =>
    have contains := ((SplitCutoff.pollContract_proven _ _).2.2 clear changed).2.mp stop
    obtain ⟨r,found,m1,m2,value⟩ := observed (attempt.walkOnMiss changed)
      (fun r m => (attempt.flags.afterLoad r m).1) contains
    exact ⟨r,List.mem_append_right _ found,m1,chain ▸ m2,value⟩
  | afterSplit _ walk live ih =>
    rcases Bool.or_eq_true_iff.mp stop with kept | marked
    · obtain ⟨r,found,physical,inside,value⟩ := ih kept
      exact ⟨r,List.mem_append_left _ found,physical,inside,value⟩
    · obtain ⟨v,member,value⟩ := List.any_eq_true.mp marked
      obtain ⟨r,found,physical,inside,val⟩ := observed walk live (value ▸ member)
      exact ⟨r,List.mem_append_right _ found,physical,inside,val⟩

theorem active_or_ancestor {h : History} {reads : List (FlagRead h.flags)}
    (trace : Execution h (slot :: parents) cache reads) (stop : cache.latched = true) :
    ∃ r ∈ reads, r ∈ h.flagReads ∧ (r.slot = slot ∨ r.slot ∈ parents) ∧ r.value = true := by
  obtain ⟨r,found,physical,member,value⟩ := positive trace stop
  exact ⟨r,found,physical,List.mem_cons.mp member,value⟩

end NeuralReversi.Threading.CutoffVisibility.StopSource

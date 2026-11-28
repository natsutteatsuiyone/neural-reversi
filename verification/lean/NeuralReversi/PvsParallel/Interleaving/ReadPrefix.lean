module
public import NeuralReversi.PvsParallel.Interleaving.AtomicReadback
public import NeuralReversi.Threading.SplitCompletion

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.ReadPrefix

open NativeTableWrites.Journal (Calls Record Site mods)
open TranspositionTable.SeqLock (Location)

/-- A synchronized boundary separates this read from the first subsequent
    modification of its atomic object. A later modification need not exist. -/
def Separated (x : NativeTableWrites.Journal.Execution c endian (pre ++ post))
    (a : FrameTableReads.Address) (l : Location) (r : Nat) :=
  ∀ s, (mods c post a l).head? = some s → (x.memory a).hb r (s.event c)

/-- All possible source ranks of a read lie in the prefix if the first later
    modification is ordered after it. A prefix closed only under happens-before
    can omit the source of an unordered relaxed read. -/
def Contract := ∀ {c endian pre post} (x : NativeTableWrites.Journal.Execution c endian (pre ++ post))
    (a : FrameTableReads.Address) (l : Location) (r i : Nat),
    AtomicReadback.ReadWrite x → Separated x a l r →
    (x.memory a).rf ((x.memory a).write l i) r →
    i ≤ (mods c pre a l).length ∧ (i = 0 ∨
      ∃ s, (mods c pre a l)[i-1]? = some s ∧ (x.memory a).write l i = s.event c)

theorem closed : Contract := by
  intro c endian pre post x a l r i coherent separates rf
  have append : mods c (pre ++ post) a l = mods c pre a l ++ mods c post a l :=
    List.filterMap_append
  have bound : i ≤ (mods c pre a l).length := by
    cases h : mods c post a l with
    | nil => simpa [append,h] using x.source a l i r rf
    | cons first rest =>
      have atFirst : (mods c (pre ++ post) a l)[(mods c pre a l).length]? = some first := by
        simp [append,h]
      exact Nat.le_of_lt_succ (coherent a l r i _ first atFirst rf (separates first (by simp [h])))
  refine ⟨bound,?_⟩
  cases i with
  | zero => exact Or.inl rfl
  | succ j =>
    have inside : j < (mods c pre a l).length := bound
    have old := List.getElem?_eq_getElem inside
    have extended := (List.getElem?_append_left (l₂ := mods c post a l) inside).trans old
    rw [←append] at extended
    exact Or.inr ⟨_,old,(x.modified _ _ _ _ extended).1⟩

/-- Every nested helper releases its participation before the owner finishes
    the enclosing search. When the next search starts after that handoff, this
    gives the required source bound for a physical TT read by the helper. -/
theorem afterRetirement (x : NativeTableWrites.Journal.Execution c endian (pre ++ post))
    (a : FrameTableReads.Address) (loc : Location) (coherent : AtomicReadback.ReadWrite x)
    (hist : Threading.CutoffVisibility.History) (stamp : Threading.AtomicWord.Event → Nat)
    (physical : ∀ {u v}, Threading.CutoffVisibility.HappensBefore hist u v → (x.memory a).hb (stamp u) (stamp v))
    {read wait finish : Threading.AtomicWord.Event}
    (kept : Threading.SplitCompletion.RetainedThrough hist read wait)
    (done_ : Threading.AtomicWord.SequencedBefore wait finish)
    (following : ∀ s, (mods c post a loc).head? = some s → (x.memory a).hb (stamp finish) (s.event c))
    (rf : (x.memory a).rf ((x.memory a).write loc i) (stamp read)) :
    i ≤ (mods c pre a loc).length ∧ (i = 0 ∨
      ∃ s, (mods c pre a loc)[i-1]? = some s ∧ (x.memory a).write loc i = s.event c) := by
  have released := Threading.SplitCompletion.retainedAccessContract_proven hist read wait finish kept done_
  exact closed x a loc _ i coherent (fun s h => (x.memory a).hb_trans (physical released) (following s h)) rf

end NeuralReversi.PvsParallel.Interleaving.ReadPrefix

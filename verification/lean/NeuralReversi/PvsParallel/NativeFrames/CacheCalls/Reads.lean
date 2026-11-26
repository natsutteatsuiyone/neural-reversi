import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Program

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls

open EndgameSearch.CacheHistory (Pool Local)
open EndgameSearch.CacheHistory.Payloads

private theorem observed_length (p : Pool) (ops : List Instruction) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) :
    (observeRun p s shape ops).length = ops.length := by
  induction ops generalizing s with
  | nil => rfl
  | cons op rest ih => simp [observeRun,ih]

private theorem observed_split (p : Pool) (xs ys : List Instruction) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) :
    let after := xs.foldl (fun st op => op.step p st) s
    ∃ sizes : ∀ t, (after.cells t).size = p.sizes t,
      observeRun p s shape (xs ++ ys) = observeRun p s shape xs ++ observeRun p after sizes ys := by
  induction xs generalizing s with
  | nil => exact ⟨shape,rfl⟩
  | cons op rest ih =>
    obtain ⟨sizes,split⟩ := ih (op.step p s) (op.shape p s shape)
    exact ⟨sizes,congrArg (_ :: ·) split⟩

/-- At every individual probe, the observed board and both bound bytes come
    from the entry selected by all prior stores, and the returned word is exactly
    the Option consumed by the recursive MIR branch. This also covers a miss. -/
theorem Program.probe_at (code : Program node s.history last) (sound : EndgameTreeCache.soundHistory mc fc last)
    (p : Pool) (shape : ∀ t, (s.cells t).size = p.sizes t)
    (arrays : EndgameSearch.CacheState.represents p.address s.history s.cells)
    (before after_ : List Access) (path : code.accesses = before ++ Access.probe t b a safe :: after_) :
    let logical := EndgameTreeCache.entryFor p.address (s.history ++ before.filterMap Access.store?) t b
    let physical := observeRun p s shape (code.accesses.map Access.instruction)
    ∃ raw, physical[before.length]? = some ⟨t,.probe (p.address t b) b (BitVec.ofInt 32 a) (EndgameSearch.CacheHistory.Payloads.words logical) raw⟩ ∧
      raw.map BitVec.toInt = EndgameCache.probe logical b a := by
  let prefixOps := before.map Access.instruction
  let atRead := prefixOps.foldl (fun st op => op.step p st) s
  have ran := execute_run p prefixOps s shape
  obtain ⟨sizes,aligned⟩ := aligned_ops p _ shape arrays ran
  have original := code.words_of_mem sound before (fun x hx => path ▸ List.mem_append_left _ hx)
  have history : atRead.history = s.history ++ before.filterMap Access.store? :=
    (trace_writes p _ shape ran).1.trans (congrArg (s.history ++ ·) ((observe_writes p prefixOps s shape).trans original))
  obtain ⟨raw,seen,used⟩ := observe_probe_words p atRead sizes aligned t b (BitVec.ofInt 32 a)
  have signed : (BitVec.ofInt 32 a).toInt = a := BitVec.toInt_ofInt_eq_self (by decide) (by have := safe.1; omega) (by have := safe.2; omega)
  rw [history,signed] at used
  refine ⟨raw,?_,used⟩
  rw [path,List.map_append]
  obtain ⟨_,joined⟩ := observed_split p prefixOps ((Access.probe t b a safe :: after_).map Access.instruction) s shape
  rw [joined,List.getElem?_append_right (by simp [observed_length,prefixOps]),observed_length]
  simpa [prefixOps,observeRun,Access.instruction,history] using congrArg some seen

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls

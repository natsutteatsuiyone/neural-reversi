import NeuralReversi.EndgameSearch.CacheHistory.Payloads.Instructions

namespace NeuralReversi.EndgameSearch.CacheHistory.Payloads

open EndgameTreeCache (Store Table entryFor)
open PvsBoardTree (legalSquares)

def lookupOps (p : Pool) (i : Input) (t : Table) : List Instruction :=
  [.slot t (p.key i.board.val), .probe t i.board.val (ScoreRefresh.toDiscDiff i.alpha).raw]

/-- Any prefix of the slot/probe pair can be audited before the search returns.
    The recursive program may still add writes after that prefix. -/
theorem check_lookup_prefix (p : Pool) (i : Input) (t : Table)
    (g : Good mc fc p.width p.sizes p.key s) (n : Nat) :
    let ops := (lookupOps p i t).take n
    check p false s (i.run mc fc p.address s.history).history (observeRun p s g.shape ops) =
      some ⟨observeRun p s g.shape ops,s⟩ := by
  obtain ⟨ws,grows,_⟩ := (after_call i g (CacheMir.budget_fits p.size p.width p.word)).2
  change s.history ++ ws = (i.run mc fc p.address s.history).history at grows
  cases n with
  | zero => simpa [← grows,lookupOps,List.take,Instruction.step,Instruction.write?] using check_run p [] s g.shape false (s.history ++ ws) (by simp)
  | succ n =>
    cases n with
    | zero =>
      simpa [← grows,lookupOps,List.take,Instruction.step,Instruction.write?] using check_run p [.slot t (p.key i.board.val)] s g.shape false (s.history ++ ws)
        (by simp [List.filterMap_cons,Instruction.write?])
    | succ n =>
      simpa [← grows,lookupOps,List.take,Instruction.step,Instruction.write?] using check_run p (lookupOps p i t) s g.shape false (s.history ++ ws)
        (by simp [lookupOps,List.filterMap_cons,Instruction.write?])

def recursiveArm (i : Input) : Table → Prop
  | .shallow => i.arm = .shallow
  | .deep => ∃ poll, i.arm = .deep poll

/-- On a legal-move node a matching physical bound returns before the recursive
    move loop. This holds for either poller and for a discarded deep score. -/
theorem check_lookup_hit (p : Pool) (i : Input) (t : Table)
    (g : Good mc fc p.width p.sizes p.key s) (branch : recursiveArm i t)
    (moves : legalSquares mc i.board.val ≠ [])
    (loaded : CacheMir.probe (s.cells t) (CacheMir.slot p.width (s.cells t).size (p.key i.board.val)) i.board.val
       (ScoreRefresh.toDiscDiff i.alpha).raw.toInt = some (some value)) :
    let ops := observeRun p s g.shape (lookupOps p i t)
    check p true s (i.run mc fc p.address s.history).history ops = some ⟨ops,s⟩ ∧
      EndgameCache.CacheRange value ∧ Solve2.NWS (ScoreRefresh.toDiscDiff i.alpha).raw.toInt
        (EndgameTreeCache.discValue mc fc i.board.val) value := by
  have cached := Option.some.inj ((g.reads t i.board.val _).symm.trans loaded)
  change EndgameCache.probe (entryFor p.address s.history t i.board.val) i.board.val _ = some value at cached
  have same : (i.run mc fc p.address s.history).history = s.history := by
    cases t with
    | shallow =>
      change i.arm = .shallow at branch
      simp only [Input.run,branch,Input.eval]
      rw [ShallowMir.run.eq_def]
      split
      · contradiction
      · simp [ShallowMir.node,cached]
    | deep =>
      obtain ⟨poll,arm⟩ := branch
      simp only [Input.run,arm,Input.eval]
      rw [DeepRecursionMir.run.eq_def]
      split
      · contradiction
      · cases poll <;> simp [DeepRecursionMir.early,PvsParallel.DeepLeafMir.pollInput,Input.state,cached]
  refine ⟨?_,CacheMir.probe_bound t i.board (g.shape t) g.arrays g.valid loaded⟩
  rw [same]
  exact check_run p (lookupOps p i t) s g.shape true s.history
    (by simp [lookupOps,List.filterMap_cons,Instruction.write?])

end NeuralReversi.EndgameSearch.CacheHistory.Payloads

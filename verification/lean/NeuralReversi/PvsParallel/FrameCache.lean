import NeuralReversi.EndgameSearch.CacheHistory
import NeuralReversi.PvsParallel.FrameReturns

namespace NeuralReversi.PvsParallel.FrameReturns

open EndgameSearch.CacheHistory (Pool Visit Timeline)

/-- The block identifier and operands are read from the same worker's boundary invocation.
    Its input history and array state are reconstructed from the preceding pool lifetime. -/
structure CacheSite {p : Pool} (trace : Timeline p) (f : Frame) (depth : TranspositionTable.Depth)
    (a z : ScoreRefresh.Score) where
  visit : Visit p
  member : visit ∈ trace.visits
  worker : visit.worker = f.worker
  frame : visit.frame = f.id
  published : visit.scoredAt = some f.scoreStep
  board : visit.input.board = f.position
  atDepth : visit.input.depth = depth
  alpha : visit.input.alpha = a
  beta : visit.input.beta = z
  exactTag : visit.input.selectivity = Cutoffs.exactSelectivity
  nonPv : f.pv = false
  boundary : visit.input.empties.count.toNat ≤ 12

theorem CacheSite.arrays {p : Pool} {trace : Timeline p} (site : CacheSite trace f depth a z)
    (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) :
    let s := trace.readAt mc fc f.worker site.visit.first
    EndgameTreeCache.soundHistory mc fc s.history ∧ EndgameSearch.RecursionCacheMir.Reads p.width p.key p.address s.history s.cells := by
  exact trace.reads mc fc f.worker site.visit.first

theorem CacheSite.physical {p : Pool} {trace : Timeline p} (site : CacheSite trace f depth a z)
    (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) :
    let prior := trace.readAt mc fc f.worker site.visit.first
    let next := site.visit.input.after mc fc p.address prior
    ∃ previous later ws, trace.visits = previous ++ site.visit :: later ∧
      next = EndgameSearch.CacheHistory.replay mc fc p
        (EndgameSearch.CacheHistory.byOwner (previous ++ [site.visit]) f.worker) ∧
      prior.history ++ ws = next.history ∧
      EndgameSearch.RecursionCacheMir.Steps p.width p.key p.address prior.history prior.cells ws next.cells := by
  obtain ⟨earlier,later,cut⟩ := List.append_of_mem site.member
  obtain ⟨same,_,ws,grew,physical⟩ := trace.after_visit mc fc earlier later site.visit cut
  rw [site.worker] at same grew physical
  exact ⟨earlier,later,ws,cut,same.symm,same ▸ grew,same ▸ physical⟩

theorem CacheSite.scored {p : Pool} {trace : Timeline p} {log : Log mc fc}
    (site : CacheSite trace f depth a z) (full : ExactDepth.Full f.position depth)
    (samples : 6 < site.visit.input.empties.count.toNat →
      TableReads.ProbeReadsFrom (Samples log endian f) site.visit.input.attempts)
    (score : (site.visit.input.run mc fc p.address (trace.readAt mc fc f.worker site.visit.first).history).score = some f.score) :
    ScoreStep (mc:=mc) (fc:=fc) log endian f depth a z := by
  let i := site.visit.input
  let past := (trace.readAt mc fc f.worker site.visit.first).history
  have aligned : i.empties.matches f.position := site.board ▸ i.aligned
  have same : i.depth = i.empties.count := site.atDepth.trans (full.trans aligned.1.symm)
  have valid := (trace.reads mc fc f.worker site.visit.first).1
  by_cases short : i.empties.count.toNat ≤ 6
  · have returns := EndgameSearch.CacheHistory.shallow_score (mc:=mc) (fc:=fc) (addr:=p.address) (past:=past) i same short
    have source := returns.symm.trans score
    change (NonPvMir.sourceRun _ _ _ _ _ _ i.board _ i.depth _ i.alpha _).map _ = _ at source
    rw [site.board,site.atDepth,site.alpha] at source
    exact .leaf i.target i.bmi2 i.checked p.address (i.state past) site.nonPv aligned short (fun _ => valid) source
  · have large : 6 < i.empties.count.toNat := by omega
    have returns := EndgameSearch.CacheHistory.deep_score (mc:=mc) (fc:=fc) (addr:=p.address) (past:=past) i same
      large site.boundary site.exactTag
    have source := returns.symm.trans score
    change DeepLeafMir.sourceRun _ _ _ _ _ _ i.board i.alpha i.beta i.depth _ _ _ _ _ _ = _ at source
    rw [site.board,site.atDepth,site.alpha,site.beta] at source
    exact .deepLeaf i.target i.bmi2 i.paired p.address (i.state past) i.attempts i.parallel i.active i.checked i.aborted
      site.nonPv aligned valid (samples large) source

end NeuralReversi.PvsParallel.FrameReturns

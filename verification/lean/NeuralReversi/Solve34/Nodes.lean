module
public import NeuralReversi.Solve34.SourcePipeline
public import NeuralReversi.Solve2.SourcePipeline

@[expose] public section

namespace NeuralReversi.Solve34

def decodedNodeCost (cost : EmptyBoard n → Score → Nat) (c : Dispatch.Call) : Nat :=
  if h : ValidCall n c then
    cost ⟨c.player,c.opponent,c.remaining,h.1,h.2.1,h.2.2.1,h.2.2.2⟩ c.alpha
  else 0

theorem decodedNodeCost_boardCall (cost : EmptyBoard n → Score → Nat)
    (b : EmptyBoard n) (alpha : Score) :
    decodedNodeCost cost (boardCall b alpha) = cost b alpha := by
  simp only [decodedNodeCost, boardCall_valid, ↓reduceDIte]
  rfl

def twoNodeCount (b : EmptyBoard 2) (alpha : Score) : Nat :=
  (Solve2.sourceFallback b.toTwoEmpty alpha).nodeDelta

def source3Run (b : EmptyBoard 3) (alpha : Score) : Dispatch.Outcome :=
  source3Fallback b (decodedNodeCost twoNodeCount) alpha

def threeNodeCount (b : EmptyBoard 3) (alpha : Score) : Nat :=
  (source3Run b alpha).nodes

def source4Run (b : EmptyBoard 4) (alpha : Score) : Dispatch.Outcome :=
  source4Fallback b (decodedNodeCost threeNodeCount) alpha

theorem source3Run_score (b : EmptyBoard 3) (alpha : Score) :
    (source3Run b alpha).score = solve3Concrete b alpha :=
  source3Fallback_score b _ alpha

theorem source4Run_score (b : EmptyBoard 4) (alpha : Score) :
    (source4Run b alpha).score = solve4Concrete b alpha :=
  source4Fallback_score b _ alpha

theorem source3Run_nws (b : EmptyBoard 3) (alpha : Score) (safe : WindowSafe alpha) :
    Solve2.NWS alpha.toInt (exactThreeValue b) (source3Run b alpha).score.toInt :=
  source3Fallback_nws b _ alpha safe

theorem source4Run_nws (b : EmptyBoard 4) (alpha : Score) (safe : WindowSafe alpha) :
    Solve2.NWS alpha.toInt (exactFourValue b) (source4Run b alpha).score.toInt :=
  source4Fallback_nws b _ alpha safe

theorem source3Run_nodes (b : EmptyBoard 3) (alpha : Score) :
    let out := source3Run b alpha
    out.nodes = 1 + out.events.count .pass +
      (out.calls.map (decodedNodeCost twoNodeCount)).sum :=
  sourceFallback_nodes true b twoSolver _ alpha

theorem source4Run_local_nodes (b : EmptyBoard 4) (alpha : Score) :
    (source4Run b alpha).events.count .root = 0 ∧
      (source4Run b alpha).events.count .pass = 0 := by
  have h := Dispatch.run_local_nodes false
    (verifiedOracle solve3Concrete (decodedNodeCost threeNodeCount)) alpha
    (Dispatch.fallbackCandidates Solve2.Guard.scalarFlip Solve2.Guard.adjacent
      b.player b.opponent (Solve2.childAlpha alpha) (boardSlots b))
    (Dispatch.fallbackCandidates Solve2.Guard.scalarFlip Solve2.Guard.adjacent
      b.opponent b.player alpha (boardSlots b)) (terminalScore b)
  simpa [source4Run, source4Fallback, sourceFallback, Dispatch.fallback] using h

theorem source4Run_nodes (b : EmptyBoard 4) (alpha : Score) :
    (source4Run b alpha).nodes =
      ((source4Run b alpha).calls.map (decodedNodeCost threeNodeCount)).sum := by
  have h := sourceFallback_nodes false b solve3Concrete
    (decodedNodeCost threeNodeCount) alpha
  change (source4Run b alpha).nodes = 0 + (source4Run b alpha).events.count .pass +
    ((source4Run b alpha).calls.map (decodedNodeCost threeNodeCount)).sum at h
  simpa only [(source4Run_local_nodes b alpha).2, Nat.zero_add] using h

def source3EagerRun (b : EmptyBoard 3) (alpha : Score) : Dispatch.Outcome :=
  Dispatch.eager true (verifiedOracle twoSolver (decodedNodeCost twoNodeCount))
    Dispatch.scalarBatch b.player b.opponent alpha (boardSlots b) (terminalScore b)

def source4EagerRun (b : EmptyBoard 4) (alpha : Score) : Dispatch.Outcome :=
  Dispatch.eager false (verifiedOracle solve3Concrete (decodedNodeCost threeNodeCount))
    Dispatch.scalarBatch b.player b.opponent alpha (boardSlots b) (terminalScore b)

theorem source3EagerRun_eq (b : EmptyBoard 3) (alpha : Score) :
    source3EagerRun b alpha = source3Run b alpha :=
  Dispatch.scalar_eager_eq_fallback true _ _ _ _ _ _

theorem source4EagerRun_eq (b : EmptyBoard 4) (alpha : Score) :
    source4EagerRun b alpha = source4Run b alpha :=
  Dispatch.scalar_eager_eq_fallback false _ _ _ _ _ _

theorem source3_batch_eq (batch : Dispatch.BatchFlip)
    (agree : ∀ squares p o, batch squares p o = Dispatch.scalarBatch squares p o)
    (b : EmptyBoard 3) (alpha : Score) :
    Dispatch.eager true (verifiedOracle twoSolver (decodedNodeCost twoNodeCount))
      batch b.player b.opponent alpha (boardSlots b) (terminalScore b) = source3Run b alpha := by
  have eq : batch = Dispatch.scalarBatch := funext fun sq => funext fun p => funext fun o => agree sq p o
  rw [eq]
  exact source3EagerRun_eq b alpha

theorem source4_batch_eq (batch : Dispatch.BatchFlip)
    (agree : ∀ squares p o, batch squares p o = Dispatch.scalarBatch squares p o)
    (b : EmptyBoard 4) (alpha : Score) :
    Dispatch.eager false (verifiedOracle solve3Concrete (decodedNodeCost threeNodeCount))
      batch b.player b.opponent alpha (boardSlots b) (terminalScore b) = source4Run b alpha := by
  have eq : batch = Dispatch.scalarBatch := funext fun sq => funext fun p => funext fun o => agree sq p o
  rw [eq]
  exact source4EagerRun_eq b alpha

end NeuralReversi.Solve34

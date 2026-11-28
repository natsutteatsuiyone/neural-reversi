module
public import NeuralReversi.Flip.Dispatch.Batch
public import NeuralReversi.Solve1.ScalarVerified
public import NeuralReversi.Solve1.NeonVerified
public import NeuralReversi.Solve2.SourcePipeline

@[expose] public section

namespace NeuralReversi.Solve2

open Flip.Dispatch (TargetConfig Backend backendFor pair_eq single_eq)

def solve1On (target : TargetConfig) (bmi2 : Bool) (b : Solve1.Bmi2.Board)
    (alpha : Score) : Score :=
  match target with
  | .x86_64 _ => if bmi2 then Solve1.Bmi2.solve b alpha else Solve1.Scalar.solve b alpha
  | .aarch64 _ _ => Solve1.Neon.solve b alpha
  | .other => Solve1.Scalar.solve b alpha

theorem solve1On_eq (target : TargetConfig) (bmi2 : Bool) (b : Solve1.Bmi2.Board) (alpha : Score) :
    solve1On target bmi2 b alpha = Solve1.Bmi2.solve b alpha := by
  cases target <;> simp [solve1On, Solve1.Scalar.solve_eq_bmi2, Solve1.Neon.solve_eq_bmi2]

def targetOracle (target : TargetConfig) (bmi2 : Bool) (c : Dispatch.Call) : Score :=
  if h : c.player.getLsbD c.square.val = false then
    solve1On target bmi2 ⟨c.player, c.square, h⟩ c.alpha
  else 0

private theorem targetOracle_eq (target : TargetConfig) (bmi2 : Bool) :
    targetOracle target bmi2 = verifiedOracle := by
  funext c
  unfold targetOracle verifiedOracle
  split <;> simp [solve1On_eq]

def targetRun (target : TargetConfig) (bmi2 : Bool) (b : TwoEmpty) (alpha : Score) : Dispatch.Outcome :=
  let backend := backendFor target
  let oracle := targetOracle target bmi2
  if backend = .avx512 then
    Dispatch.eager oracle backend.pair b.player b.opponent alpha b.first b.second (terminalScore b)
  else
    Dispatch.fallback oracle backend.single Guard.adjacent b.player b.opponent
      alpha b.first b.second (terminalScore b)

theorem targetRun_eq (target : TargetConfig) (bmi2 : Bool) (b : TwoEmpty) (alpha : Score) :
    targetRun target bmi2 b alpha = sourceFallback b alpha := by
  have swapped : b.opponent &&& b.player = 0 := by rw [BitVec.and_comm]; exact b.disjoint
  unfold targetRun sourceFallback
  dsimp only
  rw [targetOracle_eq]
  split
  · exact Dispatch.eager_eq_fallback verifiedOracle Guard.scalarFlip (backendFor target).pair Guard.adjacent
      b.player b.opponent alpha b.first b.second (terminalScore b)
      (pair_eq _ _ _ _ _ b.disjoint) (pair_eq _ _ _ _ _ swapped)
      (Guard.scalar_compatible.guard b.first b.player b.opponent)
      (Guard.scalar_compatible.guard b.second b.player b.opponent)
      (Guard.scalar_compatible.guard b.first b.opponent b.player)
      (Guard.scalar_compatible.guard b.second b.opponent b.player)
  · simp only [Dispatch.fallback, single_eq _ _ _ _ b.disjoint, single_eq _ _ _ _ swapped]
    rfl

end NeuralReversi.Solve2

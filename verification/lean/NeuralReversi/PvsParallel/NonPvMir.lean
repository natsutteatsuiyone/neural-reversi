import NeuralReversi.EndgameSearch.Direct.Mir
import NeuralReversi.EndgameSearch.ShallowMir
import NeuralReversi.PvsParallel
import NeuralReversi.PvsParallel.NonPvMir.Program

namespace NeuralReversi.PvsParallel.NonPvMir

open PvsBoardTree (Position build minimax)
open ScoreRefresh (Score toDiscDiff fromDiscDiff)
open EndgameTreeCache (Address Store soundHistory)
open EndgameSearch (EmptyStateMatches)
open Flip.Dispatch (TargetConfig)
open Bitboard.Moves (X86Config)
open EmptyListLinks (Links)

private def withShallow (target : TargetConfig) (bmi2 : Bool) (b : Position) (nodes : Links) (parity : BitVec 8)
    (depth count : BitVec 32) (alpha : Score) (checked : Bool) (score : Unit → Int) : Option Score := do
  if depth ≠ count then none else
    match ← dispatch count.toNat with
    | .fixed arm =>
      let result ← EndgameSearch.Direct.Mir.discResult arm target bmi2 b.val nodes parity checked (toDiscDiff alpha).raw
      (fromDiscDiff checked result).toOption
    | .shallow => (fromDiscDiff checked ⟨BitVec.ofInt 32 (score ())⟩).toOption

def run (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : Flip.Dispatch.X86Config)
    (address : Address) (history : List Store) (b : Position) (nodes : Links) (parity : BitVec 8)
    (depth count : BitVec 32) (alpha : Score) (checked : Bool) : Option Score :=
  withShallow target bmi2 b nodes parity depth count alpha checked fun _ =>
    (EndgameSearch.run mc fc (EndgameSearch.orderedOptions target bmi2 mc fc address)
      .shallow b (toDiscDiff alpha).raw.toInt history).score

private theorem fixed_dispatch (n : Nat) (size : n ≤ 4) :
    dispatch n = (EndgameSearch.Direct.Mir.dispatch n).map Arm.fixed := by
  have split : n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by omega
  rcases split with h|h|h|h|h <;> subst n <;> rfl

theorem run_returns (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : Flip.Dispatch.X86Config)
    (address : Address) (history : List Store) (b : Position) (nodes : Links) (parity : BitVec 8)
    (depth count : BitVec 32) (alpha beta : Score) (checked : Bool)
    (full : ExactDepth.Full b depth) (aligned : EmptyStateMatches b nodes parity count)
    (size : count.toNat ≤ 6) (cache : 5 ≤ count.toNat → soundHistory mc fc history)
    (null : beta.raw.toInt = alpha.raw.toInt + 1) :
    ∃ score, run target bmi2 mc fc address history b nodes parity depth count alpha checked = some score ∧
      EndgameSearch.Direct.Returned target bmi2 mc fc b alpha checked score := by
  have same : depth = count := full.trans aligned.1.symm
  by_cases fixed : count.toNat ≤ 4
  · obtain ⟨score, value, returned⟩ := EndgameSearch.Direct.Mir.run_returns
      target bmi2 mc fc b nodes parity count alpha checked aligned fixed
    refine ⟨score, ?_, returned⟩
    unfold EndgameSearch.Direct.Mir.run at value
    unfold run withShallow
    rw [same, fixed_dispatch _ fixed]
    cases arm : EndgameSearch.Direct.Mir.dispatch count.toNat <;> simp [arm] at value ⊢
    exact value
  · have lower : 5 ≤ count.toNat := by omega
    have arm : dispatch count.toNat = some .shallow := by
      have split : count.toNat = 5 ∨ count.toNat = 6 := by omega
      rcases split with h|h <;> rw [h] <;> rfl
    have hist := cache lower
    obtain ⟨score, cast, _, _, _⟩ := EndgameSearch.orderedContract_proven target bmi2 mc fc address
      .shallow b alpha beta history checked hist null
    refine ⟨score, ?_, .shallow address history (aligned.1 ▸ lower) (aligned.1 ▸ size) hist cast⟩
    simp [run, withShallow, same, arm, cast, Except.toOption]

theorem scoreContract_proven (exit : Exit) (children : ChildResults)
    (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : Flip.Dispatch.X86Config)
    (address : Address) (history : List Store) (b : Position) (nodes : Links) (parity : BitVec 8)
    (depth count : BitVec 32) (alpha beta : Score) (checked : Bool)
    (full : ExactDepth.Full b depth) (aligned : EmptyStateMatches b nodes parity count)
    (size : count.toNat ≤ 6) (cache : 5 ≤ count.toNat → soundHistory mc fc history)
    (null : beta.raw.toInt = alpha.raw.toInt + 1) :
    ∃ score, run target bmi2 mc fc address history b nodes parity depth count alpha checked = some score ∧
      ExecutionWithUses exit children (build mc fc b) false alpha.raw.toInt beta.raw.toInt score.raw.toInt ∧
      PvsMoveLoop.Bound (minimax mc fc b) alpha.raw.toInt beta.raw.toInt score.raw.toInt := by
  obtain ⟨s, value, returned⟩ := run_returns target bmi2 mc fc address history b nodes parity
    depth count alpha beta checked full aligned size cache null
  exact ⟨s, value, .direct target bmi2 mc fc b alpha beta s checked returned, returned.bound beta null⟩

def sourceRun (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : Flip.Dispatch.X86Config)
    (address : Address) (history : List Store) (b : Position) (nodes : Links) (parity : BitVec 8)
    (depth count : BitVec 32) (alpha : Score) (checked : Bool) : Option Score :=
  withShallow target bmi2 b nodes parity depth count alpha checked fun _ =>
    (EndgameSearch.ShallowMir.run target bmi2 mc fc address b ⟨nodes, parity, count⟩ (toDiscDiff alpha).raw.toInt history).score

private theorem fixed_bind (arm : Option EndgameSearch.Direct.Mir.Arm)
    (fixed : EndgameSearch.Direct.Mir.Arm → Option Score) (left right : Unit → Option Score) :
    ((arm.map Arm.fixed).bind fun | .fixed a => fixed a | .shallow => left ()) =
      ((arm.map Arm.fixed).bind fun | .fixed a => fixed a | .shallow => right ()) := by
  cases arm <;> rfl

private theorem withShallow_fixed (target : TargetConfig) (bmi2 : Bool) (b : Position) (nodes : Links) (parity : BitVec 8)
    (depth count : BitVec 32) (alpha : Score) (checked : Bool) (left right : Unit → Int) (fixed : count.toNat ≤ 4) :
    withShallow target bmi2 b nodes parity depth count alpha checked left =
      withShallow target bmi2 b nodes parity depth count alpha checked right := by
  unfold withShallow
  split
  · rfl
  · rw [fixed_dispatch _ fixed]
    exact fixed_bind (EndgameSearch.Direct.Mir.dispatch count.toNat) (fun arm => do
      let result ← EndgameSearch.Direct.Mir.discResult arm target bmi2 b.val nodes parity checked (toDiscDiff alpha).raw
      (fromDiscDiff checked result).toOption)
      (fun _ => (fromDiscDiff checked ⟨BitVec.ofInt 32 (left ())⟩).toOption)
      (fun _ => (fromDiscDiff checked ⟨BitVec.ofInt 32 (right ())⟩).toOption)

theorem sourceRun_eq (target : TargetConfig) (bmi2 : Bool) (mc : X86Config) (fc : Flip.Dispatch.X86Config)
    (address : Address) (history : List Store) (b : Position) (nodes : Links) (parity : BitVec 8)
    (depth count : BitVec 32) (alpha : Score) (checked : Bool)
    (aligned : EmptyStateMatches b nodes parity count) (cache : 5 ≤ count.toNat → soundHistory mc fc history) :
    sourceRun target bmi2 mc fc address history b nodes parity depth count alpha checked =
      run target bmi2 mc fc address history b nodes parity depth count alpha checked := by
  unfold sourceRun run
  by_cases fixed : count.toNat ≤ 4
  · exact withShallow_fixed _ _ _ _ _ _ _ _ _ _ _ fixed
  · congr 1
    funext _
    exact congrArg EndgameSearch.Result.score (EndgameSearch.ShallowMir.agrees target bmi2 mc fc address b ⟨nodes, parity, count⟩ _
      history aligned (EndgameSearch.ShallowMir.scoreThreshold alpha) (cache (by omega)))

end NeuralReversi.PvsParallel.NonPvMir

import NeuralReversi.PvsParallel.NonPvMir
import NeuralReversi.PvsParallel.DeepLeafMir
import NeuralReversi.PvsParallel.Replay

namespace NeuralReversi.PvsParallel.SerialScout

open PvsBoardTree (Position build)
open EndgameTreeCache (Address Store soundHistory)
open EndgameSearch.Cancellation (State EmptyState)
open ScoreRefresh (Score toDiscDiff)

structure Config where
  target : Flip.Dispatch.TargetConfig
  bmi2 : Bool
  paired : Bool
  checked : Bool
  address : Address

/-- Both projections use the same call arguments and incoming worker cache. -/
def run (c : Config) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (b : Position) (s : EmptyState) (a z : Score) (past : List Store) : Option (Int × List Store) := do
  let disc := (toDiscDiff a).raw.toInt
  if s.count.toNat ≤ 6 then
    let raw ← NonPvMir.sourceRun c.target c.bmi2 mc fc c.address past b s.mask s.count s.count a c.checked
    let after := if s.count.toNat ≤ 4 then past else
      (EndgameSearch.ShallowMir.run c.target c.bmi2 mc fc c.address b s disc past).history
    return (raw.raw.toInt, after)
  else
    let state : State := ⟨s, past, ⟨⟨0, false⟩, fun _ => ⟨0, []⟩, 0⟩⟩
    let raw ← DeepLeafMir.sourceRun c.target c.bmi2 c.paired mc fc c.address b a z s.count state
      [] false false c.checked false
    let out := EndgameSearch.DeepRecursionMir.run c.target c.bmi2 c.paired mc fc c.address b disc
      (DeepLeafMir.pollInput state false)
    return (raw, out.state.history)

theorem returns (c : Config) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (b : Position) (s : EmptyState) (a z : Score) (past : List Store)
    (aligned : s.matches b) (size : s.count.toNat ≤ 12) (cache : soundHistory mc fc past)
    (null : z.raw.toInt = a.raw.toInt + 1) :
    ∃ raw after, run c mc fc b s a z past = some (raw, after) ∧
      boundedReplay 1 (build mc fc b) false a.raw.toInt z.raw.toInt raw ∧
      soundHistory mc fc after ∧ past <+: after := by
  let disc := (toDiscDiff a).raw.toInt
  have safe := EndgameSearch.ShallowMir.scoreThreshold a
  by_cases shallow : s.count.toNat ≤ 6
  · obtain ⟨raw, returned, replay, _⟩ := NonPvMir.scoreContract_proven Cutoffs.Execution
      (fun _ => boundedReplay 0) c.target c.bmi2 mc fc c.address past b s.mask s.count s.count
      a z c.checked aligned.1 aligned shallow (fun _ => cache) null
    rw [← NonPvMir.sourceRun_eq _ _ _ _ _ _ _ _ _ _ _ _ aligned (fun _ => cache)] at returned
    by_cases fixed : s.count.toNat ≤ 4
    · exact ⟨_, _, by simp [run, shallow, fixed, returned], replay, cache, List.prefix_refl _⟩
    · let out := EndgameSearch.ShallowMir.run c.target c.bmi2 mc fc c.address b s disc past
      have same := EndgameSearch.ShallowMir.agrees c.target c.bmi2 mc fc c.address b s disc past aligned safe cache
      have valid := EndgameSearch.contract_proven mc fc _ .shallow b disc past
        (EndgameSearch.orderedOptions_correct c.target c.bmi2 mc fc c.address) cache (fun _ _ => safe)
      rw [← same] at valid
      exact ⟨_, out.history, by simp [run, shallow, fixed, returned, out, disc], replay, valid.2.2⟩
  · let state : State := ⟨s, past, ⟨⟨0, false⟩, fun _ => ⟨0, []⟩, 0⟩⟩
    let input := DeepLeafMir.pollInput state false
    have quiet : EndgameSearch.Cancellation.NoSplit input.polls := ⟨rfl, fun _ => rfl⟩
    have live : DeepLeafMir.uncancelled c.target c.bmi2 mc fc c.address b a state false :=
      (EndgameSearch.Cancellation.run_noSplit mc fc _ b disc input quiet).1
    obtain ⟨raw, returned, replay, _⟩ := DeepLeafMir.normalReturn (children := fun _ => boundedReplay 0)
      c.target c.bmi2 mc fc c.address b a z s.count state [] false false c.checked
      .initial (fun _ h => by simp at h) aligned cache aligned.1 ⟨by omega, size⟩ null live
    rw [← DeepLeafMir.sourceRun_eq c.target c.bmi2 c.paired mc fc c.address b a z s.count state
      [] false false c.checked false aligned cache] at returned
    have valid := EndgameSearch.DeepRecursionMir.returned_score c.target c.bmi2 c.paired mc fc c.address
      b disc input aligned safe cache
    exact ⟨_, _, by simp [run, shallow, state, returned, disc, input] at *, replay, valid.2.1, valid.2.2.1⟩

/-- The caller only consumes the returned word. Up to four empties the solver
    leaves the supplied cache untouched and its contents cannot affect the bound. -/
theorem score_returns (c : Config) (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig)
    (b : Position) (s : EmptyState) (a z : Score) (past : List Store)
    (aligned : s.matches b) (size : s.count.toNat ≤ 12) (cache : 5 ≤ s.count.toNat → soundHistory mc fc past)
    (null : z.raw.toInt = a.raw.toInt + 1) :
    ∃ raw after, run c mc fc b s a z past = some (raw, after) ∧
      boundedReplay 1 (build mc fc b) false a.raw.toInt z.raw.toInt raw := by
  by_cases fixed : s.count.toNat ≤ 4
  · have shallow : s.count.toNat ≤ 6 := by omega
    obtain ⟨raw, answer, replay, _⟩ := NonPvMir.scoreContract_proven Cutoffs.Execution
      (fun _ => boundedReplay 0) c.target c.bmi2 mc fc c.address past b s.mask s.count s.count
      a z c.checked aligned.1 aligned shallow cache null
    rw [← NonPvMir.sourceRun_eq _ _ _ _ _ _ _ _ _ _ _ _ aligned cache] at answer
    exact ⟨_,past,by simp [run, shallow, fixed, answer],replay⟩
  · obtain ⟨raw,after,computed,sound,_⟩ := returns c mc fc b s a z past aligned size (cache (by omega)) null
    exact ⟨raw,after,computed,sound⟩

end NeuralReversi.PvsParallel.SerialScout

module
public import NeuralReversi.PvsParallel

@[expose] public section

namespace NeuralReversi.PvsParallel

open Aspiration (SoundResult Window)
open PvsMoveLoop (Tree minimax scoreInf)

theorem ExecutionWithUses.weaken {observed accepted : ChildResults} {seen verified : Exit}
    (liftChild : ∀ stage tree pv alpha beta score,
      observed stage tree pv alpha beta score → accepted stage tree pv alpha beta score)
    (liftExit : ∀ mc fc board pv alpha beta score,
      seen mc fc board pv alpha beta score → verified mc fc board pv alpha beta score)
    (run : ExecutionWithUses seen observed tree pv alpha beta score) :
    ExecutionWithUses verified accepted tree pv alpha beta score := by
  cases run with
  | leaf => exact .leaf
  | pass child => exact .pass (liftChild _ _ _ _ _ _ child)
  | cutoff mc fc b exit => exact .cutoff mc fc b (liftExit _ _ _ _ _ _ _ exit)
  | direct target bmi2 mc fc board a b raw checked call =>
    exact .direct target bmi2 mc fc board a b raw checked call
  | shallow target bmi2 mc fc address board a b raw state checked aborted aligned cache result =>
    exact .shallow target bmi2 mc fc address board a b raw state checked aborted aligned cache result
  | sequential moves child order observe calls =>
    exact .sequential moves child order observe (calls.weaken (fun _ => liftChild .serial _))
  | split moves order observe split continued earlierCalls reports =>
    exact .split moves order observe split continued
      (earlierCalls.weaken fun _ => liftChild .serial _) (reports.weaken liftChild)

theorem ExecutionWith.weaken {observed accepted : Search} {seen verified : Exit}
    (liftChild : ∀ tree pv alpha beta score,
      observed tree pv alpha beta score → accepted tree pv alpha beta score)
    (liftExit : ∀ mc fc board pv alpha beta score,
      seen mc fc board pv alpha beta score → verified mc fc board pv alpha beta score)
    (run : ExecutionWith seen observed tree pv alpha beta score) :
    ExecutionWith verified accepted tree pv alpha beta score :=
  ExecutionWithUses.weaken (fun _ => liftChild) liftExit run

def boundedReplay : Nat → Search
  | 0 => fun _ _ _ _ _ => False
  | n + 1 => Execution (boundedReplay n)

theorem boundedReplay_mono {old new : Nat} (ordered : old ≤ new) :
    ∀ tree pv alpha beta score, boundedReplay old tree pv alpha beta score →
      boundedReplay new tree pv alpha beta score := by
  induction old generalizing new with
  | zero => intro _ _ _ _ _ impossible; exact impossible.elim
  | succ old ih =>
    cases new with
    | zero => omega
    | succ new =>
      intro tree pv alpha beta score replayed
      exact (show Execution (boundedReplay old) tree pv alpha beta score from replayed).weaken
        (ih (Nat.le_of_succ_le_succ ordered)) (fun _ _ _ _ _ _ _ found => found)

def Replayed : Search := fun tree pv alpha beta score =>
  ∃ frames, boundedReplay frames tree pv alpha beta score

theorem replay_follows : Follows Replayed := by
  intro tree pv alpha beta score replayed
  obtain ⟨frames, run⟩ := replayed
  cases frames with
  | zero => exact run.elim
  | succ frames =>
    exact (show Execution (boundedReplay frames) tree pv alpha beta score from run).weaken
      (fun _ _ _ _ _ child => ⟨frames, child⟩) (fun _ _ _ _ _ _ _ found => found)

def ReplayContract : Prop :=
  ∀ tree pv alpha beta score frames,
    Window alpha beta → (pv = false → beta = alpha + 1) →
    boundedReplay frames tree pv alpha beta score → SoundResult (minimax tree) alpha beta score

theorem replayContract_proven : ReplayContract := by
  intro tree pv alpha beta score frames window null trace
  exact sound Replayed replay_follows tree pv alpha beta score window null ⟨frames, trace⟩

def ReplayedBoardContract : Prop :=
  ∀ mc fc board score frames,
    boundedReplay frames (PvsBoardTree.build mc fc board) true (-scoreInf) scoreInf score →
    score = PvsBoardTree.minimax mc fc board

theorem replayedBoardContract_proven : ReplayedBoardContract := by
  intro mc fc board score frames trace
  exact fullWindow_exact Replayed replay_follows mc fc board score ⟨frames, trace⟩

end NeuralReversi.PvsParallel

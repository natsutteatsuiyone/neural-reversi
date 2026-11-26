import NeuralReversi.PvsParallel.SerialRecursion
import NeuralReversi.PublicationMir.ExactScore

namespace NeuralReversi.AspirationMir.SerialExact

open PvsParallel
open PvsBoardTree (Position)
open RootMoveStorage (Row)

/-- At entry to the exact level the root vector covers the legal moves. A retry
    may observe different published TT values and different ordering keys.
    All recursive child scores are computed inside each trial. -/
def PublicScoreContract : Prop :=
  ∀ mc fc (c : SerialScout.Config) (b : Position) (rows : List Row) (m : Machine)
    (inputs : Nat → SerialRecursion.Inputs mc fc) (root : Nat → Word → Word → SerialRecursion.RootInput)
    (sort : PublicationMir.SliceSort) (nodes : Nat) (fallback : Unit → PublicationMir.SearchResult),
    (Board.emptyCount b.val).toNat ≤ 60 → Root.Covers mc b (rows.map Row.square) → 0 < rows.length →
    (∀ i, SerialRecursion.Admissible (inputs i)) →
    m.pc = 0 → Aspiration.Window m.regs.alpha.toInt m.regs.beta.toInt →
    let trial := fun i a z => SerialRecursion.rootTrial c (inputs i) [] b (root i a z) rows 0 a z
    ∃ raw last answer result, Returns (SmallRoot.queryFor trial) (fun _ => false) m raw last ∧
      trial last.index last.alpha last.beta = some answer ∧ answer.value = raw.toInt ∧
      PublicationMir.completed sort (RootMoveStorage.sortRows answer.rows 0)
        (Board.emptyCount b.val).toNat 5 true nodes fallback = some result ∧
      PublicationMir.ExactScore.scoreIfExact result = some (256 * EndgameTreeCache.discValue mc fc b.val)

theorem publicScore : PublicScoreContract := by
  intro mc fc c b rows m inputs root sort nodes fallback fits covers within valid entry window trial
  obtain ⟨raw, last, answer, snap, returned, ran, score, _, written, inside, sound⟩ :=
    SmallRoot.completeTrials mc fc b rows 0 m trial (fun i a z w => by
      obtain ⟨answer, snap, ran, best, written, _, sound⟩ := SerialRecursion.rootTrialContract_proven
        mc fc c (inputs i) [] b (root i a z) rows 0 a z (valid i) fits covers within w
      exact ⟨answer, snap, ran, best, written, sound⟩) entry window
  obtain ⟨result, published, exactValue⟩ := publishSingle covers within written inside sound
    (PublicationMir.ExactScore.afterRootSort sort) (Board.emptyCount b.val).toNat nodes fallback
  refine ⟨raw, last, answer, result, returned, ran, score, published, ?_⟩
  rw [PublicationMir.ExactScore.completed_tagged published, EndgameTreeCache.scaled_discValue]
  exact exactValue

end NeuralReversi.AspirationMir.SerialExact

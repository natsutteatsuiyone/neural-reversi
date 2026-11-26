import NeuralReversi.PvsParallel.Interleaving.AlphaReplay

namespace NeuralReversi.PvsParallel.Interleaving.OwnerTape

open Threading
open AtomicWord (Event Read)
open SplitOwnerReadback (Load bestWords moveWords bestSite moveSite)
open SlotFrameMir.RawProjection (OwnerSetup PhysicalOwner)
open AlphaOrigin (SetupOperands OwnerOperands)

def loads (es : Joint.Tape Op) : List Load := es.filterMap fun row => match row with
  | .ownerLoad w t p bank source word =>
    if fits : p < 64 then
      let what := match word with | .best v => .score v | .picked v => .move v
      some ⟨(⟨p,fits⟩,bank),⟨w,t⟩,what,false,source⟩
    else none
  | _ => none

private theorem order (c : MaskClock.Capture h) (r : Read ws)
    (ok : AlphaReplay.inOrder c ws r.event r.source.val) : SplitRebookingMir.Coherence h ws r := by
  simp only [AlphaReplay.inOrder,decide_eq_true_eq,← MaskClock.published h c] at ok
  exact ⟨fun i => (ok i).1,fun i => (ok i).2⟩

structure Sources {t : NativeSplit.Tickets rc pv beta} {g : SlotFrameMir.RawMask mask}
    (b : OwnerSetup g h time slot t) where
  score : Fin (bestWords b).length
  move : Fin (moveWords b).length
  scoreOrder : SplitRebookingMir.Coherence h (bestWords b) ⟨bestSite b,score,false⟩
  moveOrder : SplitRebookingMir.Coherence h (moveWords b) ⟨moveSite b,move,false⟩

def sources {t : NativeSplit.Tickets rc pv beta} {g : SlotFrameMir.RawMask mask}
    (b : OwnerSetup g h time slot t) (c : MaskClock.Capture h) (es : Joint.Tape Op) : Option (Sources b) := do
  let (score,move) ← SplitOwnerReadback.readback b (loads es)
  if sc : AlphaReplay.inOrder c (bestWords b) (bestSite b) score.rank.val then
    if mv : AlphaReplay.inOrder c (moveWords b) (moveSite b) move.rank.val then
      return ⟨score.rank,move.rank,order c _ sc,order c _ mv⟩
  none

def physical (b : SetupOperands t) (src : Sources b.owner) : OwnerOperands t :=
  ⟨b.toSplitOperands,{ toOwnerSetup := b.owner, source := src.score, moveSource := src.move,
                       coherence := src.scoreOrder, moveCoherence := src.moveOrder }⟩

end NeuralReversi.PvsParallel.Interleaving.OwnerTape

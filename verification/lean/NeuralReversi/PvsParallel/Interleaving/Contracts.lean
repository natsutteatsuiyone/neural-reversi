import NeuralReversi.PvsParallel.Interleaving.StopSchedule
import NeuralReversi.PvsParallel.Interleaving.NativeNodes
import NeuralReversi.PvsParallel.Interleaving.History

namespace NeuralReversi.PvsParallel.Interleaving

open Threading
open Aspiration (Window SoundResult)
open PvsMoveLoop (Bound ChildRelation scoreMax)
open TranspositionTable (SeqLock.Payload)
open NativeTableWrites.Journal (Calls Record Even)
open PvsBoardTree (Position minimax)

/-- A normal mask return accounts for every pending ticket or a genuine local cutoff.
    The two physical relaxed alpha samples of each accepted report may be stale. -/
def SplitContract : Prop :=
  ∀ (memory : NativeStopReturns.Memory) (slot : Nat) (words prior : List AtomicWord.Write) (owner : SplitCompletion.Worker)
    (g : Mask.Generation) (following : List Mask.Generation) (o : Mask.Observed memory.history prior owner g following)
    (stop : SplitReturn.Observation.OwnerCheck memory.history) (t : Schedule.Task) (children : ChildRelation Nat)
    (w len : Nat) (start : BitVec w) (s : Tickets.State w) (orig prefixBest : Int) (value : Nat → Int),
    StopSchedule.Scored memory stop o.waitAt slot words t children len start owner g.actions s →
    len < 2^w → stop.stopped = false → Window t.entry.alpha t.beta → t.entry.alpha ≤ max orig t.entry.best →
    (t.pv = false → t.beta = t.entry.alpha+1) →
    Bound prefixBest orig t.beta t.entry.best → prefixBest ≤ scoreMax →
    (∀ i ∈ SplitResult.slots start.toNat len, value i ≤ scoreMax) →
    (∀ i, start.toNat ≤ i → i < len → ∀ p a z v, Window a z → (p = false → z = a+1) →
      children i p a z v → SoundResult (-(value i)) a z v) →
    s.retired ∧ s.outside = false ∧ Schedule.Result t orig value start.toNat len prefixBest s

theorem split : SplitContract := by
  intro memory slot words prior owner g following o stop t children w len start s orig prefixBest value
    run fit active window cap null seed upper limits child
  obtain ⟨j,state,_,_,_,answer⟩ := StopSchedule.join o stop run window cap
  have done_ : j.answer = some (t.locked s.reports).best := by simpa [SplitReturn.checked,active] using answer
  obtain ⟨retired,normal,_⟩ := j.finished done_
  exact state ▸ ⟨retired,normal,(j.accepted fit done_ window cap null seed upper limits child).1⟩

/-- Finite prefixes of the physical TT and child-return partial order may include
    unrelated concurrent writers and chosen frames that have yet to return. -/
def RecursiveContract : Prop :=
  ∀ (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (c : Calls) (endian : Bool) (rs : List Record)
    (x : NativeTableWrites.Journal.Execution c endian rs) (init : ∀ a, Even (x.memory a))
    (frames : List CausalTable.Frame) (before : Nat → Nat → Prop),
    CausalTable.Prefix x init frames before (NativeNodes.Body mc fc) →
    (∀ a, Table.Safe mc fc endian ((x.slot a (init a)).payloadAt 0)) →
    (∀ f ∈ frames, Window f.alpha f.beta → (f.pv = false → f.beta = f.alpha+1) →
      SoundResult (minimax mc fc f.board) f.alpha f.beta f.score) ∧
    ∀ t b e, CausalTable.lookup x init t b e → Table.EntrySafe mc fc b e

theorem recursive : RecursiveContract := fun _ _ _ _ _ x init _ _ prefixRun seed =>
  CausalTable.certified x init prefixRun NativeNodes.preserves seed

end NeuralReversi.PvsParallel.Interleaving

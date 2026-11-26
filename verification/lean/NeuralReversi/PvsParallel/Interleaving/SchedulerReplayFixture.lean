import NeuralReversi.PvsParallel.Interleaving.SplitOwnerReadbackFixture
import NeuralReversi.PvsParallel.Interleaving.SchedulerTrace
import NeuralReversi.PvsParallel.Interleaving.BestPublisher

namespace NeuralReversi.PvsParallel.Interleaving.SchedulerReplayFixture

open Threading.AtomicWord (Event)
open SplitOwnerReadback.Regression (tickets gen hist full)
open SchedulerReplay (Instruction)

private def operands : AlphaOrigin.Operands tickets where
  memory := ⟨hist,[],[],fun _ => [],fun _ => []⟩
  mask := _
  generation := gen
  clock := Event.step
  bankIndex := 0
  flagAddress := fun slot => slot.1.val * 64 + slot.2
  owner := full
  seed := ⟨rfl,rfl,rfl⟩
  registers := {
    pre := []
    mutex := ⟨⟨0,0⟩,[],by simp⟩
    checked := rfl
    stateLocks := []
    started := rfl
    rows := []
    coherent := by simp
    operands := rfl
  }

/-- Rust calls the exhausted iterator, then clears the participation bit. The
    captured RF input supplies the initial iterator word at modification index zero. -/
private def empty : Instruction tickets.width := .exhausted 0 0 []
private def reset : Instruction tickets.width := .mask ⟨0,85⟩
private def run :=
  SchedulerReplay.capture operands false false 1 (fun _ _ _ => none)
private def tape := [empty,reset]

private theorem chronology :
    IteratorFields.chronological tickets.state.writes tickets.journal tickets.stream tickets.workers = [] := by
  change ([] : List SplitWorkerMir.Iterator.LockedCall).mergeSort _ = []
  simp

private theorem noCalls : tickets.calls = [] := by
  unfold NativeSplit.Tickets.calls
  rw [chronology]
  rfl

private theorem mapSome (x : Option α) (f : α → β) :
    (x.bind fun a => some (f a)).isSome = x.isSome := by cases x <;> rfl

private theorem accepted : (run tape).isSome := by
  unfold run tape SchedulerReplay.capture
  erw [mapSome, noCalls]
  decide

example : (run [empty]).isNone ∧ (run [reset,empty]).isNone ∧
    (run [.exhausted 0 1 [],reset]).isNone ∧ (run [empty,.mask ⟨0,86⟩]).isNone := by decide

private theorem winner : BestPublisher.publisher tickets = some (⟨⟨0,30⟩,(-1536 : BitVec 32)⟩,none) := by
  unfold BestPublisher.publisher BestPublisher.rows
  rw [chronology]
  rfl

private def words (operations : List (Nat × Joint.SplitWord)) : Joint.Tape Unit :=
  operations.map fun (at_,op) => .splitWord 0 at_ op

private def marked := (20, Joint.SplitWord.mark 0)
private def declined := (81, Joint.SplitWord.exhausted 0 0 0 [])
private def retired := (85, Joint.SplitWord.maskRmw 0 0 0 false 1 0)
private def result := (120, Joint.SplitWord.result 0 (-1536))
private def native := words [marked,declined,retired,result]

private abbrev decode := SchedulerTrace.instructions operands (Op := Unit)
private def replay := SchedulerTrace.fromTape operands false false 1 (fun _ _ _ => none) (Op := Unit)

private theorem projected : decode native = some tape := by
  unfold decode SchedulerTrace.instructions SchedulerTrace.interval
  rw [winner]
  rfl

private theorem fromRaw : (replay native).isSome := by
  unfold replay SchedulerTrace.fromTape
  rw [show SchedulerTrace.instructions operands native = some tape from projected]
  exact accepted

private def captured := (replay native).get fromRaw

example : (decode (words [marked,declined,result])).isNone ∧
    (decode (words [marked,declined,(85,.maskRmw 0 0 0 false 7 0),result])).isNone ∧
    (decode (words [marked,declined,(85,.maskRmw 0 1 0 false 1 0),result])).isNone ∧
    (decode (words [(20,.mark 1),declined,retired,result])).isNone ∧
    (decode (words [marked,declined,retired,(120,.result 0 1536)])).isNone := by
  simp only [decode,SchedulerTrace.instructions,SchedulerTrace.interval,winner]
  decide

example : (replay (words [marked,retired,declined,result])).isNone ∧
    (replay (words [marked,(81,.exhausted 0 1 0 []),retired,result])).isNone ∧
    (replay (words [marked,declined,retired,(120,.result 0 1536)])).isNone := by
  simp only [replay,SchedulerTrace.fromTape,SchedulerTrace.instructions,SchedulerTrace.interval,winner]
  decide

/-- The verified scheduler result feeds the publisher theorem as a Capture.
    The result field is read at exactly the physical source in its word history. -/
example : (BestPublisher.publisher tickets).map (fun r => r.1.word.toInt) =
    some (Threading.AtomicWord.signed32At captured.score.frame.physical.scores
      captured.score.frame.physical.load.source) :=
  BestPublisher.returnWord captured.score

example : Threading.AtomicWord.signed32At captured.score.frame.physical.scores
    captured.score.frame.physical.load.source = -1536 := by
  have reads := BestPublisher.returnWord captured.score
  rw [winner] at reads
  exact (Option.some.inj reads).symm

namespace DispatchOrder

private def a : SchedulerTrace.Row := ⟨⟨0,13⟩,.claim 19 0 0 0 1⟩
private def b : SchedulerTrace.Row := ⟨⟨1,10⟩,.claim 66 0 0 1 2⟩
private def lock : SchedulerTrace.Row := ⟨⟨1,11⟩,.lock 0 0⟩
private def done_ : SchedulerTrace.Row := ⟨⟨2,1⟩,.exhausted 0 0 2 []⟩
private def ops : List (MoveList.ConcurrentTrace.Write 32) := [⟨0,19,0,1⟩,⟨1,66,1,2⟩]
private def hb (u v : Event) := u.worker == v.worker && u.step < v.step
private def sites (rs : List SchedulerTrace.Row) := rs.map (·.site)
private def ordered := (SchedulerTrace.order ops [done_,b,lock,a] hb).map sites

-- The shared tape lists the exhausted observation first. The split projection
-- orders both CASs before it and preserves worker 1's claim-before-lock edge.
#guard (ordered.map fun es => es.filter (· != lock.site)) == some [a.site,b.site,done_.site]
#guard (ordered.map fun es => es.filter (·.worker == 1)) == some [b.site,lock.site]
#guard (SchedulerTrace.order ops [done_,b,lock] hb).isNone
#guard (SchedulerTrace.order ops [done_,b,a,lock,a] hb).isNone
#guard (SchedulerTrace.order ops [done_,b,lock,a] (fun _ _ => true)).isNone

end DispatchOrder

end NeuralReversi.PvsParallel.Interleaving.SchedulerReplayFixture

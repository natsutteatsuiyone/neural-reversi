import NeuralReversi.PvsParallel.NativeSplit
import NeuralReversi.PvsParallel.SplitRebookingMir.Ordering

namespace NeuralReversi.PvsParallel.NativeSplit.AlphaLoads

open Threading
open AtomicWord (Event Write Read signed32At)
open CutoffVisibility (History)
open SplitWorkerMir (Input Word retry Safe)
open IteratorFrames (Observation Posted)

structure Memory (history : History) (words : List Write) where
  reads : List (Read words)
  coherent : ∀ r ∈ reads, SplitRebookingMir.Coherence history words r

/-- The physical load has one source in this alpha word's modification order. An
    equal value from another load at a different instruction cannot identify it. -/
def operand (reads : List (Read words)) (e : Event) (v : Word) : Bool :=
  match reads.filter (fun r => r.event == e) with
  | [r] => !r.acquire && signed32At words r.source == v.toInt
  | _ => false

def Memory.load (m : Memory hist words) (hit : operand m.reads event word)
    (published : SplitPublication.PublishedThrough hist seed event) : IteratorReports.Load hist words seed event word :=
  match site : m.reads.filter (fun r => r.event == event) with
  | [r] => by
    have member : r ∈ m.reads.filter (fun r => r.event == event) := by simp [site]
    obtain ⟨issued,actor⟩ := List.mem_filter.mp member
    have observed : !r.acquire && signed32At words r.source == word.toInt := by simpa [operand,site] using hit
    have same := beq_iff_eq.mp actor
    have coherent := m.coherent r issued
    exact ⟨r.source,(beq_iff_eq.mp (Bool.and_eq_true_iff.mp observed).2).symm,published,
      same ▸ coherent.writeRead,same ▸ coherent.readWrite⟩
  | [] | _ :: _ :: _ => False.elim (by simp [operand,site] at hit)

theorem Memory.load_source (m : Memory hist words) (hit : operand m.reads event word)
    (published : SplitPublication.PublishedThrough hist seed event)
    (site : m.reads.filter (fun r => r.event == event) = [r]) :
    (m.load hit published).source = r.source := by
  unfold Memory.load
  split <;> simp_all

private def safeWords (i : Input) : Bool :=
  decide (-2147483648 < i.alpha.toInt ∧ i.alpha.toInt < 2147483647 ∧ -2147483648 < i.scout.toInt ∧
    (retry i = true → -2147483648 < i.beta.toInt ∧ -2147483648 < i.fresh.toInt ∧ -2147483648 < i.wide.toInt))

private theorem safe (good : safeWords i) : Safe i := by
  obtain ⟨lo,hi,scout,wide⟩ := of_decide_eq_true good
  exact ⟨lo,hi,scout,fun h => (wide h).1,fun h => (wide h).2.1,fun h => (wide h).2.2⟩

def operands (reads : List (Read words)) (obs : Observation) (i : Input) : Bool :=
  safeWords i && operand reads obs.scoutLoad i.alpha && (!retry i || operand reads obs.wideLoad i.fresh)

def ticket (reads : List (Read words)) (journal : List Observation) (stream : List NativeFrames.Record)
    (op : MoveList.ConcurrentTrace.Write w) : Bool := (do
  let c ← SplitWorkerMir.Iterator.acquired (IteratorFrames.source journal stream) op
  let obs ← journal[c.order]?
  return c.ticket.locked.stopped || operands reads obs c.ticket.input).getD false

theorem checked {s : MoveList.ConcurrentTrace.State w} (x : Posted rc s journal stream earlier r seq)
    (hit : ticket reads journal stream x.op) :
    Safe x.input ∧ operand reads x.lock.scoutLoad x.input.alpha = true ∧
      (retry x.input = true → operand reads x.lock.wideLoad x.input.fresh = true) := by
  have read : operands reads x.lock x.input := by
    simpa [ticket,x.acquired,x.atLock,x.running] using hit
  obtain ⟨narrow,wide⟩ := Bool.and_eq_true_iff.mp read
  obtain ⟨bounds,first⟩ := Bool.and_eq_true_iff.mp narrow
  exact ⟨safe bounds,first,fun yes => by simpa [yes] using wide⟩

end NeuralReversi.PvsParallel.NativeSplit.AlphaLoads

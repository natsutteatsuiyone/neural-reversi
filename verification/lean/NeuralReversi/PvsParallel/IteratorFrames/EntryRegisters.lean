import NeuralReversi.PvsParallel.IteratorFrames.ChildSites
import NeuralReversi.PvsParallel.NativeFrames.EntryStream

namespace NeuralReversi.PvsParallel.IteratorFrames

open NativeFrames.EntryStream (Entry)
open NativeWorkerFeed (returns received completeControl)
open SplitWorkerMir (scoutQuery wideQuery retry)
open MoveList.ConcurrentTrace (Write State)


/-- Execute the child-call ABI for each completed ticket. Incomplete calls can
    still have entry snapshots in the surrounding frame stream. -/
def ticketEntries (b : Board.Board) (rc : List Move.Move) (journal : List Observation)
    (stream : List NativeFrames.Record) (op : Write w) : List Entry := Id.run do
  let some mv := rc[op.old.toNat]? | return []
  let some (_,o) := findLock journal op.worker op.opening | return []
  let some rs := returns op.worker stream | return []
  if !completeControl o.prefix o.task rs o.alpha (fun wide => o.stamp (.load wide)) o.ends o.acquired o.locked.stopped then return []
  let i := received o.task o.prefix rs op.old o.alpha o.fresh
  let scout := ((op.worker,o.scoutOpening),childArgs b mv (scoutQuery i))
  let wide := ((op.worker,o.wideOpening),childArgs b mv (wideQuery i))
  return scout :: (if retry i then [wide] else [])

def entries (b : Board.Board) (rc : List Move.Move) (journal : List Observation)
    (stream : List NativeFrames.Record) (ops : List (Write w)) : List Entry :=
  ops.flatMap (ticketEntries b rc journal stream)

theorem Posted.operands {s : State w} (x : Posted rc s journal stream earlier r seq) (b : Board.Board)
    (args : ∀ e ∈ entries b rc journal stream s.writes, inputs e.1.1 e.1.2 = some e.2) :
    EntryOperands inputs b x := by
  have atTicket := fun e h => args e (List.mem_flatMap.mpr ⟨x.op,x.issued,h⟩)
  have emits : ticketEntries b rc journal stream x.op =
      ((x.op.worker,x.lock.scoutOpening),childArgs b x.move (scoutQuery x.input)) ::
      (if retry x.input then [((x.op.worker,x.lock.wideOpening),childArgs b x.move (wideQuery x.input))] else []) := by
    simp [ticketEntries,x.cell,x.foundLock,x.returns,x.control,Posted.input]
  exact ⟨atTicket _ (emits ▸ List.mem_cons_self),
    fun again => atTicket _ (by rw [emits,ite_eq_left again]; exact List.Mem.tail _ List.mem_cons_self)⟩

end NeuralReversi.PvsParallel.IteratorFrames

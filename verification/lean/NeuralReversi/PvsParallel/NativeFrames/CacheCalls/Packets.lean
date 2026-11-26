import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Schedule

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls

open EndgameSearch.CacheHistory (Pool Local)
open EndgameSearch.CacheHistory.Payloads (Instruction Values Op packet observeRun)
open EndgameSearch.CacheOwner (Thread)

private theorem Access.observation (op : Access) (p : Pool) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) :
    let sample := op.instruction.observe p s shape
    sample.table = op.tableKind.1 ∧ sample.data.kind = op.tableKind.2 := by
  cases op <;> exact ⟨rfl,rfl⟩

/-- The value at a call's opening comes from the array at that point in the
    worker's execution. Values at unrelated worker/time pairs are retained. -/
def Issued.install (p : Pool) (w : Nat) (clock : Nat → Nat) (other : Values) (xs : List Issued)
    (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) : Values :=
  match xs with
  | [] => other
  | x::rest => fun worker opening =>
    let op := x.op.instruction
    if worker = w ∧ opening = clock x.index then
      some ⟨clock (x.index+1),(op.observe p s shape).data⟩
    else install p w clock other rest (op.step p s) (op.shape p s shape) worker opening

theorem Issued.read (p : Pool) (w : Nat) (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b)
    (other : Values) (xs : List Issued) (ordered : xs.Pairwise (fun a b => a.index < b.index))
    (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) (values : Values)
    (registered : ∀ x ∈ xs, values w (clock x.index) = install p w clock other xs s shape w (clock x.index)) :
    (xs.map (Issued.request clock)).mapM (packet values w) =
      some (observeRun p s shape (xs.map (fun x => x.op.instruction))) := by
  induction xs generalizing s with
  | nil => rfl
  | cons x rest ih =>
    obtain ⟨first,more⟩ := List.pairwise_cons.mp ordered
    let op := x.op.instruction
    have atFirst := registered x List.mem_cons_self
    have head : packet values w (x.request clock) = some (op.observe p s shape) := by
      have tags := x.op.observation p s shape
      simp only [Issued.install,ite_true,true_and] at atFirst
      unfold packet
      simp [request,atFirst,tags.2,←tags.1,op]
    have remainder := ih more (op.step p s) (op.shape p s shape) (fun y hy => by
      have atY := registered y (List.mem_cons_of_mem _ hy)
      simpa [install,Nat.ne_of_gt (time (first y hy)),op] using atY)
    simp only [List.map_cons,List.mapM_cons,observeRun,head,remainder]
    rfl

theorem Program.read_packets (code : Program node h last) (th : Thread) (i : Nat) (p : Pool) (w : Nat)
    (clock : Nat → Nat) (time : ∀ {a b}, a < b → clock a < clock b) (other : Values)
    (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) :
    let xs := code.schedule th i
    let values := Issued.install p w clock other xs s shape
    (xs.map (Issued.request clock)).mapM (packet values w) = some (observeRun p s shape (code.accesses.map Access.instruction)) := by
  have observed := Issued.read p w clock time other (code.schedule th i) (code.schedule_order th i).1 s shape _ (fun _ _ => rfl)
  rw [←code.schedule_ops th i,List.map_map]
  exact observed

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls

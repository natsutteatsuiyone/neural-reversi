module
public import NeuralReversi.PvsParallel.NativeFrames.CacheCalls.Packets
public import NeuralReversi.EndgameSearch.CacheHistory.Payloads.Workers

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeFrames.CacheCalls

open EndgameSearch.CacheHistory (Pool Local Timeline Visit)
open EndgameSearch.CacheHistory.NativeTrace (Worker)
open EndgameSearch.CacheHistory.Payloads (Values Interval)

/-- All cache calls from one worker's visit, with the physical array state at entry. -/
structure Block (p : Pool) where
  window : Interval
  clock : Nat → Nat
  ops : List Issued
  state : Local
  shape : ∀ t, (state.cells t).size = p.sizes t
  inside : ∀ x ∈ ops, window.first ≤ clock x.index ∧ clock x.index ≤ window.last

def Block.earlier (a b : Block p) := a.window.worker = b.window.worker → a.window.last < b.window.first

def Block.install (b : Block p) (other : Values) :=
  Issued.install p b.window.worker b.clock other b.ops b.state b.shape

def Block.registers (blocks : List (Block p)) (other : Values) : Values :=
  blocks.foldr Block.install other

private theorem Issued.unassigned (xs : List Issued) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t)
    (clock : Nat → Nat) (other : Values) (outside : ∀ x ∈ xs, ¬ (w = worker ∧ key = clock x.index)) :
    install p worker clock other xs s shape w key = other w key := by
  induction xs generalizing s with
  | nil => rfl
  | cons x rest ih =>
    simp only [install,ite_eq_right (outside x List.mem_cons_self)]
    exact ih _ _ (fun y hy => outside y (List.mem_cons_of_mem _ hy))

private theorem Issued.assigned (xs : List Issued) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t)
    (clock : Nat → Nat) (left right : Values) (x : Issued) (member : x ∈ xs) :
    install p w clock left xs s shape w (clock x.index) = install p w clock right xs s shape w (clock x.index) := by
  induction xs generalizing s with
  | nil => contradiction
  | cons y rest ih =>
    by_cases eq : clock x.index = clock y.index
    · simp [install,eq]
    · have mem := List.mem_cons.mp member
      simp only [install,true_and,ite_eq_right eq]
      exact ih _ _ (mem.resolve_left (fun e => eq (e ▸ rfl)))

/-- Visits may overlap across workers. Within one worker only the owning
    visit can supply the value at this call's opening. -/
theorem Block.registered (blocks : List (Block p)) (ordered : blocks.Pairwise Block.earlier)
    (other localValues : Values) (b : Block p) (hb : b ∈ blocks) (x : Issued) (hx : x ∈ b.ops) :
    registers blocks other b.window.worker (b.clock x.index) =
      b.install localValues b.window.worker (b.clock x.index) := by
  induction blocks with
  | nil => contradiction
  | cons a rest ih =>
    obtain ⟨first,more⟩ := List.pairwise_cons.mp ordered
    rcases List.mem_cons.mp hb with rfl | tail
    · exact Issued.assigned _ _ _ _ _ _ _ hx
    · change a.install (registers rest other) _ _ = _
      rw [show a.install (registers rest other) b.window.worker (b.clock x.index) = _ from
        Issued.unassigned a.ops a.state a.shape a.clock (registers rest other) (fun y hy clash => by
          have before := first b tail clash.1.symm
          have upper := (a.inside y hy).2
          have lower := (b.inside x hx).1
          omega)]
      exact ih more tail

def Program.block (code : Program node h last) (w : Worker p) (clock : Nat → Nat)
    (time : ∀ {a b}, a < b → clock a < clock b) (n : Nat) (s : Local) (shape : ∀ t, (s.cells t).size = p.sizes t) : Block p where
  window := ⟨w.ordinal,clock n,clock (n+5+(code.forest.calls (thread w)).count)⟩
  clock := clock
  ops := code.schedule (thread w) (n+4)
  state := s
  shape := shape
  inside x hx := by
    obtain ⟨lo,hi⟩ := (code.schedule_order (thread w) (n+4)).2 x hx
    exact ⟨Nat.le_of_lt (time (by omega)),Nat.le_of_lt (time (by omega))⟩

/-- Installing one visit over the merged register file leaves every key intact.
    This also holds when the observed stream stops inside one of its calls. -/
theorem Block.install_registers (blocks : List (Block p)) (ordered : blocks.Pairwise Block.earlier)
    (other : Values) (b : Block p) (member : b ∈ blocks) :
    b.install (registers blocks other) = registers blocks other := by
  funext worker key
  by_cases present : ∃ x ∈ b.ops, worker = b.window.worker ∧ key = b.clock x.index
  · obtain ⟨x,hx,rfl,rfl⟩ := present
    exact (registered blocks ordered other _ b member x hx).symm
  · exact Issued.unassigned _ _ _ _ _ (fun x hx atKey => present ⟨x,hx,atKey⟩)

theorem Block.visit_order (timeline : Timeline p) (blocks : Visit p → Block p)
    (atVisit : ∀ v, (blocks v).window = ⟨v.worker,v.first,v.last⟩) :
    (timeline.visits.map blocks).Pairwise Block.earlier := by
  apply List.pairwise_map.mpr
  exact timeline.ordered.imp (fun {a b} earlier => by simpa [Block.earlier,EndgameSearch.CacheHistory.Earlier,atVisit] using earlier)

end NeuralReversi.PvsParallel.NativeFrames.CacheCalls

module
public import NeuralReversi.EndgameSearch.CacheHistory.Local

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheHistory

open CacheOwner (EntryPoint Thread Resource Descends)
open EndgameTreeCache (Table Store Address)
open RecursionCacheMir (Reads Steps)

private def canStore : CacheOwner.Node → Bool
  | .deep | .deepChild | .shallow | .shallowMoves | .shallowChild
  | .deepStore | .shallowStore => true
  | _ => false

private theorem storeEntry {e : CacheOwner.Edge} (inside : e ∈ CacheOwner.transfers)
    (stores : canStore e.callee = true) : canStore e.caller = true ∨ e.caller = .boundary := by
  have checks : CacheOwner.transfers.all (fun edge =>
      !canStore edge.callee || canStore edge.caller || edge.caller == .boundary) = true := by decide
  simpa [stores] using List.all_eq_true.mp checks e inside

private theorem crossed_boundary {outside initial arg dest value}
    (source : canStore initial = false) (path : Descends outside initial arg dest value)
    (stores : canStore dest = true) :
    ∃ receiver, Descends outside initial arg .boundary receiver ∧ Descends outside .boundary receiver dest value := by
  induction path with
  | same => simp_all
  | call parent member actual ih =>
    rcases storeEntry member stores with borrowed | started
    · obtain ⟨receiver,first,tail⟩ := ih borrowed
      exact ⟨receiver,first,.call tail member actual⟩
    · exact ⟨_,started ▸ parent,.call (started ▸ Descends.same) member actual⟩

/-- Non-leaf frames have no additional local-cache writes. Every nested store descends
    from a boundary event using the thread that was captured by the native spawn. -/
theorem store_inside_boundary {kind base foreign outside receiver}
    (table : Table) (path : Descends outside (CacheOwner.entry kind)
      (.thread (CacheOwner.start (CacheOwner.programFor kind) base foreign).running) (CacheOwner.storeNode table) receiver) :
    ∃ localThread, Descends outside (CacheOwner.entry kind)
      (.thread (CacheOwner.start (CacheOwner.programFor kind) base foreign).running) .boundary localThread ∧
      Descends outside .boundary localThread (CacheOwner.storeNode table) receiver := by
  apply crossed_boundary (path:=path)
  · cases kind <;> rfl
  · cases table <;> rfl

structure Pool where
  size : Nat
  width : Nat
  word : 32 ≤ width
  key : Board.Board → BitVec 64
  base : Nat
  foreign : Thread
  outside : Thread

def Pool.sizes (p : Pool) (t : Table) : Nat := CacheMir.budget p.size t / 18

def Pool.address (p : Pool) : Address := CacheMir.address p.width p.sizes p.key

def Pool.empty (p : Pool) : Local := ⟨[],CacheState.empty p.sizes⟩

theorem Pool.fresh (p : Pool) : Good mc fc p.width p.sizes p.key p.empty ∧
    (∀ t, CacheMir.newCache (CacheMir.budget p.size t) = some (p.empty.cells t)) := by
  have init := CacheMir.initialize_threads p.width p.word (fun _ => p.size) p.key
  obtain ⟨shape,arrays⟩ := init.2 0
  exact ⟨⟨fun _ e => (List.not_mem_nil e).elim,shape,arrays⟩,init.1 0⟩

/-- One complete call of EndGameStrategy::shallow_search. A recursive deep/shallow child is
    already included in its parent's write sequence. A discarded return still has a finish. -/
structure Visit (p : Pool) where
  worker : Nat
  kind : EntryPoint
  starts : (CacheOwner.poolPlan p.size)[worker]? = some kind
  receiver : Resource
  called : Descends p.outside (CacheOwner.entry kind)
    (.thread (CacheOwner.start (CacheOwner.programFor kind) (p.base+3*worker) p.foreign).running)
    .boundary receiver
  frame : Nat
  first : Nat
  last : Nat
  inOrder : first ≤ last
  scoredAt : Option Nat
  scoreAfter : ∀ step, scoredAt = some step → last ≤ step
  input : Input

theorem Visit.registered (v : Visit p) :
    ∃ endpoint, (CacheOwner.starts p.foreign (CacheOwner.poolPlan p.size) p.base)[v.worker]? = some endpoint ∧
      v.receiver = .thread endpoint.registry := by
  refine ⟨_, CacheOwner.scheduled v.starts p.foreign p.base, ?_⟩
  rw [(CacheOwner.start_identity v.kind (p.base+3*v.worker) p.foreign).1,
      (CacheOwner.start_identity v.kind (p.base+3*v.worker) p.foreign).2.1]
  exact CacheOwner.native_receiver v.called

def Earlier (a b : Visit p) : Prop := a.worker = b.worker → a.last < b.first

structure Timeline (p : Pool) where
  visits : List (Visit p)
  ordered : visits.Pairwise Earlier

def byOwner (visits : List (Visit p)) (owner : Nat) := visits.filter (fun v => v.worker == owner)

def beforeStep (visits : List (Visit p)) (owner step : Nat) :=
  visits.filter (fun v => v.worker == owner && decide (v.last < step))

theorem before_visit (before after : List (Visit p)) (v : Visit p)
    (sorted : (before ++ v :: after).Pairwise Earlier) :
    beforeStep (before ++ v :: after) v.worker v.first = byOwner before v.worker := by
  obtain ⟨_,tail,cross⟩ := List.pairwise_append.mp sorted
  have later := (List.pairwise_cons.mp tail).1
  let wanted := fun x : Visit p => x.worker == v.worker && decide (x.last < v.first)
  have prev : before.filter wanted = byOwner before v.worker := by
    apply List.filter_congr
    intro x mem
    by_cases same : x.worker = v.worker
    · have time := cross x mem v (by simp) same
      simp [wanted,same,time]
    · simp [wanted,same]
  have rest : after.filter wanted = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro x mem
    by_cases same : x.worker = v.worker
    · have time := later x mem same.symm
      have a := v.inOrder
      have b := x.inOrder
      have no : ¬ x.last < v.first := by omega
      simp [wanted,same,no]
    · simp [wanted,same]
  have notSelf : wanted v = false := by
    have no := Nat.not_lt.mpr v.inOrder
    simp [wanted,no]
  change List.filter wanted _ = _
  simp [List.filter_append,notSelf,prev,rest]

def replay (mc : Bitboard.Moves.X86Config) (fc : FlipConfig) (p : Pool) (calls : List (Visit p)) : Local :=
  calls.foldl (fun state v => v.input.after mc fc p.address state) p.empty

theorem replay_good (mc : Bitboard.Moves.X86Config) (fc : FlipConfig) (p : Pool) (calls : List (Visit p)) :
    Good mc fc p.width p.sizes p.key (replay mc fc p calls) := by
  have step (initial : Local) (g : Good mc fc p.width p.sizes p.key initial) :
      Good mc fc p.width p.sizes p.key (calls.foldl (fun state v => v.input.after mc fc p.address state) initial) := by
    induction calls generalizing initial with
    | nil => exact g
    | cons v vs ih => exact ih _ (after_call v.input g (CacheMir.budget_fits p.size p.width p.word)).1
  exact step p.empty p.fresh.1

def Timeline.readAt (trace : Timeline p) (mc : Bitboard.Moves.X86Config) (fc : FlipConfig) (owner step : Nat) : Local :=
  replay mc fc p (beforeStep trace.visits owner step)

theorem Timeline.reads (trace : Timeline p) (mc : Bitboard.Moves.X86Config) (fc : FlipConfig) (owner step : Nat) :
    let prior := trace.readAt mc fc owner step
    EndgameTreeCache.soundHistory mc fc prior.history ∧ Reads p.width p.key p.address prior.history prior.cells :=
  let g := replay_good mc fc p (beforeStep trace.visits owner step)
  ⟨g.valid,g.reads⟩

theorem Timeline.after_visit (trace : Timeline p) (mc : Bitboard.Moves.X86Config) (fc : FlipConfig)
    (before after : List (Visit p)) (v : Visit p) (atVisit : trace.visits = before ++ v :: after) :
    let prior := trace.readAt mc fc v.worker v.first
    let next := replay mc fc p (byOwner (before ++ [v]) v.worker)
    next = v.input.after mc fc p.address prior ∧ Good mc fc p.width p.sizes p.key next ∧
      ∃ ws, prior.history ++ ws = next.history ∧ Steps p.width p.key p.address prior.history prior.cells ws next.cells := by
  have order := trace.ordered
  rw [atVisit] at order
  have prefixEq := before_visit before after v order
  have eq : trace.readAt mc fc v.worker v.first = replay mc fc p (byOwner before v.worker) := by
    simp [Timeline.readAt,atVisit,prefixEq]
  have append : byOwner (before ++ [v]) v.worker = byOwner before v.worker ++ [v] := by
    simp [byOwner]
  simp only [eq,append,replay,List.foldl_append,List.foldl_cons,List.foldl_nil]
  exact ⟨trivial, after_call v.input (replay_good mc fc p (byOwner before v.worker)) (CacheMir.budget_fits p.size p.width p.word)⟩

end NeuralReversi.EndgameSearch.CacheHistory

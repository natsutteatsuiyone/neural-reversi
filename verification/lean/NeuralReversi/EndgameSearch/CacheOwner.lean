module
public import NeuralReversi.EndgameSearch.CacheOwner.Program
public import NeuralReversi.EndgameSearch.CacheMir

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheOwner

structure Pair where
  deep : Nat
  shallow : Nat
  deriving DecidableEq

structure Thread where
  object : Nat
  tables : Pair
  deriving DecidableEq

structure Endpoint where
  kind : EntryPoint
  registry : Thread
  running : Thread

def Handle.resolve (fresh returned captured foreign : Thread) : Handle → Thread
  | .newArc => fresh | .returned => returned | .captured => captured | .foreign => foreign

/-- Fresh allocation identities are numbered in construction order while the objects are live. -/
def start (program : Boot) (base : Nat) (foreign : Thread) : Endpoint :=
  let pair := if program.pair = .fresh then ⟨base,base + 1⟩ else foreign.tables
  let fresh : Thread := ⟨base + 2,pair⟩
  let registered := program.registered.resolve fresh foreign foreign foreign
  let returned := program.returned.resolve fresh foreign foreign foreign
  let capture := program.capture.resolve fresh returned foreign foreign
  ⟨program.kind,registered,program.enter.resolve fresh returned capture foreign⟩

def programFor : EntryPoint → Boot
  | .main => bootMain | .worker => bootWorker

theorem start_identity (kind : EntryPoint) (base : Nat) (foreign : Thread) :
    let e := start (programFor kind) base foreign
    e.registry = e.running ∧ e.running = ⟨base+2,base,base+1⟩ ∧ e.kind = kind := by
  cases kind <;> exact ⟨rfl,rfl,rfl⟩

inductive Resource where
  | thread (value : Thread) | pair (value : Pair) | table (storage : Nat) | none
  deriving DecidableEq

def Flow.apply (outside : Thread) : Flow → Resource → Option Resource
  | .forward, value => some value
  | .borrowPair, .thread w => some (.pair w.tables)
  | .borrowShallow, .thread w => some (.table w.tables.shallow)
  | .deepField, .pair p => some (.table p.deep)
  | .shallowField, .pair p => some (.table p.shallow)
  | .callback, _ => some .none
  | .foreign, _ => some (.thread outside)
  | _,_ => none

inductive Kind where
  | thread | pair | deep | shallow | none
  deriving DecidableEq

def Node.kind : Node → Kind
  | .failLow | .deep | .deepChild => .pair
  | .deepSlot | .deepProbe | .deepStore => .deep
  | .shallow | .shallowMoves | .shallowChild | .shallowSlot | .shallowProbe | .shallowStore => .shallow
  | .progress => .none
  | _ => .thread

def Kind.resource (w : Thread) : Kind → Resource
  | .thread => .thread w | .pair => .pair w.tables | .deep => .table w.tables.deep
  | .shallow => .table w.tables.shallow | .none => .none

def typed (e : Edge) : Bool :=
  match e.flow with
  | .forward => e.caller.kind == e.callee.kind
  | .borrowPair => e.caller.kind == .thread && e.callee.kind == .pair
  | .borrowShallow => e.caller.kind == .thread && e.callee.kind == .shallow
  | .deepField => e.caller.kind == .pair && e.callee.kind == .deep
  | .shallowField => e.caller.kind == .pair && e.callee.kind == .shallow
  | .callback => e.callee.kind == .none
  | .foreign => false

private theorem edge_typed {e : Edge} (inside : e ∈ transfers) : typed e = true :=
  List.all_eq_true.mp (show transfers.all typed = true by decide) _ inside

theorem flows {e : Edge} (inside : e ∈ transfers) (w outside : Thread) :
    e.flow.apply outside (e.caller.kind.resource w) = some (e.callee.kind.resource w) := by
  have h := edge_typed inside
  cases e with
  | mk caller n callee flow =>
    cases flow <;> simp [typed] at h ⊢
    all_goals first | simp [h, Flow.apply, Kind.resource] | simp [h.1,h.2, Flow.apply, Kind.resource]

def entry : EntryPoint → Node | .main => .main | .worker => .idle

/-- Each step enters a synchronous callee with the MIR operand chosen at the call site. -/
inductive Descends (outside : Thread) : Node → Resource → Node → Resource → Prop
  | same : Descends outside n r n r
  | call {n r e value arg} :
      Descends outside n r e.caller value → e ∈ transfers → e.flow.apply outside value = some arg →
      Descends outside n r e.callee arg

theorem callee_resource {outside w n t r}
    (path : Descends outside n (n.kind.resource w) t r) : r = t.kind.resource w := by
  generalize initial : n.kind.resource w = argument at path
  induction path with
  | same => exact initial.symm
  | call shorter member passed ih =>
    rw [ih,flows member] at passed
    exact (Option.some.inj passed).symm

theorem native_receiver {outside kind base foreign node resource}
    (path : Descends outside (entry kind) (.thread (start (programFor kind) base foreign).running) node resource) :
    resource = node.kind.resource ⟨base+2,base,base+1⟩ := by
  have actual := (start_identity kind base foreign).2.1
  rw [actual] at path
  cases kind <;> exact callee_resource path

def hasBorrow : Kind → Bool
  | .pair | .deep | .shallow => true | _ => false

private theorem retained {e : Edge} (member : e ∈ transfers) :
    hasBorrow e.caller.kind = true → hasBorrow e.callee.kind = true := by
  have all : transfers.all (fun e => !hasBorrow e.caller.kind || hasBorrow e.callee.kind) = true := by decide
  intro held
  simpa [held] using List.all_eq_true.mp all e member

theorem no_reentry {outside n arg m received} (underLoan : hasBorrow n.kind = true)
    (path : Descends outside n arg m received) :
    m ≠ .access ∧ m ≠ .split ∧ m ≠ .idle ∧ m ≠ .progress := by
  have live : hasBorrow m.kind = true := by
    induction path with
    | same => exact underLoan
    | call _ member _ ih => exact retained member ih
  cases m <;> first | decide | contradiction

/-- The boundary invokes its getter once (the MIR getter block is outside every cycle).
    Every other sensitive call in the same activation is a descendant of that loan. -/
theorem boundary_call {e : Edge} (inside : e ∈ transfers) (fromBoundary : e.caller = .boundary) :
    e.callee = .access ∨ hasBorrow e.callee.kind = true := by
  have valid : transfers.all (fun edge =>
      edge.caller != .boundary || edge.callee == .access || hasBorrow edge.callee.kind) = true := by decide
  have h := List.all_eq_true.mp valid e inside
  simpa [fromBoundary] using h

theorem ecCutoff_call {e : Edge} (inside : e ∈ transfers) (fromCutoff : e.caller = .ecCutoff) :
    e.callee = .access ∨ hasBorrow e.callee.kind = true := by
  have valid : transfers.all (fun edge =>
      edge.caller != .ecCutoff || edge.callee == .access || hasBorrow edge.callee.kind) = true := by decide
  have h := List.all_eq_true.mp valid e inside
  simpa [fromCutoff] using h

def chosen (shallow : Bool) : Kind := if shallow then .shallow else .deep

theorem separate_allocations {a b : Nat} (before : a + 3 ≤ b) (l r : Bool)
    {lk rk outside foreign node node' x y}
    (path : Descends outside (entry lk) (.thread (start (programFor lk) a foreign).running) node x)
    (path' : Descends outside (entry rk) (.thread (start (programFor rk) b foreign).running) node' y)
    (table : node.kind = chosen l) (table' : node'.kind = chosen r) : x ≠ y := by
  rw [native_receiver path, native_receiver path', table, table']
  cases l <;> cases r <;> simp [chosen, Kind.resource] <;> omega

/-- A fresh constructor reserves the two owned buffers and their enclosing thread object.
    Starting identities enumerate live allocations and do not encode physical addresses. -/
def starts (foreign : Thread) : List EntryPoint → Nat → List Endpoint
  | [], _ => []
  | k :: ks, base => start (programFor k) base foreign :: starts foreign ks (base+3)

def poolPlan (size : Nat) : List EntryPoint := .main :: List.replicate (size-1) .worker

theorem scheduled {kinds : List EntryPoint} {i k} (atIndex : kinds[i]? = some k)
    (foreign : Thread) (base : Nat) :
    (starts foreign kinds base)[i]? = some (start (programFor k) (base+3*i) foreign) := by
  induction kinds generalizing i base with
  | nil => simp at atIndex
  | cons head tail ih =>
    cases i with
    | zero =>
      simp at atIndex
      simp [starts, atIndex]
    | succ n =>
      simp only [List.getElem?_cons_succ] at atIndex
      simp only [starts, List.getElem?_cons_succ, ih atIndex]
      congr 2 <;> omega

/-- The two operations can belong to any live worker builds, including distinct pools.
    Registries and stack entries use the exact thread captured by their own spawn. -/
theorem scheduled_tables_disjoint {kinds : List EntryPoint} {i j : Nat} {ki kj foreign base outside a b l r na nb}
    (left : kinds[i]? = some ki) (right : kinds[j]? = some kj) (distinct : i ≠ j)
    {wi wj : Endpoint} (xi : (starts foreign kinds base)[i]? = some wi)
    (xj : (starts foreign kinds base)[j]? = some wj)
    (p : Descends outside (entry ki) (.thread wi.running) na a)
    (q : Descends outside (entry kj) (.thread wj.running) nb b)
    (ka : na.kind = chosen l) (kb : nb.kind = chosen r) : a ≠ b := by
  have ei := scheduled left foreign base
  have ej := scheduled right foreign base
  rw [xi] at ei
  rw [xj] at ej
  cases Option.some.inj ei
  cases Option.some.inj ej
  by_cases lt : i < j
  · exact separate_allocations (by omega) l r p q ka kb
  · exact Ne.symm (separate_allocations (by omega) r l q p kb ka)

def storeNode : EndgameTreeCache.Table → Node
  | .deep => .deepStore | .shallow => .shallowStore

def storage (tables : Pair) : EndgameTreeCache.Table → Nat
  | .deep => tables.deep | .shallow => tables.shallow

/-- An invoked Rust store carries a receiver, board, window and score. Its owner and table
    are recovered from the spawn ordinal and call target. -/
structure StoreCall (kinds : List EntryPoint) (base : Nat) (foreign outside : Thread) where
  ordinal : Nat
  kind : EntryPoint
  started : kinds[ordinal]? = some kind
  table : EndgameTreeCache.Table
  value : EndgameCache.Write
  receiver : Resource
  path : Descends outside (entry kind)
    (.thread (start (programFor kind) (base+3*ordinal) foreign).running) (storeNode table) receiver

def StoreCall.effect (call : StoreCall kinds base foreign outside) : CacheMir.OwnedWrite :=
  ⟨base+3*call.ordinal+2, call.table, call.value⟩

theorem StoreCall.registered_receiver (call : StoreCall kinds base foreign outside) :
    ∃ endpoint, (starts foreign kinds base)[call.ordinal]? = some endpoint ∧
      endpoint.registry.object = call.effect.owner ∧
      call.receiver = .table (storage endpoint.registry.tables call.effect.event.table) := by
  refine ⟨_, scheduled call.started foreign base, ?_⟩
  have owner := start_identity call.kind (base+3*call.ordinal) foreign
  rw [owner.1,owner.2.1]
  refine ⟨rfl,?_⟩
  rw [native_receiver call.path]
  cases t : call.table <;> simp [StoreCall.effect,t,storeNode,storage,Node.kind,Kind.resource]

theorem interleaved_native_stores (calls : List (StoreCall kinds base foreign outside))
    (fit : ∀ owner, CacheMir.Fits width (sizes owner))
    (before : CacheMir.Aligned width sizes key histories cells) :
    let writes := calls.map StoreCall.effect
    ∃ after, writes.foldlM (CacheMir.writeOwned width key) cells = some after ∧
      CacheMir.Aligned width sizes key (writes.foldl CacheMir.appendOwned histories) after :=
  CacheMir.interleaved _ fit before

end NeuralReversi.EndgameSearch.CacheOwner



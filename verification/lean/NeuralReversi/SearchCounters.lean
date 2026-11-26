import Lean

namespace NeuralReversi.SearchCounters

abbrev Counter := BitVec 64
abbrev Diagnostic := Fin 10
abbrev Stats := Diagnostic → Counter

structure Snapshot where
  nodes : Counter
  stats : Stats

def withNodes (enabled : Bool) (nodes : Counter) (stats : Stats) : Snapshot :=
  ⟨nodes, if enabled then stats else fun _ => 0⟩

def merge (enabled : Bool) (current other : Snapshot) : Snapshot :=
  withNodes enabled (current.nodes + other.nodes) (fun i => current.stats i + other.stats i)

def increment (enabled : Bool) (stats : Stats) (slot : Diagnostic) : Stats :=
  if enabled then fun i => if i = slot then stats i + 1 else stats i else fun _ => 0

def saturatingSub (current previous : Counter) : Counter :=
  BitVec.ofNat 64 (current.toNat - previous.toNat)

def deltaSince (current previous : Snapshot) : Snapshot :=
  ⟨saturatingSub current.nodes previous.nodes,
    fun i => saturatingSub (current.stats i) (previous.stats i)⟩

def CounterMergeContract : Prop :=
  ∀ enabled current other,
    let total := merge enabled current other
    total.nodes.toNat = (current.nodes.toNat + other.nodes.toNat) % 2 ^ 64 ∧
    (current.nodes.toNat + other.nodes.toNat < 2 ^ 64 →
      total.nodes.toNat = current.nodes.toNat + other.nodes.toNat) ∧
    ∀ i, (total.stats i).toNat = if enabled then
      ((current.stats i).toNat + (other.stats i).toNat) % 2 ^ 64 else 0

theorem counterMergeContract_proven : CounterMergeContract := by
  intro enabled current other
  refine ⟨BitVec.toNat_add _ _, fun fits => BitVec.toNat_add_of_lt fits, fun i => ?_⟩
  cases enabled <;> simp [merge, withNodes, BitVec.toNat_add]

def CounterIncrementContract : Prop :=
  ∀ enabled nodes stats slot,
    let result := withNodes enabled nodes (increment enabled stats slot)
    result.nodes = nodes ∧
    (enabled = false → ∀ i, result.stats i = 0) ∧
    (enabled = true → result.stats slot = stats slot + 1 ∧
      ∀ i, i ≠ slot → result.stats i = stats i)

theorem counterIncrementContract_proven : CounterIncrementContract := by
  intro enabled nodes stats slot
  cases enabled <;> simp [withNodes, increment]

theorem saturatingSub_toNat (current previous : Counter) :
    (saturatingSub current previous).toNat = current.toNat - previous.toNat := by
  have h := Nat.lt_of_le_of_lt (Nat.sub_le current.toNat previous.toNat) current.isLt
  exact Nat.mod_eq_of_lt h

def CounterDeltaContract : Prop :=
  ∀ current previous,
    let delta := deltaSince current previous
    delta.nodes.toNat = current.nodes.toNat - previous.nodes.toNat ∧
    ∀ i, (delta.stats i).toNat = (current.stats i).toNat - (previous.stats i).toNat

theorem counterDeltaContract_proven : CounterDeltaContract := by
  intro current previous
  exact ⟨saturatingSub_toNat _ _, fun _ => saturatingSub_toNat _ _⟩

structure Context where
  nodes : Counter
  stats : Stats

def Context.counters (enabled : Bool) (ctx : Context) : Snapshot :=
  withNodes enabled ctx.nodes ctx.stats

def Context.visit (ctx : Context) : Context :=
  { ctx with nodes := ctx.nodes + 1 }

def Context.mergeCounters (enabled : Bool) (ctx : Context) (other : Snapshot) : Context :=
  ⟨ctx.nodes + other.nodes,
    if enabled then fun i => ctx.stats i + other.stats i else ctx.stats⟩

def ContextCounterContract : Prop :=
  ∀ enabled (ctx : Context) other,
    (ctx.counters enabled).nodes = ctx.nodes ∧
    ctx.visit.counters enabled = { ctx.counters enabled with nodes := ctx.nodes + 1 } ∧
    (ctx.mergeCounters enabled other).counters enabled = merge enabled (ctx.counters enabled) other

theorem contextCounterContract_proven : ContextCounterContract := by
  intro enabled ctx other
  cases enabled <;> exact ⟨rfl, rfl, rfl⟩

private theorem saturatingSub_add (current contribution : Counter)
    (fits : current.toNat + contribution.toNat < 2 ^ 64) :
    saturatingSub (current + contribution) current = contribution := by
  apply BitVec.eq_of_toNat_eq
  rw [saturatingSub_toNat, BitVec.toNat_add_of_lt fits]
  omega

def CounterContributionContract : Prop :=
  ∀ (current contribution : Snapshot),
    current.nodes.toNat + contribution.nodes.toNat < 2 ^ 64 →
    (∀ i, (current.stats i).toNat + (contribution.stats i).toNat < 2 ^ 64) →
    deltaSince (merge true current contribution) current = contribution

theorem counterContributionContract_proven : CounterContributionContract := by
  intro current contribution nodesFit statsFit
  have nodes := saturatingSub_add current.nodes contribution.nodes nodesFit
  have stats := funext fun i => saturatingSub_add (current.stats i) (contribution.stats i) (statsFit i)
  cases contribution
  simp_all [deltaSince, merge, withNodes]

structure ResultCounters where
  nodes : Counter
  counters : Snapshot

def fromSnapshot (snapshot : Snapshot) : ResultCounters :=
  ⟨snapshot.nodes, snapshot⟩

def quickResult (moveCount : Nat) : Option ResultCounters :=
  if moveCount = 0 then none
  else some ⟨BitVec.ofNat 64 moveCount, withNodes false (BitVec.ofNat 64 moveCount) (fun _ => 0)⟩

def addFallbackCost (searched : Snapshot) (fallback : Option ResultCounters) : Option ResultCounters :=
  fallback.map fun result =>
    let total := searched.nodes + result.nodes
    ⟨total, { searched with nodes := total }⟩

def resultNodes (result : Option ResultCounters) : Counter :=
  (result.map ResultCounters.nodes).getD 0

def resultSnapshot (result : Option ResultCounters) : Snapshot :=
  (result.map ResultCounters.counters).getD (withNodes false 0 (fun _ => 0))

def ResultCounterContract : Prop :=
  ∀ (searched : Snapshot) (moveCount : Nat),
    let quick := quickResult moveCount
    let cached := some (fromSnapshot (withNodes false 0 (fun _ => 0)))
    let completed := some (fromSnapshot searched)
    let resumed := addFallbackCost searched quick
    resultSnapshot completed = searched ∧
    resultSnapshot (addFallbackCost searched cached) = searched ∧
    resultNodes quick = BitVec.ofNat 64 moveCount ∧
    resultSnapshot quick = withNodes false (BitVec.ofNat 64 moveCount) (fun _ => 0) ∧
    (moveCount = 0 → resumed = none) ∧
    (moveCount ≠ 0 → resultSnapshot resumed = { searched with
      nodes := searched.nodes + BitVec.ofNat 64 moveCount }) ∧
    ∀ result ∈ [none, quick, cached, completed, resumed, addFallbackCost searched cached],
      resultNodes result = (resultSnapshot result).nodes

theorem resultCounterContract_proven : ResultCounterContract := by
  intro searched moveCount
  cases searched with
  | mk nodes stats =>
    by_cases noMove : moveCount = 0
    · simp [noMove, quickResult, fromSnapshot, addFallbackCost, resultNodes, resultSnapshot, withNodes]
    · simp [noMove, quickResult, fromSnapshot, addFallbackCost, resultNodes, resultSnapshot, withNodes]

end NeuralReversi.SearchCounters

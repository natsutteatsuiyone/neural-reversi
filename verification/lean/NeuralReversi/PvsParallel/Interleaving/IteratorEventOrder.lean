module
public import NeuralReversi.PvsParallel.Interleaving.EventOrder
public import NeuralReversi.MoveList.ConcurrentExecution
public import NeuralReversi.Threading.AtomicWord
meta import NeuralReversi.MoveList.ConcurrentExecution
meta import NeuralReversi.PvsParallel.Interleaving.EventOrder
meta import NeuralReversi.Threading.AtomicWord

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.IteratorEventOrder

def reach (hb : α → α → Prop) (u v : α) := u = v ∨ hb u v

/-- A path in the union can be shortened to one strict iterator step between
    two possibly empty HB segments, since HB between tickets respects their ranks. -/
def augmented (hb : Fin n → Fin n → Prop) (ticket : Fin k → Fin n) (u v : Fin n) :=
  hb u v ∨ ∃ i j, reach hb u (ticket i) ∧ i < j ∧ reach hb (ticket j) v

instance {hb : Fin n → Fin n → Prop} [DecidableRel hb] {ticket : Fin k → Fin n} :
    DecidableRel (augmented hb ticket) := fun u v => by unfold augmented reach; infer_instance

variable {hb : Fin n → Fin n → Prop} {ticket : Fin k → Fin n}

private theorem reachTrans (trans : ∀ {a b c}, hb a b → hb b c → hb a c)
    (left : reach hb u v) (right : reach hb v w) : reach hb u w := by
  rcases left with rfl | x
  · exact right
  rcases right with rfl | y
  · exact Or.inr x
  · exact Or.inr (trans x y)

private theorem reachRank (unique : Function.Injective ticket)
    (coherent : ∀ i j, hb (ticket i) (ticket j) → i < j)
    (path : reach hb (ticket i) (ticket j)) : i ≤ j := by
  rcases path with eq | edge
  · exact Fin.le_of_eq (unique eq)
  · exact Fin.le_of_lt (coherent _ _ edge)

private theorem irrefl (unique : Function.Injective ticket) (noLoop : ∀ u, ¬ hb u u)
    (trans : ∀ {a b c}, hb a b → hb b c → hb a c)
    (coherent : ∀ i j, hb (ticket i) (ticket j) → i < j) :
    ∀ u, ¬ augmented hb ticket u u := by
  rintro u (self | ⟨i,j,left,lt,right⟩)
  · exact noLoop u self
  · exact (Nat.not_lt_of_ge (reachRank unique coherent (reachTrans trans right left))) lt

private theorem transitive (unique : Function.Injective ticket)
    (trans : ∀ {a b c}, hb a b → hb b c → hb a c)
    (coherent : ∀ i j, hb (ticket i) (ticket j) → i < j) :
    ∀ {u v w}, augmented hb ticket u v → augmented hb ticket v w → augmented hb ticket u w := by
  rintro u v w (first | ⟨i,j,left,lt,right⟩) (last | ⟨p,q,begin_,ordered,end_⟩)
  · exact Or.inl (trans first last)
  · exact Or.inr ⟨p,q,reachTrans trans (Or.inr first) begin_,ordered,end_⟩
  · exact Or.inr ⟨i,j,left,lt,reachTrans trans right (Or.inr last)⟩
  · have bridge := reachRank unique coherent (reachTrans trans right begin_)
    exact Or.inr ⟨i,q,left,by omega,end_⟩

/-- A computable HB extension for one concrete iterator generation. -/
def ordered (hb : Fin n → Fin n → Prop) [DecidableRel hb] (ticket : Fin k → Fin n) :=
  EventOrder.events (augmented hb ticket)

/-- A single list preserves all HB edges (including worker program order and
    mutex handoffs) and reproduces one iterator's exact modification order.
    Two successful relaxed ticket claims need not themselves be HB-related. -/
theorem ordered_spec (hb : Fin n → Fin n → Prop) [DecidableRel hb] (ticket : Fin k → Fin n)
    (unique : Function.Injective ticket) (noLoop : ∀ u, ¬ hb u u)
    (trans : ∀ {a b c}, hb a b → hb b c → hb a c) (coherent : ∀ i j, hb (ticket i) (ticket j) → i < j) :
    let xs := ordered hb ticket
    xs.Perm (List.finRange n) ∧
      (∀ u v, hb u v → xs.idxOf u < xs.idxOf v) ∧
      xs.filter (fun e => (List.ofFn ticket).contains e) = List.ofFn ticket ∧
      ∀ (select : Fin n → Bool) (ys : List (Fin n)), (∀ e, select e = true ↔ e ∈ ys) →
        ys.Pairwise hb → xs.filter select = ys := by
  let rel := augmented hb ticket
  have irr := irrefl unique noLoop trans coherent
  have tr := @transitive _ _ hb ticket unique trans coherent
  refine ⟨EventOrder.permutation _,fun u v edge =>
    EventOrder.extendsHB rel irr tr (u := u) (v := v) (Or.inl edge),?_,?_⟩
  rotate_left
  · intro pick ys chosen chain
    exact EventOrder.projection rel irr tr pick ys chosen (chain.imp (fun edge => Or.inl edge))
  apply EventOrder.projection rel irr tr _ (List.ofFn ticket) (by simp)
  apply List.pairwise_iff_getElem.mpr
  intro i j hi hj lt
  simp only [List.getElem_ofFn]
  exact Or.inr ⟨⟨i,by simpa using hi⟩,⟨j,by simpa using hj⟩,Or.inl rfl,lt,Or.inl rfl⟩

theorem common (hb : Fin n → Fin n → Prop) [DecidableRel hb] (ticket : Fin k → Fin n)
    (unique : Function.Injective ticket) (noLoop : ∀ u, ¬ hb u u)
    (trans : ∀ {a b c}, hb a b → hb b c → hb a c) (coherent : ∀ i j, hb (ticket i) (ticket j) → i < j) :
    ∃ xs : List (Fin n), xs.Perm (List.finRange n) ∧
      (∀ u v, hb u v → xs.idxOf u < xs.idxOf v) ∧
      xs.filter (fun e => (List.ofFn ticket).contains e) = List.ofFn ticket := by
  have facts := ordered_spec hb ticket unique noLoop trans coherent
  exact ⟨ordered hb ticket,facts.1,facts.2.1,facts.2.2.1⟩

/-- The observed final loads follow every successful ticket. Incomparable
    exhausted readers may use the order chosen by the base HB extension. -/
def observations (hb : Fin n → Fin n → Prop) [DecidableRel hb] (ticket : Fin k → Fin n) (isNone : Fin n → Bool) :=
  List.ofFn ticket ++ (EventOrder.events hb).filter isNone

def withExhausted (hb : Fin n → Fin n → Prop) [DecidableRel hb] (ticket : Fin k → Fin n) (isNone : Fin n → Bool) :=
  ordered hb (observations hb ticket isNone).get

private theorem observationOrder (hb : Fin n → Fin n → Prop) [DecidableRel hb] (ticket : Fin k → Fin n)
    (isNone : Fin n → Bool) (unique : Function.Injective ticket) (noLoop : ∀ u, ¬ hb u u)
    (trans : ∀ {a b c}, hb a b → hb b c → hb a c) (coherent : ∀ i j, hb (ticket i) (ticket j) → i < j)
    (separate : ∀ i, isNone (ticket i) = false) (finalRead : ∀ e, isNone e = true → ∀ i, ¬hb e (ticket i)) :
    (observations hb ticket isNone).Pairwise (fun a b => a ≠ b ∧ ¬ hb b a) := by
  apply List.pairwise_append.mpr
  refine ⟨?_,?_,?_⟩
  · apply List.pairwise_iff_getElem.mpr
    intro i j hi hj lt
    simp only [List.getElem_ofFn]
    exact ⟨fun eq => by have := Fin.mk.inj (unique eq); omega,fun h => by have back : j < i := coherent _ _ h; omega⟩
  · let xs := EventOrder.events hb
    have distinct : xs.Nodup := (EventOrder.permutation hb).symm.nodup (List.nodup_finRange n)
    have order : xs.Pairwise (fun a b => ¬ hb b a) := by
      apply List.pairwise_iff_getElem.mpr
      intro i j hi hj lt backward
      have opposite := EventOrder.extendsHB hb noLoop trans backward
      change xs.idxOf xs[j] < xs.idxOf xs[i] at opposite
      rw [distinct.idxOf_getElem j hj,distinct.idxOf_getElem i hi] at opposite
      omega
    exact (distinct.and order).filter isNone
  · intro a member b exhausted
    obtain ⟨i,rfl⟩ := List.mem_ofFn.mp member
    have yes := (List.mem_filter.mp exhausted).2
    refine ⟨?_,finalRead _ yes i⟩
    rintro rfl
    rw [separate] at yes
    contradiction

/-- Read-write coherence for a load at the final modification rank excludes
    an HB edge from its exhausted read to any present successful CAS. -/
theorem finalRead_not_before {hb : α → α → Prop} (ticket : Fin k → α) (read : α)
    (readWrite : ∀ i, hb read (ticket i) → k < i.val + 1) (i : Fin k) : ¬hb read (ticket i) := by
  intro edge
  have := readWrite i edge
  omega

/-- A marker emitted after the terminal load can represent the exhausted return. -/
theorem finalReturn_not_before {hb : α → α → Prop} {read marker : α} (ticket : Fin k → α)
    (trans : ∀ {a b c}, hb a b → hb b c → hb a c) (returned : hb read marker)
    (readWrite : ∀ i, hb read (ticket i) → k < i.val + 1) (i : Fin k) : ¬hb marker (ticket i) :=
  fun edge => finalRead_not_before ticket read readWrite i (trans returned edge)

/-- The recorded load or final failed CAS which made next() return None supplies
    the final rank used by read-write coherence, even if earlier attempts read stale words. -/
theorem exhaustedReturn_not_before {hb : α → α → Prop} {read marker : α}
    (ops : List (MoveList.ConcurrentTrace.Write w)) (ticket : Fin ops.length → α)
    (run : MoveList.ConcurrentTrace.replay len start ops = some finalWord)
    (observed : MoveList.ConcurrentTrace.observed len (MoveList.ConcurrentTrace.words start ops) load failures = some seen)
    (stopped : MoveList.Concurrent.advance len seen = none)
    (trans : ∀ {a b c}, hb a b → hb b c → hb a c) (returned : hb read marker)
    (coherent : ∀ i, hb read (ticket i) → MoveList.ConcurrentTrace.lastSource load failures < i.val + 1) :
    ∀ i, ¬hb marker (ticket i) := by
  rw [MoveList.ConcurrentTrace.declined_source ops run observed stopped] at coherent
  exact finalReturn_not_before ticket trans returned coherent

theorem withExhausted_spec (hb : Fin n → Fin n → Prop) [DecidableRel hb] (ticket : Fin k → Fin n)
    (isNone : Fin n → Bool) (unique : Function.Injective ticket) (noLoop : ∀ u, ¬ hb u u)
    (trans : ∀ {a b c}, hb a b → hb b c → hb a c) (coherent : ∀ i j, hb (ticket i) (ticket j) → i < j)
    (separate : ∀ i, isNone (ticket i) = false) (finalRead : ∀ e, isNone e = true → ∀ i, ¬hb e (ticket i)) :
    let xs := withExhausted hb ticket isNone
    let projected := observations hb ticket isNone
    xs.Perm (List.finRange n) ∧ (∀ u v, hb u v → xs.idxOf u < xs.idxOf v) ∧
      xs.filter (fun e => projected.contains e) = projected ∧
      ∀ (select : Fin n → Bool) (ys : List (Fin n)), (∀ e, select e = true ↔ e ∈ ys) →
        ys.Pairwise hb → xs.filter select = ys := by
  let ws := observations hb ticket isNone
  have good := observationOrder hb ticket isNone unique noLoop trans coherent separate finalRead
  have dist : ws.Nodup := good.imp fun h => h.1
  have inject : Function.Injective ws.get := by
    intro i j eq
    have pos := congrArg ws.idxOf eq
    simp only [List.get_eq_getElem,dist.idxOf_getElem] at pos
    exact Fin.ext pos
  have compat (i j : Fin ws.length) (h : hb (ws.get i) (ws.get j)) : i < j := by
    by_cases goes : j.val < i.val
    · exact False.elim ((List.pairwise_iff_getElem.mp good _ _ j.isLt i.isLt goes).2 h)
    · by_cases same : i = j
      · subst j
        exact False.elim (noLoop _ h)
      · omega
  have all := ordered_spec hb ws.get inject noLoop trans compat
  have restore : List.ofFn ws.get = ws := List.ofFn_getElem
  simpa only [withExhausted,restore] using all

open Threading.AtomicWord (Event)
open MoveList.ConcurrentTrace (Write)

/-- The atomic which observed exhaustion can precede its recorded return marker.
    Its final instruction is either the initial load or a failed weak CAS. -/
structure ExhaustedRead (start : BitVec w) (len : Nat) (ops : List (Write w))
    (cas : Fin ops.length → Event) (before : Event → Event → Bool) (marker : Event) where
  read : Event
  load : Nat
  failures : List Nat
  seen : BitVec w
  observed : MoveList.ConcurrentTrace.observed len (MoveList.ConcurrentTrace.words start ops) load failures = some seen
  declined : MoveList.Concurrent.advance len seen = none
  returned : before read marker = true
  readWrite : ∀ i, before read (cas i) = true → MoveList.ConcurrentTrace.lastSource load failures < i.val + 1

def ProjectionContract := ∀ {w n len} (start finalWord : BitVec w) (ops : List (Write w))
    (site : Fin n → Event) (before : Event → Event → Bool) (ticket : Fin ops.length → Fin n)
    (exhausted : Fin n → Bool), MoveList.ConcurrentTrace.replay len start ops = some finalWord →
    Function.Injective ticket → (∀ e, before e e = false) →
    (∀ a b c, before a b = true → before b c = true → before a c = true) →
    (∀ i j, before (site (ticket i)) (site (ticket j)) = true → i < j) →
    (∀ i, exhausted (ticket i) = false) →
    (∀ e, exhausted e = true → ExhaustedRead start len ops (site ∘ ticket) before (site e)) →
    let hb := fun u v => before (site u) (site v) = true
    let xs := withExhausted hb ticket exhausted
    let projected := observations hb ticket exhausted
    xs.Perm (List.finRange n) ∧ (∀ i j, hb i j → xs.idxOf i < xs.idxOf j) ∧
      xs.filter (fun i => projected.contains i) = projected ∧
      ∀ (pick : Fin n → Bool) (ys : List (Fin n)), (∀ e, pick e = true ↔ e ∈ ys) →
        ys.Pairwise hb → xs.filter pick = ys

theorem projectsReads : ProjectionContract := by
  intro w n len start word ops site before ticket empty run unique noLoop trans writeWrite separate reads
  apply withExhausted_spec (fun a b => before (site a) (site b) = true) ticket empty unique
    (fun i => by simp [noLoop]) (fun a b => trans _ _ _ a b) writeWrite separate
  intro e ended i
  let sample := reads e ended
  exact exhaustedReturn_not_before (hb := fun a b => before a b = true) ops (site ∘ ticket)
    run sample.observed sample.declined (fun a b => trans _ _ _ a b) sample.returned sample.readWrite i

namespace Regression

-- The first relaxed load and one spurious failure read rank zero.
-- The second failed CAS returns the exhausted word from modification two.
private def claims : List (Write 32) := [⟨0,17,0,1⟩,⟨1,29,1,2⟩]
example : MoveList.ConcurrentTrace.lastSource 0 [0,2] = claims.length :=
  MoveList.ConcurrentTrace.declined_source (len := 2) (start := 0) (last := 2) (seen := 2) claims
    (by decide) (by decide) (by decide)

/-- Two workers access two distinct iterators in opposite order. Each
    iterator's tickets satisfy local coherence even though the union cycles. -/
def workerHB (u v : Fin 4) := (u = 0 ∧ v = 1) ∨ (u = 2 ∧ v = 3)
def x (i : Fin 2) : Fin 4 := if i = 0 then 3 else 0
def y (i : Fin 2) : Fin 4 := if i = 0 then 1 else 2
def ordered (ticket : Fin 2 → Fin 4) (rank : Fin 4 → Nat) :=
  ∀ i j, i < j → rank (ticket i) < rank (ticket j)
def lawful (ticket : Fin 2 → Fin 4) :=
  (∀ i j, ticket i = ticket j → i = j) ∧ (∀ i j, workerHB (ticket i) (ticket j) → i < j)

def LocalTrace (xs : List (Nat × Nat)) :=
  ∃ s, MoveList.ConcurrentTrace.Execution 3 (1 : BitVec 64) s ∧
    s.writes.map (fun t => (t.worker,t.opening)) = xs

private theorem localTrace (w v first second : Nat) (diff : w ≠ v) :
    LocalTrace [(w,first),(v,second)] := by
  have start := MoveList.ConcurrentTrace.Execution.init (len := 3) (start := (1 : BitVec 64))
  have load₁ := start.load (worker := w) rfl (source := 0) rfl first
  have claim₁ := load₁.success (worker := w) (opening := first) (by simp [MoveList.ConcurrentTrace.replace]) (wanted := 2) rfl
  have load₂ := claim₁.load (worker := v) (by simp [MoveList.ConcurrentTrace.replace,Ne.symm diff])
    (source := 1) rfl second
  have claim₂ := load₂.success (worker := v) (opening := second) (by simp [MoveList.ConcurrentTrace.replace]) (wanted := 3) rfl
  exact ⟨_,claim₂,rfl⟩

def TwoChains : Prop :=
  (∀ u, ¬workerHB u u) ∧ (∀ u v w, workerHB u v → workerHB v w → workerHB u w) ∧
  lawful x ∧ lawful y ∧ (∀ i j, x i ≠ y j) ∧
  LocalTrace [(1,(x 0).val),(0,(x 1).val)] ∧ LocalTrace [(0,(y 0).val),(1,(y 1).val)] ∧
  ¬∃ rank, (∀ u v, workerHB u v → rank u < rank v) ∧ ordered x rank ∧ ordered y rank

theorem twoChains : TwoChains := by
  refine ⟨by unfold workerHB; decide,by unfold workerHB; decide,by unfold lawful x workerHB; decide,
    by unfold lawful y workerHB; decide,by unfold x y; decide,
    localTrace _ _ _ _ (by decide),localTrace _ _ _ _ (by decide),?_⟩
  rintro ⟨rank,hb,xOrder,yOrder⟩
  have t₁ := hb 0 1 (Or.inl ⟨rfl,rfl⟩)
  have t₂ := hb 2 3 (Or.inr ⟨rfl,rfl⟩)
  have xx : rank 3 < rank 0 := xOrder 0 1 (by decide)
  have yy : rank 1 < rank 2 := yOrder 0 1 (by decide)
  omega

-- Events 3 and 0 are successful tickets. The two exhausted readers are
-- HB-ordered 4 before 1, independently of the neutral event at 2.
private def endHB (u v : Fin 5) := u = 4 ∧ v = 1
private instance : DecidableRel endHB := fun u v => inferInstanceAs (Decidable (u = 4 ∧ v = 1))
private def modified (i : Fin 2) : Fin 5 := if i = 0 then 3 else 0
private def exhausted (e : Fin 5) := e == 1 || e == 4
#guard ((withExhausted endHB modified exhausted).filter (· != 2)).map Fin.val == [3,0,4,1]

end Regression

/-- A worker which leaves split X before being booked on Y, and another which
    leaves Y before being booked on X, cannot realize the two-chain cycle.
    Each split mutex orders its detach and the other worker's booking. -/
theorem detachBook {hb : α → α → Prop} (noLoop : ∀ u, ¬ hb u u)
    (trans : ∀ {a b c}, hb a b → hb b c → hb a c)
    (fromX : hb xHigh leaveX) (toY : hb leaveX bookY) (onY : hb bookY yLow)
    (fromY : hb yHigh leaveY) (toX : hb leaveY bookX) (onX : hb bookX xLow)
    (lockX : hb leaveX bookX ∨ hb bookX leaveX) (lockY : hb leaveY bookY ∨ hb bookY leaveY)
    (xMO : ¬hb xHigh xLow) (yMO : ¬hb yHigh yLow) : False := by
  rcases lockX with forward | backX
  · exact xMO (trans fromX (trans forward onX))
  rcases lockY with forward | backY
  · exact yMO (trans fromY (trans forward onY))
  exact noLoop bookX (trans backX (trans toY (trans backY toX)))

end NeuralReversi.PvsParallel.Interleaving.IteratorEventOrder

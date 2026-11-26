import NeuralReversi.PvsParallel.Interleaving.ReadCandidates

namespace NeuralReversi.PvsParallel.Interleaving.SerialMemory

open TranspositionTable.SeqLock (Word Location Memory)

structure Read where
  time : Nat
  location : Location
  word : Word

/-- One slot's trace on a single worker. Physical instruction numbers are in [0,limit). -/
structure Sample where
  limit : Nat
  /-- Reserve event IDs 0 through 3 for the initial words when the public replay requires timestamp order. -/
  initInPrefix : Bool := false
  ranks : Nat
  initial : Location → Word
  writes : Location → List (Nat × Word)
  reads : List Read
  release : Nat → Bool
  acquire : Nat → Bool
  releaseFence : Nat → Bool
  acquireFence : Nat → Bool

/-- Each location has its own initialization event, separate from all instructions
    in the public tape and from the sentinel for an absent write. A prefixed trace
    reserves four inert positions before the first physical instruction. -/
def Sample.birth (s : Sample) (l : Location) :=
  (if s.initInPrefix then 0 else s.limit) + match l with
    | .sequence => 0 | .payload .player => 1 | .payload .opponent => 2 | .payload .metadata => 3

def Sample.key (s : Sample) (t : Nat) :=
  if t < s.limit then t+1 else if !s.initInPrefix && decide (t < s.limit+4) then 0 else s.limit+1

theorem Sample.ordered (s : Sample) (front : s.initInPrefix = true)
    (lt : s.key i < s.key j) (bound : j < s.limit) : i < j := by
  simp [key,front,bound] at lt
  split at lt <;> omega

theorem Sample.real (s : Sample) (inside : t < s.limit) : s.key t < s.limit+1 := by
  simp [key,inside]

def Sample.atRank (s : Sample) (l : Location) (i : Nat) : Nat × Word :=
  ((s.birth l,s.initial l) :: s.writes l)[i]?.getD (s.limit+4,0)

def Sample.rank (s : Sample) (l : Location) (r : Nat) :=
  ((s.writes l).filter (fun w => decide (w.1 < r))).length

def Sample.edges (s : Sample) :=
  s.reads.map (fun r => ((s.atRank r.location (s.rank r.location r.time)).1,r.time))

def Sample.value (s : Sample) (t : Nat) :=
  (s.reads.find? (fun r => r.time == t)).map Read.word |>.getD 0

/-- These finite comparisons are evaluated over the native read instructions and
    the actual modification-order ranks of each location. -/
structure Checked (s : Sample) : Prop where
  bounded : ∀ l, (s.writes l).length < s.ranks
  times : ∀ e ∈ s.edges, s.key e.1 < s.key e.2 ∧ e.2 < s.limit
  values : ∀ l (i : Fin s.ranks), ∀ e ∈ s.edges, (s.atRank l i).1 = e.1 → s.value e.2 = (s.atRank l i).2
  ww : ∀ l (i j : Fin s.ranks), s.key (s.atRank l i).1 < s.key (s.atRank l j).1 → (s.atRank l j).1 < s.limit → i < j
  wr : ∀ l (i j : Fin s.ranks), ∀ e ∈ s.edges,
    s.key (s.atRank l i).1 < s.key e.2 → (s.atRank l j).1 = e.1 → i ≤ j
  rr : ∀ l (i j : Fin s.ranks), ∀ e ∈ s.edges, ∀ f ∈ s.edges,
    s.key e.2 < s.key f.2 → (s.atRank l i).1 = e.1 → (s.atRank l j).1 = f.1 → i ≤ j

theorem Sample.present (s : Sample) (h : s.key (s.atRank l i).1 < s.limit+1) : i ≤ (s.writes l).length := by
  by_cases ok : i ≤ (s.writes l).length
  · exact ok
  · have absent : ((s.birth l,s.initial l) :: s.writes l)[i]? = none := List.getElem?_eq_none (by simp; omega)
    have big : ¬s.limit+4 < s.limit := by omega
    simp [atRank,absent,key,big] at h

def Sample.memory (s : Sample) (c : Checked s) : Memory Nat where
  sb := fun i j => i < j ∧ j < s.limit
  hb := fun i j => s.key i < s.key j ∧ j < s.limit
  rf := fun w r => (w,r) ∈ s.edges
  write := fun l i => (s.atRank l i).1
  writeValue := fun l i => (s.atRank l i).2
  readValue := s.value
  rf_value := by
    intro l i r h
    let support := c.times _ h
    have ok := Nat.lt_of_le_of_lt (s.present (Nat.lt_trans support.1 (s.real support.2))) (c.bounded l)
    exact c.values l ⟨i,ok⟩ _ h rfl
  release := fun i => s.release i = true
  acquire := fun i => s.acquire i = true
  releaseFence := fun i => s.releaseFence i = true
  acquireFence := fun i => s.acquireFence i = true
  sb_hb := by
    intro a b h
    exact ⟨by simpa [key,h.2,Nat.lt_trans h.1 h.2] using h.1,h.2⟩
  sb_trans := fun a b => ⟨Nat.lt_trans a.1 b.1,b.2⟩
  hb_trans := fun a b => ⟨Nat.lt_trans a.1 b.1,b.2⟩
  hb_irrefl := fun i h => Nat.lt_irrefl (s.key i) h.1
  release_acquire := fun _ _ edge => c.times _ edge
  fence_fence := by
    intro a w r b _ _ old edge later
    have order := (c.times _ edge).1
    rw [key,ite_eq_left old.2,key,ite_eq_left (Nat.lt_trans later.1 later.2)] at order
    refine ⟨?_,later.2⟩
    simp only [key,ite_eq_left (Nat.lt_trans old.1 old.2),ite_eq_left later.2]
    omega
  write_write := by
    intro l i j hb
    have iBound := Nat.lt_of_le_of_lt (s.present (Nat.lt_trans hb.1 (s.real hb.2))) (c.bounded l)
    have jBound := Nat.lt_of_le_of_lt (s.present (s.real hb.2)) (c.bounded l)
    exact c.ww l ⟨i,iBound⟩ ⟨j,jBound⟩ hb.1 hb.2
  write_read := by
    intro l i j r hb rf
    have active := c.times _ rf
    have iBound := Nat.lt_of_le_of_lt (s.present (Nat.lt_trans hb.1 (s.real hb.2))) (c.bounded l)
    have jBound := Nat.lt_of_le_of_lt (s.present (Nat.lt_trans active.1 (s.real active.2))) (c.bounded l)
    exact c.wr l ⟨i,iBound⟩ ⟨j,jBound⟩ _ rf hb.1 rfl
  read_read := by
    intro l i j p r hb old new_
    have one := c.times _ old
    have two := c.times _ new_
    have iBound := Nat.lt_of_le_of_lt (s.present (Nat.lt_trans one.1 (s.real one.2))) (c.bounded l)
    have jBound := Nat.lt_of_le_of_lt (s.present (Nat.lt_trans two.1 (s.real two.2))) (c.bounded l)
    exact c.rr l ⟨i,iBound⟩ ⟨j,jBound⟩ _ old _ new_ hb.1 rfl rfl

end NeuralReversi.PvsParallel.Interleaving.SerialMemory

module
public import NeuralReversi.PvsParallel.NativeTableWrites.Memory

@[expose] public section

namespace NeuralReversi.PvsParallel.Interleaving.ModificationIndex

open TranspositionTable
open SeqLock (Word Location)
open FrameTableReads (Address)
open NativeTableWrites.Journal (Calls Record Site mods)

def tag : Location → Nat
  | .sequence => 0
  | .payload .player => 1
  | .payload .opponent => 2
  | .payload .metadata => 3

def locations : List Location := [.sequence,.payload .player,.payload .opponent,.payload .metadata]

private theorem locationAt (l : Location) : locations[tag l]? = some l := by
  cases l with | sequence => rfl | payload f => cases f <;> rfl

private theorem tag_lt (l : Location) : tag l < 4 := by
  cases l with | sequence => decide | payload f => cases f <;> decide

private theorem tag_unique {l k : Location} (same : tag l = tag k) : l = k := by
  exact Option.some.inj ((locationAt l).symm.trans (same ▸ locationAt k))

def stamps (c : Calls) (rs : List Record) (n : Nat) (a : Address) (loc : Location) : List Nat :=
  (n + tag loc) :: (mods c rs a loc).map (Site.event c)

/-- Events below n are emitted instructions. Each atomic location in one slot has its own
    initial token above the emitted prefix. Missing ranks share a token outside that domain. -/
def event (c : Calls) (rs : List Record) (n : Nat) (a : Address) (loc : Location) (i : Nat) : Nat :=
  ((stamps c rs n a loc)[i]?).getD (n + 4)

def word (c : Calls) (endian : Bool) (rs : List Record) (initial : Address → Location → Word)
    (a : Address) (loc : Location) (i : Nat) : Word :=
  ((initial a loc :: (mods c rs a loc).map (fun s => s.word c endian))[i]?).getD 0

structure Timed (c : Calls) (rs : List Record) (n : Nat) : Prop where
  bounded : ∀ s, .writer s ∈ rs → s.event c < n
  unique : ∀ s t, .writer s ∈ rs → .writer t ∈ rs → s.event c = t.event c → s = t
  distinct : ∀ a loc, ((mods c rs a loc).map (Site.event c)).Nodup

private theorem atSite (hit : (mods c rs a loc)[k]? = some s) :
    event c rs n a loc (k+1) = s.event c := by simp [event,stamps,hit]

private theorem site_lt {k : Nat} {s : Site} (timed : Timed c rs n) (hit : (mods c rs a loc)[k]? = some s) :
    s.event c < n := timed.bounded _ (NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hit)).1

theorem event_lt (timed : Timed c rs n) (inside : i ≤ (mods c rs a loc).length) :
    event c rs n a loc i < n + 4 := by
  cases i with
  | zero => change n + tag loc < n+4; have := tag_lt loc; omega
  | succ k =>
    have hit := List.getElem?_eq_getElem (show k < _ from inside)
    rw [atSite hit]
    have := site_lt timed hit
    omega

theorem event_outside (outside : (mods c rs a loc).length < i) : event c rs n a loc i = n+4 := by
  unfold event
  rw [List.getElem?_eq_none (by simp [stamps]; omega)]
  rfl

/-- Only the first rank is assumed to exist. In particular the sentinel for an arbitrary
    out-of-range second rank cannot masquerade as an initial write or a modifying opcode. -/
theorem event_injective (timed : Timed c rs n) (inside : i ≤ (mods c rs a loc).length)
    (eq : event c rs n a loc i = event c rs n a other j) : loc = other ∧ i = j := by
  have there : j ≤ (mods c rs a other).length := by
    apply Nat.le_of_not_gt
    intro outside
    have := event_lt timed inside
    rw [eq,event_outside outside] at this
    omega
  cases i with
  | zero =>
    cases j with
    | zero => exact ⟨tag_unique (Nat.add_left_cancel eq),rfl⟩
    | succ j =>
      have hit := List.getElem?_eq_getElem (show j < _ from there)
      rw [atSite hit] at eq
      change n + tag loc = _ at eq
      have := site_lt timed hit
      omega
  | succ i =>
    have hit := List.getElem?_eq_getElem (show i < _ from inside)
    rw [atSite hit] at eq
    cases j with
    | zero =>
      change _ = n + tag other at eq
      have := site_lt timed hit
      omega
    | succ j =>
      have hit' := List.getElem?_eq_getElem (show j < _ from there)
      rw [atSite hit'] at eq
      obtain ⟨present,_,_,atLoc⟩ := NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hit)
      obtain ⟨present',_,_,atOther⟩ := NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hit')
      have same := timed.unique _ _ present present' eq
      have l := Option.some.inj (atLoc.symm.trans ((congrArg (fun s : Site => s.pc.location) same).trans atOther))
      subst other
      have ranked := (timed.distinct a loc).eq_of_getElem?_eq (i := i) (j := j)
        (by simp; omega) (by simp [List.getElem?_map,hit,hit',eq])
      exact ⟨rfl,congrArg Nat.succ ranked⟩

def catalog (c : Calls) (rs : List Record) (n : Nat) (a : Address) :=
  locations.flatMap fun loc => (stamps c rs n a loc).zipIdx.map fun (ev,i) => (ev,(loc,i))

private theorem member : (w,(loc,i)) ∈ catalog c rs n a ↔ (stamps c rs n a loc)[i]? = some w := by
  constructor
  · intro mem
    obtain ⟨l,_,row⟩ := List.mem_flatMap.mp mem
    obtain ⟨⟨ev,j⟩,indexed,eq⟩ := List.mem_map.mp row
    cases eq
    exact List.mem_zipIdx_iff_getElem?.mp indexed
  · intro hit
    exact List.mem_flatMap.mpr ⟨loc,List.mem_of_getElem? (locationAt loc),List.mem_map.mpr
      ⟨(w,i),List.mem_zipIdx_iff_getElem?.mpr hit,rfl⟩⟩

/-- Look up the actual physical source event without examining its written word. -/
def decode (c : Calls) (rs : List Record) (n : Nat) (a : Address) (w : Nat) : Option (Location × Nat) :=
  ((catalog c rs n a).find? (fun row => row.1 == w)).map Prod.snd

private theorem bound_word (hit : (stamps c rs n a loc)[i]? = some w) :
    i ≤ (mods c rs a loc).length ∧ event c rs n a loc i = w :=
  ⟨by have b := (List.getElem?_eq_some_iff.mp hit).1; simp [stamps] at b; omega,by simp [event,hit]⟩

theorem decode_eq_some_iff (timed : Timed c rs n) :
    decode c rs n a w = some (loc,i) ↔ i ≤ (mods c rs a loc).length ∧ event c rs n a loc i = w := by
  unfold decode
  constructor
  · intro found
    obtain ⟨⟨ev,l,j⟩,hit,eq⟩ := Option.map_eq_some_iff.mp found
    cases eq
    have matched := List.find?_some hit
    have same : ev = w := beq_iff_eq.mp (show (ev == w) = true from matched)
    exact same ▸ bound_word (member.mp (List.mem_of_find?_eq_some hit))
  · rintro ⟨inside,rfl⟩
    have index : i < (stamps c rs n a loc).length := by simp [stamps]; omega
    have target : (stamps c rs n a loc)[i]? = some (event c rs n a loc i) := by
      simp [event,List.getElem?_eq_getElem index]
    cases result : (catalog c rs n a).find? (fun row => row.1 == event c rs n a loc i) with
    | none =>
      have cannot := List.find?_eq_none.mp result _ (member.mpr target)
      simp at cannot
    | some row =>
      obtain ⟨ev,l,j⟩ := row
      have matched := List.find?_some result
      have same : ev = event c rs n a loc i := beq_iff_eq.mp (show (ev == _) = true from matched)
      obtain ⟨_,recorded⟩ := bound_word (member.mp (List.mem_of_find?_eq_some result))
      obtain ⟨rfl,rfl⟩ := event_injective timed inside (same.symm.trans recorded.symm)
      rfl

theorem decode_initial (timed : Timed c rs n) (a : Address) (loc : Location) :
    decode c rs n a (n + tag loc) = some (loc,0) :=
  (decode_eq_some_iff timed).mpr ⟨Nat.zero_le _,rfl⟩

theorem decode_modification (timed : Timed c rs n) (hit : (mods c rs a loc)[i]? = some s) :
    decode c rs n a (s.event c) = some (loc,i+1) :=
  (decode_eq_some_iff timed).mpr ⟨(List.getElem?_eq_some_iff.mp hit).1,atSite hit⟩

/-- The source bound needed by Journal.Atomics.source follows from raw graph edges whose write
    endpoints are covered by the physical catalog, even when the queried rank is arbitrary. -/
theorem source (timed : Timed c rs n) {rf : Nat → Nat → Prop}
    (covered : ∀ w r, rf w r → (decode c rs n a w).isSome = true)
    (edge : rf (event c rs n a loc i) r) : i ≤ (mods c rs a loc).length := by
  obtain ⟨⟨other,j⟩,found⟩ := Option.isSome_iff_exists.mp (covered _ _ edge)
  obtain ⟨valid,eq⟩ := (decode_eq_some_iff timed).mp found
  obtain ⟨rfl,rfl⟩ := event_injective timed valid eq
  exact valid

theorem modified (hit : (mods c rs a loc)[i]? = some s) :
    event c rs n a loc (i+1) = s.event c ∧
      NativeTableWrites.Journal.writes (c s.invocation).success ((c s.invocation).program endian).1[s.pc.index]? =
        some (loc,word c endian rs initial a loc (i+1)) := by
  obtain ⟨_,_,success,atLoc⟩ := NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hit)
  refine ⟨atSite hit,?_⟩
  rw [s.modification,success,ite_eq_left rfl,atLoc]
  simp [word,hit]

namespace Regression

private def addr : Address := ⟨9,2⟩
private def copyAddr : Address := ⟨10,2⟩

private def clears : Calls := fun k => ⟨if k = 3 then copyAddr else addr,
  if k = 1 then ⟨2,by decide⟩ else ⟨0,by decide⟩,.clear,decide (k ≠ 2),fun pc => 10*k + pc⟩

private def writer (k : Nat) : List Record :=
  [.cas,.fence,.field .player,.field .opponent,.field .metadata,.finish].map fun pc => .writer ⟨k,pc⟩

private def rs := writer 0 ++ writer 1 ++ [.writer ⟨2,.cas⟩] ++ writer 3
private def lookup := decode clears rs 40 addr

-- The same payload value at distinct successful instructions has distinct modification ranks.
example : word clears false rs (fun _ _ => 7) addr (.payload .player) 1 = 0 ∧
    word clears false rs (fun _ _ => 7) addr (.payload .player) 2 = 0 := ⟨rfl,rfl⟩
example : lookup 2 = some (.payload .player,1) ∧ lookup 12 = some (.payload .player,2) := ⟨rfl,rfl⟩
example : [40,41,42,43].map lookup =
    [some (.sequence,0),some (.payload .player,0),some (.payload .opponent,0),some (.payload .metadata,0)] := rfl

-- A fence, failed CAS, foreign allocation, and the missing-rank sentinel have no entry here.
example : [1,20,32,44].map lookup = [none,none,none,none] := rfl
example : decode clears rs 40 copyAddr 32 = some (.payload .player,1) := rfl
example : event clears rs 40 addr .sequence 99 = 44 := rfl

end Regression

end NeuralReversi.PvsParallel.Interleaving.ModificationIndex

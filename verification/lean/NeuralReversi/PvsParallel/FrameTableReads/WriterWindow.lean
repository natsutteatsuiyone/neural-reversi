module
public import NeuralReversi.PvsParallel.FrameTableReads.Execution
public import NeuralReversi.PvsParallel.NativeTableWrites.Memory

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeTableWrites.Journal

open TranspositionTable
open FrameTableReads.Operations.Execution (Atomic Boundaries)

def isClaim (s : Site) : Bool := decide (s.pc = .cas)

/-- A writer can already hold the slot when the interval ends. Its release
    is the only sequence modification that can precede the next claim. -/
theorem Execution.pending_at_most_one {journal : List Record} (x : Execution c endian journal)
    (a : FrameTableReads.Address) (init : Even (x.memory a)) (pre post : List Record)
    (partition : journal = pre ++ post) : (mods c post a .sequence).findIdx isClaim ≤ 1 := by
  let before := mods c pre a .sequence
  let tail := mods c post a .sequence
  have split : mods c journal a .sequence = before ++ tail := by simp [partition,mods,before,tail]
  change tail.findIdx isClaim ≤ 1
  apply Classical.byContradiction
  intro more
  let k := before.length % 2
  have skipped : k < tail.findIdx isClaim := by dsimp [k]; omega
  have inside := Nat.lt_of_lt_of_le skipped List.findIdx_le_length
  let s := tail[k]
  have notClaim : s.pc ≠ .cas := of_decide_eq_false (List.not_of_lt_findIdx skipped : isClaim s = false)
  let i := before.length+k
  have atSite : (mods c journal a .sequence)[i]? = some s := by
    rw [split,List.getElem?_append_right (by dsimp [i]; omega)]
    simp [i,s]
  let seq := x.toSequence a init
  have valid : i < seq.history.length := by
    change i < (sequence c endian journal a).length
    simpa [sequence] using (List.getElem?_eq_some_iff.mp atSite).1
  have even : i % 2 = 0 := by dsimp [i,k]; omega
  obtain ⟨call,claim,_⟩ := seq.order.callAt (i/2+1) (by omega) (by omega)
  rw [show 2*(i/2+1)-2 = i by omega] at claim
  change (sequence c endian journal a)[i]? = some (.claim call) at claim
  simp [sequence,List.getElem?_map,atSite,Site.action,notClaim] at claim

/-- The last successful sequence modification in the prefix and the first new claim in
    the suffix are concrete writer instructions. A publication that completes an
    already pending writer belongs to the upper bound. Two ordering edges suffice for
    every register of the caller; either side of the journal may be empty. -/
theorem Execution.readerBoundaries {journal : List Record} (x : Execution c endian journal) (a : FrameTableReads.Address)
    (init : Even (x.memory a)) (registers : List Atomic) (consumer : Nat) (pre active post : List Record)
    (partition : journal = pre ++ active ++ post) :
    let left := mods c pre a .sequence
    let mid := mods c active a .sequence
    let right := mods c post a .sequence
    (∀ s, left.getLast? = some s → ∀ first, registers[0]? = some first → (x.memory a).hb (s.event c) first.id) →
    (∀ s, right.find? isClaim = some s → (x.memory a).hb consumer (s.event c)) →
    Boundaries (x.toSequence a init) consumer registers left.length (left.length + mid.length + right.findIdx isClaim) := by
  intro left mid right earlier later
  have split : mods c journal a .sequence = left ++ mid ++ right := by
    simp [partition,mods,left,mid,right]
  constructor
  · intro nonempty first found
    let i := left.length-1
    have inside : i < left.length := by dsimp [i]; omega
    have atSite := List.getElem?_eq_getElem inside
    have prior := earlier left[i] (by simpa only [List.getLast?_eq_getElem?] using atSite) first found
    have hit : (mods c journal a .sequence)[i]? = some left[i] := by
      rw [split,List.getElem?_append_left (by simp; omega),List.getElem?_append_left inside]
      exact atSite
    have id := (x.modified _ _ _ _ hit).1
    rw [show left.length = i+1 by dsimp [i]; omega, id]
    exact prior
  · intro inside
    let k := right.findIdx isClaim
    change left.length + mid.length + k < (sequence c endian journal a).length at inside
    simp only [sequence,List.length_map,split,List.length_append] at inside
    have valid : k < right.length := by omega
    let s := right[k]
    have atSite : right[k]? = some s := List.getElem?_eq_getElem valid
    have tag : s.pc = .cas := of_decide_eq_true (List.findIdx_of_getElem?_eq_some atSite : isClaim s)
    have ahead := later s (by rwa [List.find?_eq_getElem?_findIdx])
    have hit : (mods c journal a .sequence)[left.length+mid.length+k]? = some s := by
      rw [split,List.getElem?_append_right (by simp)]
      simpa using atSite
    refine ⟨(c s.invocation).call endian,?_,?_⟩
    · change (sequence c endian journal a)[left.length+mid.length+k]? = _
      simp [sequence,List.getElem?_map,hit,Site.action,tag]
    · simpa [Site.event,tag,Pc.index,Attempt.call] using ahead

end NeuralReversi.PvsParallel.NativeTableWrites.Journal

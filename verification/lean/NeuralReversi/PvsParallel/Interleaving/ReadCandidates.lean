import NeuralReversi.PvsParallel.Interleaving.Joint

namespace NeuralReversi.PvsParallel.Interleaving.ReadCandidates

open TranspositionTable
open SeqLock (Word Location)
open FrameTableReads (Address)
open NativeTableWrites.Journal (Calls Record Site mods)

private def lower (c : Calls) (sites : List Site) (before : Nat → Bool) : Nat :=
  ((sites.zipIdx.filter fun row => before (row.1.event c)).map (fun row => row.2 + 1)).max?.getD 0

/-- Zero denotes the initial word. Positive indices enumerate successful modifications to this
    allocation, slot and location in modification order. Only certified HB edges can exclude an
    older equal word. Event numbers on different workers need not follow modification order. -/
def sources (c : Calls) (endian : Bool) (rs : List Record) (a : Address) (loc : Location)
    (initial seen : Word) (before : Nat → Bool) : List Nat :=
  let sites := mods c rs a loc
  let floor := lower c sites before
  let words := initial :: sites.map (fun site => site.word c endian)
  (words.zipIdx.filter (fun row => row.1 == seen && floor ≤ row.2)).map Prod.snd

variable {x : NativeTableWrites.Journal.Execution c endian rs}

private theorem modification (x : NativeTableWrites.Journal.Execution c endian rs)
    (hit : (mods c rs a loc)[j]? = some s) :
    (x.memory a).write loc (j+1) = s.event c ∧ (x.memory a).writeValue loc (j+1) = s.word c endian := by
  obtain ⟨event,word⟩ := x.modified _ _ _ _ hit
  obtain ⟨_,_,success,location⟩ := NativeTableWrites.Journal.mem_mods.mp (List.mem_of_getElem? hit)
  rw [s.modification,success,ite_eq_left rfl,location] at word
  exact ⟨event,(congrArg Prod.snd (Option.some.inj word)).symm⟩

/-- Every actual source is retained, also for a stale relaxed load or a wrapped sequence.
    Multiple surviving candidates need not all extend to a full execution. -/
theorem complete (x : NativeTableWrites.Journal.Execution c endian rs) (before : Nat → Bool)
    (sound : ∀ w, before w = true → (x.memory a).hb w r)
    (rf : (x.memory a).rf ((x.memory a).write loc i) r) :
    i ∈ sources c endian rs a loc ((x.memory a).writeValue loc 0) ((x.memory a).readValue r) before := by
  let sites := mods c rs a loc
  have value := (x.memory a).rf_value rf
  have word : (((x.memory a).writeValue loc 0) :: sites.map (fun s => s.word c endian))[i]? =
      some ((x.memory a).readValue r) := by
    cases i with
    | zero => simp [value]
    | succ j =>
      have inside : j < sites.length := x.source _ _ _ _ rf
      have hit := List.getElem?_eq_getElem inside
      rw [List.getElem?_cons_succ,List.getElem?_map,hit,value,(modification x hit).2]
      rfl
  have floor : lower c sites before ≤ i := by
    let bounds := (sites.zipIdx.filter fun row => before (row.1.event c)).map (fun row => row.2 + 1)
    change bounds.max?.getD 0 ≤ i
    cases h : bounds.max? with
    | none => exact Nat.zero_le _
    | some n =>
      change n ≤ i
      obtain ⟨row,mem,rfl⟩ := List.mem_map.mp (List.max?_mem h)
      obtain ⟨atIndex,known⟩ := List.mem_filter.mp mem
      have atWrite := (modification x (List.mem_zipIdx_iff_getElem?.mp atIndex)).1
      apply (x.memory a).write_read _ rf
      rw [atWrite]
      exact sound _ known
  exact List.mem_map.mpr ⟨(_,i),List.mem_filter.mpr
    ⟨List.mem_zipIdx_iff_getElem?.mpr word,by simpa using floor⟩,rfl⟩

theorem identified (x : NativeTableWrites.Journal.Execution c endian rs) (before : Nat → Bool)
    (sound : ∀ w, before w = true → (x.memory a).hb w r)
    (rf : (x.memory a).rf ((x.memory a).write loc i) r)
    (one : sources c endian rs a loc ((x.memory a).writeValue loc 0) ((x.memory a).readValue r) before = [j]) :
    i = j := by simpa [one] using complete x before sound rf

def sameWorkerBefore (es : Joint.Tape Op) (r w : Nat) : Bool :=
  decide (w < r ∧ r < es.length ∧ Joint.worker es w = Joint.worker es r)

/-- The executable calculation ignores source labels on the tape. The theorem compares its
    answer to the physical source of an issued load in any compatible execution. -/
def onTape (c : Calls) (endian : Bool) (rs : List Record) (es : Joint.Tape Op)
    (r : Nat) (a : Address) (loc : Location) (initial : Word) : List Nat :=
  sources c endian rs a loc initial (Joint.word es r) (sameWorkerBefore es r)

theorem issued (m : Joint.Atomics x es) (load : Joint.opcode es r = some (.load a loc mode)) :
    Joint.origin es r ∈ onTape c endian rs es r a loc ((x.memory a).writeValue loc 0) := by
  obtain ⟨rf,word⟩ := m.reads _ _ _ _ load
  unfold onTape
  rw [← word]
  apply complete x _ _ rf
  intro w known
  obtain ⟨earlier,inside,worker⟩ := of_decide_eq_true known
  exact (x.memory a).sb_hb (m.sb _ _ _ earlier inside worker)

/-- Coherent successive reads of the same location cannot go backwards in modification order.
    The earlier load may itself have several candidates. -/
def afterRead (previous current : List Nat) := current.filter fun j => previous.any fun i => i ≤ j

theorem afterRead_complete {mem : SeqLock.Memory Nat} (hb : mem.hb p r)
    (first : mem.rf (mem.write loc i) p) (last : mem.rf (mem.write loc j) r)
    (earlier : i ∈ previous) (later : j ∈ current) : j ∈ afterRead previous current :=
  List.mem_filter.mpr ⟨later,List.any_eq_true.mpr ⟨i,earlier,decide_eq_true (mem.read_read hb first last)⟩⟩

namespace Regression

private def addr : Address := ⟨9,2⟩
private def clearAt (a : Address) (expected : {w : Word // w &&& 1 = 0}) (k : Nat) :
    NativeTableWrites.Journal.Attempt := ⟨a,expected,.clear,true,fun pc => 10*k + pc⟩
private def clear : Calls := fun k => clearAt addr (if k = 0 then ⟨0,by decide⟩ else ⟨2,by decide⟩) k
private def writer (k : Nat) : List Record :=
  [.cas,.fence,.field .player,.field .opponent,.field .metadata,.finish].map fun pc => .writer ⟨k,pc⟩
private def payload (before : Nat → Bool) :=
  sources clear false (writer 0 ++ writer 1) addr (.payload .player) 0 0 before

-- Initial state and both clears carry the identical player word.
example : payload (fun _ => false) = [0,1,2] := by rfl
example : payload (· == 2) = [1,2] := by rfl
example : payload (· == 12) = [2] := by rfl
example : afterRead [1,2] [0,1,2] = [1,2] := by rfl
example : sources clear false (writer 0 ++ writer 1) addr .sequence 0 3 (fun _ => false) = [3] := by rfl

-- Publication from the largest even word crosses the machine-word boundary.
private def wrapping : Calls := fun k =>
  clearAt addr (if k = 0 then ⟨0xfffffffffffffffe,by decide⟩ else ⟨0,by decide⟩) k
example : sources wrapping false (writer 0 ++ writer 1) addr .sequence
    0xfffffffffffffffe 0 (fun _ => false) = [2] := by rfl

-- Two per-location memory witnesses with identical words and HB. The only variation
-- is which of the equal payload writes the relaxed load observes.
private inductive Action where
  | write (loc : Location) (i : Nat)
  | load

private def edge (i : Nat) (a b : Action) :=
  a = .write (.payload .player) i ∧ b = .load

private def relaxed (first picked : Nat) (ok : first ≤ picked) : SeqLock.Memory Action where
  sb := edge first
  hb := edge first
  rf := edge picked
  write := Action.write
  writeValue := fun _ _ => 0
  readValue := fun _ => 0
  rf_value := fun _ => rfl
  release := fun _ => False
  acquire := fun _ => False
  releaseFence := fun _ => False
  acquireFence := fun _ => False
  sb_hb := id
  sb_trans := by rintro _ _ _ ⟨_,rfl⟩ ⟨h,_⟩; contradiction
  hb_trans := by rintro _ _ _ ⟨_,rfl⟩ ⟨h,_⟩; contradiction
  hb_irrefl := by rintro _ ⟨rfl,h⟩; contradiction
  release_acquire := fun h => h.elim
  fence_fence := fun h => h.elim
  write_write := by rintro _ _ _ ⟨_,h⟩; contradiction
  write_read := by
    rintro _ _ _ _ ⟨e,_⟩ ⟨f,_⟩
    cases e
    cases f
    exact ok
  read_read := by rintro _ _ _ _ _ ⟨e,_⟩ ⟨_,f⟩ _; rw [f] at e; contradiction

example :
    let old := relaxed 1 1 (by decide)
    let newer := relaxed 1 2 (by decide)
    old.hb = newer.hb ∧ old.readValue .load = newer.readValue .load ∧
      old.rf (old.write (.payload .player) 1) .load ∧ newer.rf (newer.write (.payload .player) 2) .load :=
  ⟨rfl,rfl,⟨rfl,rfl⟩,⟨rfl,rfl⟩⟩

end Regression
end NeuralReversi.PvsParallel.Interleaving.ReadCandidates

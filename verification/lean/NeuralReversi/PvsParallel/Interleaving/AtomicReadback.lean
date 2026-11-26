import NeuralReversi.PvsParallel.Interleaving.ReadCandidates

namespace NeuralReversi.PvsParallel.Interleaving.AtomicReadback

open TranspositionTable
open SeqLock (Word Location)
open FrameTableReads (Address)
open NativeTableWrites.Journal (Calls Record Site mods)

/-- A read that happens before a later modification takes its value strictly
    before that modification in the order of the same atomic object (C++20
    [intro.races]/17). Only actual recorded modifications are queried. -/
def ReadWrite (x : NativeTableWrites.Journal.Execution c endian rs) := ∀ a loc r i j s,
  (mods c rs a loc)[j]? = some s → (x.memory a).rf ((x.memory a).write loc i) r →
  (x.memory a).hb r (s.event c) → i < j + 1

/-- Use both directions of same-worker program order. With several compatible
    results, this list is a superset of the sources allowed by the full model. -/
def candidates (c : Calls) (endian : Bool) (rs : List Record) (es : Joint.Tape Op)
    (r : Nat) (a : Address) (loc : Location) (initial : Word) : List Nat :=
  let later := (mods c rs a loc).zipIdx.filter fun (s,_) =>
    ReadCandidates.sameWorkerBefore es (s.event c) r
  (ReadCandidates.onTape c endian rs es r a loc initial).filter fun i => later.all fun (_,j) => i ≤ j

/-- A semantic execution supplies a source for each atomic load. The instrumented
    machine supplies its word, opcode and worker. No source label on the input
    tape is consulted. -/
structure Observed (x : NativeTableWrites.Journal.Execution c endian rs) (es : Joint.Tape Op) : Prop where
  reads : ∀ r a loc mode, Joint.opcode es r = some (.load a loc mode) →
    (∃ i, (x.memory a).rf ((x.memory a).write loc i) r) ∧ (x.memory a).readValue r = Joint.word es r
  acquire : ∀ r a loc, Joint.opcode es r = some (.load a loc .acquire) → (x.memory a).acquire r
  fence : ∀ r, Joint.opcode es r = some (.fence .acquire) → ∀ a, (x.memory a).acquireFence r
  sb : ∀ a i j, i < j → j < es.length → Joint.worker es i = Joint.worker es j → (x.memory a).sb i j

variable {x : NativeTableWrites.Journal.Execution c endian rs} {es : Joint.Tape Op}

private theorem sameHB (obs : Observed x es) (a) (edge : ReadCandidates.sameWorkerBefore es r w = true) :
    (x.memory a).hb w r := by
  obtain ⟨lt,inside,worker⟩ := of_decide_eq_true edge
  exact (x.memory a).sb_hb (obs.sb _ _ _ lt inside worker)

theorem keeps (obs : Observed x es) (coherent : ReadWrite x)
    (load : Joint.opcode es r = some (.load a loc mode))
    (rf : (x.memory a).rf ((x.memory a).write loc i) r) :
    i ∈ candidates c endian rs es r a loc ((x.memory a).writeValue loc 0) := by
  refine List.mem_filter.mpr ⟨?_,List.all_eq_true.mpr ?_⟩
  · unfold ReadCandidates.onTape
    rw [← (obs.reads _ _ _ _ load).2]
    exact ReadCandidates.complete x _ (fun _ => sameHB obs a) rf
  · rintro ⟨s,j⟩ member
    obtain ⟨indexed,later⟩ := List.mem_filter.mp member
    have lt := coherent a loc r i j s (List.mem_zipIdx_iff_getElem?.mp indexed) rf (sameHB obs a later)
    exact decide_eq_true (by omega)

private def withSource (n : Nat) : Joint.Event Op → Joint.Event Op
  | .issue w code _ value => .issue w code n value
  | e => e

/-- Only the ghost source register is restored. The capture's opcodes,
    workers, return words and writer/frame events retain their positions. -/
def annotate (es : Joint.Tape Op) (source : Nat → Nat) : Joint.Tape Op :=
  es.zipIdx.map fun (e,i) => withSource (source i) e

private theorem atIndex : (annotate es source)[i]? = es[i]?.map (withSource (source i)) := by
  simp [annotate, List.getElem?_zipIdx, Option.map_map, Function.comp_def]

private theorem codeEq : Joint.opcode (annotate es source) i = Joint.opcode es i := by
  unfold Joint.opcode
  rw [atIndex]
  cases es[i]? with
  | none => rfl
  | some e => cases e <;> rfl

private theorem workerEq : Joint.worker (annotate es source) i = Joint.worker es i := by
  unfold Joint.worker
  rw [atIndex]
  cases es[i]? with
  | none => rfl
  | some e => cases e <;> rfl

private theorem loadEq (load : Joint.opcode es i = some (.load a loc mode)) :
    Joint.origin (annotate es source) i = source i ∧ Joint.word (annotate es source) i = Joint.word es i := by
  unfold Joint.opcode at load
  unfold Joint.origin Joint.word
  rw [atIndex]
  cases h : es[i]? with
  | none => simp [h] at load
  | some e =>
    cases e <;> simp_all [Joint.Event.op,withSource]

theorem origin_le (es : Joint.Tape Op) (source : Nat → Nat) (r : Nat) :
    Joint.origin (annotate es source) r ≤ source r := by
  unfold Joint.origin
  rw [atIndex]
  cases es[r]? with
  | none => exact Nat.zero_le _
  | some e => cases e <;> first | exact Nat.zero_le _ | exact Nat.le_refl _

theorem engineEvents (es : Joint.Tape Op) (source : Nat → Nat) :
    (annotate es source).map Joint.Event.engineEvent = es.map Joint.Event.engineEvent := by
  apply List.ext_getElem?
  intro i
  simp only [List.getElem?_map,atIndex,Option.map_map]
  cases es[i]? with
  | none => rfl
  | some e => cases e <;> rfl

def restores (x : NativeTableWrites.Journal.Execution c endian rs) (es : Joint.Tape Op) (source : Nat → Nat) :=
  ∀ r a loc mode, Joint.opcode es r = some (.load a loc mode) →
    (x.memory a).rf ((x.memory a).write loc (source r)) r

theorem restored (obs : Observed x es) (hits : restores x es source) :
    Joint.Atomics x (annotate es source) where
  reads r a loc mode load := by
    rw [codeEq] at load
    rw [(loadEq load).1,(loadEq load).2]
    exact ⟨hits _ _ _ _ load,(obs.reads _ _ _ _ load).2⟩
  acquire := fun _ _ _ load => obs.acquire _ _ _ (codeEq ▸ load)
  fence := fun _ op => obs.fence _ (codeEq ▸ op)
  sb a i j lt inside worker := by
    simp only [workerEq] at worker
    exact obs.sb a i j lt (by simpa [annotate] using inside) worker

/-- Existence also covers captures with ambiguous equal words. The same
    semantic execution supplies all choices. A cross-product of separately
    surviving candidates need not satisfy joint memory-order constraints. -/
def Contract := ∀ {c endian rs Op} (x : NativeTableWrites.Journal.Execution c endian rs) (es : Joint.Tape Op),
  Observed x es → ReadWrite x → ∃ source : Nat → Nat, Joint.Atomics x (annotate es source) ∧
    ∀ r a loc mode, Joint.opcode es r = some (.load a loc mode) →
      source r ∈ candidates c endian rs es r a loc ((x.memory a).writeValue loc 0)

theorem reconstruct : Contract := by
  intro c endian rs Op x es obs coherent
  have available (r : Nat) : ∃ i, ∀ a loc mode, Joint.opcode es r = some (.load a loc mode) →
      (x.memory a).rf ((x.memory a).write loc i) r := by
    cases h : Joint.opcode es r with
    | none => exact ⟨0,by simp⟩
    | some inst =>
      cases inst with
      | load a loc mode =>
        obtain ⟨i,rf⟩ := (obs.reads _ _ _ _ h).1
        exact ⟨i,by intro _ _ _ eq; cases eq; exact rf⟩
      | _ => exact ⟨0,by simp⟩
  let source : Nat → Nat := fun r => Classical.choose (available r)
  have hits : restores x es source := fun r => Classical.choose_spec (available r)
  exact ⟨source,restored obs hits,fun _ _ _ _ load => keeps obs coherent load (hits _ _ _ _ load)⟩

/-- A uniquely determined register can also be reconstructed by computation. -/
def uniqueSource (c : Calls) (endian : Bool) (rs : List Record) (es : Joint.Tape Op)
    (initial : Address → Location → Word) (r : Nat) : Option Nat :=
  match Joint.opcode es r with
  | some (.load a loc _) =>
    match candidates c endian rs es r a loc (initial a loc) with
    | [i] => some i
    | _ => none
  | _ => some 0

theorem fromSingletons (obs : Observed x es) (coherent : ReadWrite x)
    (unique : ∀ r a loc mode, Joint.opcode es r = some (.load a loc mode) →
      (uniqueSource c endian rs es (fun a l => (x.memory a).writeValue l 0) r).isSome = true) :
    Joint.Atomics x (annotate es fun r =>
      (uniqueSource c endian rs es (fun a l => (x.memory a).writeValue l 0) r).getD 0) := by
  apply restored obs
  intro r a loc mode load
  have decoded := unique _ _ _ _ load
  obtain ⟨i,rf⟩ := (obs.reads _ _ _ _ load).1
  have mem := keeps obs coherent load rf
  simp only [uniqueSource,load] at decoded ⊢
  split at decoded <;> simp_all

end NeuralReversi.PvsParallel.Interleaving.AtomicReadback

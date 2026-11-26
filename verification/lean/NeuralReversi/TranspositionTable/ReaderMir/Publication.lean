import NeuralReversi.TranspositionTable.ReaderMir.Program
import NeuralReversi.PvsParallel.Cutoffs

namespace NeuralReversi.TranspositionTable.ReaderMir

open SeqLock (Word Memory)
open PvsParallel.Cutoffs (RawSample)

private def occupiedActions : List AtomicAction :=
  [.load .sequence .acquire, .load (.payload .metadata) .relaxed,
   .load (.payload .player) .relaxed, .load (.payload .opponent) .relaxed,
   .fence .acquire, .load .sequence .relaxed]

private theorem occupiedPath (bigEndian : Bool) (w : Nat → Word) (result : SeqLock.Snapshot)
    (hit : (run bigEndian w).returned = some result)
    (occupied : Snapshot.empty bigEndian result.metadata = false) :
    (run bigEndian w).actions = occupiedActions ∧ w 0 &&& 1 = 0 ∧ w 0 = w 5 ∧
      result = ⟨w 2, w 3, w 0, w 1⟩ := by
  unfold run at hit ⊢
  split at hit
  · contradiction
  next even =>
    rw [ite_eq_right even]
    split at hit
    next empty =>
      split at hit
      · contradiction
      · cases hit
        simp [Snapshot.empty, empty] at occupied
    next full =>
      rw [ite_eq_right full]
      split at hit
      · contradiction
      next same =>
        rw [ite_eq_right same]
        exact ⟨rfl, Classical.not_not.mp even, Classical.not_not.mp same, (Option.some.inj hit).symm⟩

/-- Interpretation of the atomic actions of one call in a shared execution.
Event numbers are identities, while indices in the list are local program order. -/
structure Realizes (m : Memory Nat) (actions : List AtomicAction)
    (ev source : Nat → Nat) : Prop where
  adjacent : ∀ i, i + 1 < actions.length → m.sb (ev i) (ev (i + 1))
  reads : ∀ i loc mode, actions[i]? = some (.load loc mode) →
    m.rf (m.write loc (source i)) (ev i)
  acquire : ∀ i loc, actions[i]? = some (.load loc .acquire) → m.acquire (ev i)
  fence : ∀ i, actions[i]? = some (.fence .acquire) → m.acquireFence (ev i)

private theorem Realizes.before {m : Memory Nat} (run : Realizes m actions ev source)
    {i j : Nat} (lt : i < j) (inside : j < actions.length) : m.sb (ev i) (ev j) := by
  induction j with
  | zero => omega
  | succ j ih =>
    by_cases eq : i = j
    · subst j
      exact run.adjacent i inside
    · exact m.sb_trans (ih (by omega) (by omega)) (run.adjacent j inside)

private def fromPath (m : Memory Nat) (slot : WriterTrace.Slot m) (bigEndian : Bool)
    (ev source : Nat → Nat) (steps : Realizes m occupiedActions ev source)
    (noWrap : source 5 < source 0 + SeqLock.modulus) : RawSample := by
  let reader : SeqLock.RawReader m (source 0) (source 5) :=
    ⟨ev 0, ev 4, ev 5, steps.acquire 0 _ rfl, steps.fence 4 rfl, steps.before (by decide) (by decide),
     steps.adjacent 4 (by decide), steps.reads 0 _ _ rfl, steps.reads 5 _ _ rfl⟩
  let field (index : Nat) (f : SeqLock.Field) (act : occupiedActions[index]? = some (.load (.payload f) .relaxed))
      (positive : 0 < index) (lt : index < 4) : SeqLock.FieldRead reader f (source index) :=
    ⟨ev index, steps.before positive (by change index < 6; omega),
      steps.before lt (by decide), steps.reads index _ _ act⟩
  exact ⟨m, source 0, source 5, slot, reader, source 2, source 3, source 1,
     field 2 .player rfl (by decide) (by decide), field 3 .opponent rfl (by decide) (by decide),
     field 1 .metadata rfl (by decide) (by decide), noWrap, bigEndian⟩

theorem rawSample_of_occupied_return (m : Memory Nat) (slot : WriterTrace.Slot m)
    (bigEndian : Bool) (ev source : Nat → Nat) (result : SeqLock.Snapshot)
    (steps : Realizes m (run bigEndian (m.readValue ∘ ev)).actions ev source)
    (hit : (run bigEndian (m.readValue ∘ ev)).returned = some result)
    (occupied : Snapshot.empty bigEndian result.metadata = false)
    (noWrap : source 5 < source 0 + SeqLock.modulus) :
    ∃ raw : RawSample, raw.memory = m ∧ raw.slot.history = slot.history ∧
      raw.bigEndian = bigEndian ∧ raw.firstIndex = source 0 ∧ raw.reader.first = ev 0 ∧
      raw.slot.payloadAt (raw.firstIndex / 2) = slot.payloadAt (source 0 / 2) ∧
      raw.first &&& 1 = 0 ∧ Snapshot.read raw.bigEndian raw.first raw.last raw.payload = some result ∧
      result = ⟨raw.payload.player, raw.payload.opponent, raw.first, raw.payload.metadata⟩ := by
  obtain ⟨path, even, changeFree, keys⟩ := occupiedPath bigEndian _ _ hit occupied
  rw [path] at steps
  let raw := fromPath m slot bigEndian ev source steps noWrap
  refine ⟨raw, rfl, rfl, rfl, rfl, rfl, rfl, even, ?_, keys⟩
  apply (Snapshot.readContract_proven _ _ _ _ _).mpr
  rw [keys] at occupied ⊢
  change Snapshot.empty raw.bigEndian raw.payload.metadata = false at occupied
  exact ⟨even, changeFree, rfl, rfl, by rw [occupied]; exact ⟨rfl, rfl⟩⟩

/-- The two sequence loads and the three payload loads of an occupied read
    all resolve to the same publication epoch in their own modification orders. -/
theorem occupied_indices (m : Memory Nat) (slot : WriterTrace.Slot m)
    (bigEndian : Bool) (ev source : Nat → Nat) (result : SeqLock.Snapshot)
    (steps : Realizes m (run bigEndian (m.readValue ∘ ev)).actions ev source)
    (hit : (run bigEndian (m.readValue ∘ ev)).returned = some result)
    (occupied : Snapshot.empty bigEndian result.metadata = false)
    (noWrap : source 5 < source 0 + SeqLock.modulus) :
    source 5 = source 0 ∧ source 1 = source 0 / 2 ∧ source 2 = source 0 / 2 ∧
      source 3 = source 0 / 2 ∧ source 0 = 2 * (source 0 / 2) := by
  obtain ⟨path,even,same,_⟩ := occupiedPath bigEndian _ _ hit occupied
  rw [path] at steps
  let raw := fromPath m slot bigEndian ev source steps noWrap
  have first : source 0 = 2 * (source 0 / 2) := raw.reader.evenIndex slot.writers even
  have last : source 0 = source 5 := raw.reader.validatedIndex slot.writers noWrap same
  have short : source 5 < 2 * (source 0 / 2) + SeqLock.modulus := by rwa [← first]
  have fields {f : SeqLock.Field} {k : Nat} (read : SeqLock.FieldRead raw.reader f k) :
      k = source 0 / 2 := by
    apply SeqLock.validated_field_same_epoch slot.writers (raw.reader.asStable slot.writers even)
      (read.asStable slot.writers even) short
    change SeqLock.seqWord (2 * slot.writers.baseEpoch + 2 * (source 0 / 2)) =
      SeqLock.seqWord (2 * slot.writers.baseEpoch + source 5)
    rw [← first,last]
  exact ⟨last.symm,fields raw.metadata,fields raw.player,fields raw.opponent,first⟩

/-- Map the five observed loads back to the occupied return registers, in MIR order. -/
theorem occupied_loads (m : Memory Nat) (bigEndian : Bool) (ev source : Nat → Nat)
    (result : SeqLock.Snapshot) (steps : Realizes m (run bigEndian (m.readValue ∘ ev)).actions ev source)
    (hit : (run bigEndian (m.readValue ∘ ev)).returned = some result)
    (occupied : Snapshot.empty bigEndian result.metadata = false) :
    let load := fun i loc word => m.rf (m.write loc (source i)) (ev i) ∧ m.readValue (ev i) = word
    load 0 .sequence result.sequence ∧ load 1 (.payload .metadata) result.metadata ∧
      load 2 (.payload .player) result.player ∧ load 3 (.payload .opponent) result.opponent ∧
      load 5 .sequence result.sequence := by
  obtain ⟨path,_,same,keys⟩ := occupiedPath bigEndian _ _ hit occupied
  rw [path] at steps
  rw [keys]
  exact ⟨⟨steps.reads 0 _ _ rfl,rfl⟩,⟨steps.reads 1 _ _ rfl,rfl⟩,⟨steps.reads 2 _ _ rfl,rfl⟩,
    ⟨steps.reads 3 _ _ rfl,rfl⟩,⟨steps.reads 5 _ _ rfl,same.symm⟩⟩

theorem origin_of_occupied_return (m : Memory Nat) (slot : WriterTrace.Slot m)
    (bigEndian : Bool) (ev source : Nat → Nat) (result : SeqLock.Snapshot)
    (steps : Realizes m (run bigEndian (m.readValue ∘ ev)).actions ev source)
    (hit : (run bigEndian (m.readValue ∘ ev)).returned = some result)
    (occupied : Snapshot.empty bigEndian result.metadata = false)
    (noWrap : source 5 < source 0 + SeqLock.modulus) :
    let words : SeqLock.Payload := ⟨result.player, result.opponent, result.metadata⟩
    (source 0 = 0 ∧ words = slot.payloadAt 0) ∨
      ∃ call, slot.history[source 0 - 1]? = some (.publish call) ∧ m.hb call.finish (ev 0) ∧
        words = call.payload := by
  obtain ⟨raw, memory, history, endian, index, first, payload, even, accepted, sampled⟩ :=
    rawSample_of_occupied_return m slot bigEndian ev source result steps hit occupied noWrap
  have sampledOccupied : Snapshot.empty raw.bigEndian raw.payload.metadata = false := by
    rw [sampled] at occupied
    simpa only [endian] using occupied
  have writers := raw.slot.writers
  have twice := raw.reader.evenIndex writers even
  have short : raw.finalIndex < 2 * (raw.firstIndex / 2) + SeqLock.modulus := by
    rw [← twice]
    exact raw.noWrap
  have physical := Snapshot.occupiedPublicationContract_proven writers (raw.reader.asStable writers even)
    (raw.player.asStable writers even) (raw.opponent.asStable writers even) (raw.metadata.asStable writers even)
    raw.bigEndian result sampledOccupied short accepted
  have words : (⟨result.player, result.opponent, result.metadata⟩ : SeqLock.Payload) =
      raw.slot.payloadAt (raw.firstIndex / 2) := by rw [physical]; rfl
  by_cases zero : source 0 = 0
  · exact .inl ⟨zero, by simpa only [zero, Nat.zero_div] using words.trans payload⟩
  · obtain ⟨call, atIndex, published, hb⟩ := raw.publishingCall even (by omega)
    exact .inr ⟨call, by simpa only [history, index] using atIndex,
      by simpa only [memory, first] using hb, words.trans published⟩

theorem publishingCall_of_occupied_return (m : Memory Nat) (slot : WriterTrace.Slot m)
    (bigEndian : Bool) (ev source : Nat → Nat) (result : SeqLock.Snapshot)
    (steps : Realizes m (run bigEndian (m.readValue ∘ ev)).actions ev source)
    (hit : (run bigEndian (m.readValue ∘ ev)).returned = some result)
    (occupied : Snapshot.empty bigEndian result.metadata = false)
    (noWrap : source 5 < source 0 + SeqLock.modulus)
    (initial : Snapshot.empty bigEndian (slot.payloadAt 0).metadata = true) :
    ∃ call, slot.history[source 0 - 1]? = some (.publish call) ∧ m.hb call.finish (ev 0) ∧
      (⟨result.player, result.opponent, result.metadata⟩ : SeqLock.Payload) = call.payload := by
  rcases origin_of_occupied_return m slot bigEndian ev source result steps hit occupied noWrap with
    ⟨_, inherited⟩ | written
  · have same := congrArg SeqLock.Payload.metadata inherited
    change result.metadata = _ at same
    rw [same, initial] at occupied
    contradiction
  · exact written

end NeuralReversi.TranspositionTable.ReaderMir

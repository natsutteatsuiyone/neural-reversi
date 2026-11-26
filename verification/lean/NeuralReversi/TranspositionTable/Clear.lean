import NeuralReversi.TranspositionTable.LookupCut

namespace NeuralReversi.TranspositionTable.Clear

open NeuralReversi.TranspositionTable.SeqLock

def clusterSize : Nat := 2

def mulHi64 (key count : Word) : Nat :=
  (key.toNat * count.toNat) / modulus

def getClusterIdx (key clusterCount : Word) : Nat :=
  mulHi64 key clusterCount * clusterSize

structure TTSlot where
  index : Nat
  deriving DecidableEq, Repr

theorem mulHi64_lt (key clusterCount : Word) (hpos : 0 < clusterCount.toNat) :
    mulHi64 key clusterCount < clusterCount.toNat := by
  unfold mulHi64 modulus
  have hkey : key.toNat < 18446744073709551616 := key.isLt
  have hmul : key.toNat * clusterCount.toNat < 18446744073709551616 * clusterCount.toNat :=
    Nat.mul_lt_mul_of_pos_right hkey hpos
  rw [Nat.mul_comm 18446744073709551616 clusterCount.toNat] at hmul
  exact (Nat.div_lt_iff_lt_mul (by omega : 0 < 18446744073709551616)).mpr hmul

theorem probe_slot_in_bounds (key clusterCount : Word) (offset : Nat)
    (hpos : 0 < clusterCount.toNat) (hoff : offset < clusterSize) :
    let slot : TTSlot := ⟨getClusterIdx key clusterCount + offset⟩
    slot.index < clusterCount.toNat * clusterSize := by
  dsimp only [getClusterIdx, clusterSize] at hoff ⊢
  have hlt := mulHi64_lt key clusterCount hpos
  omega

structure SlotState where
  seq : Word
  player : Word
  opponent : Word
  data : Word
  deriving DecidableEq, Repr

def emptySlot : SlotState := ⟨0, 0, 0, 0⟩

def clearTable (entries : List SlotState) : List SlotState :=
  entries.map (fun _ => emptySlot)

theorem clearTable_length (entries : List SlotState) :
    (clearTable entries).length = entries.length := by
  simp [clearTable]

theorem clearTable_getElem (entries : List SlotState) (i : Nat) (hi : i < entries.length) :
    (clearTable entries)[i]'(by simpa [clearTable_length] using hi) = emptySlot := by
  simp [clearTable]

theorem clear_slot_snapshot_empty :
    snapshotDecision emptySlot.seq emptySlot.seq
      ⟨emptySlot.player, emptySlot.opponent, emptySlot.data⟩ (fun d => d == 0) =
      some ⟨0, 0, 0, 0⟩ := by
  decide

theorem clear_slot_lookup_miss (bigEndian : Bool) (qp qo : Word) :
    LookupCut.readOccupied bigEndian emptySlot.seq emptySlot.seq
      ⟨emptySlot.player, emptySlot.opponent, emptySlot.data⟩ qp qo = none := by
  cases hit : LookupCut.readOccupied bigEndian emptySlot.seq emptySlot.seq
      ⟨emptySlot.player, emptySlot.opponent, emptySlot.data⟩ qp qo with
  | none => rfl
  | some entry =>
    obtain ⟨_, _, _, _, decoded, occupied⟩ := (LookupCut.readContract_proven _ _ _ _ _ _ _).mp hit
    change Packing.unpack bigEndian 0 = entry at decoded
    have zero : (Packing.unpack bigEndian 0).bound = 0#8 := by cases bigEndian <;> rfl
    exact (occupied (decoded ▸ zero)).elim

def ExclusiveContract : Prop :=
  ∀ entries, (clearTable entries).length = entries.length ∧
    ∀ i, i < entries.length → (clearTable entries)[i]? = some emptySlot

theorem exclusiveContract_proven : ExclusiveContract := by
  intro entries
  refine ⟨clearTable_length entries, ?_⟩
  intro i valid
  have within : i < (clearTable entries).length := by simpa [clearTable_length] using valid
  exact (List.getElem?_eq_getElem within).trans (congrArg some (clearTable_getElem entries i valid))

inductive ClearOutcome where
  | retry
  | skippedEmpty
  | cleared (nextSeq : Word)
  deriving DecidableEq, Repr

def clearSharedAttempt (firstSeq lastSeq currentSeq dataWord : Word) : ClearOutcome :=
  if firstSeq &&& 1 ≠ 0 then
    .retry
  else if dataWord = 0 then
    if firstSeq = lastSeq then .skippedEmpty else .retry
  else
    if (claim currentSeq firstSeq).1 then .cleared (publish firstSeq) else .retry

theorem clearSharedAttempt_odd_retries (firstSeq lastSeq currentSeq dataWord : Word)
    (hodd : firstSeq &&& 1 ≠ 0) :
    clearSharedAttempt firstSeq lastSeq currentSeq dataWord = .retry := by
  simp only [clearSharedAttempt, ne_eq, ite_eq_left_iff, Decidable.not_not]
  intro h
  exact False.elim (hodd h)

theorem clearSharedAttempt_skippedEmpty_iff (firstSeq lastSeq currentSeq dataWord : Word) :
    clearSharedAttempt firstSeq lastSeq currentSeq dataWord = .skippedEmpty ↔
      firstSeq &&& 1 = 0 ∧ dataWord = 0 ∧ firstSeq = lastSeq := by
  by_cases heven : firstSeq &&& 1 = 0 <;>
    by_cases hdata : dataWord = 0 <;>
    by_cases hsame : firstSeq = lastSeq <;>
    by_cases hclaim : (claim currentSeq firstSeq).1 = true <;>
    simp_all [clearSharedAttempt]

theorem clearSharedAttempt_cleared_iff (firstSeq lastSeq currentSeq dataWord nextSeq : Word) :
    clearSharedAttempt firstSeq lastSeq currentSeq dataWord = .cleared nextSeq ↔
      firstSeq &&& 1 = 0 ∧ dataWord ≠ 0 ∧ currentSeq = firstSeq ∧
      nextSeq = publish firstSeq := by
  by_cases heven : firstSeq &&& 1 = 0 <;>
    by_cases hdata : dataWord = 0 <;>
    by_cases hsame : firstSeq = lastSeq <;>
    by_cases hcur : currentSeq = firstSeq <;>
    simp_all [clearSharedAttempt, claim_success_iff] <;>
    exact eq_comm

theorem clearSharedAttempt_cleared_monotone (epoch : Nat) (lastSeq currentSeq dataWord nextSeq : Word)
    (hclear : clearSharedAttempt (seqWord (2 * epoch)) lastSeq currentSeq dataWord = .cleared nextSeq) :
    nextSeq = seqWord (2 * (epoch + 1)) ∧ nextSeq &&& 1 = 0 := by
  obtain ⟨_, _, _, hpub⟩ := (clearSharedAttempt_cleared_iff _ _ _ _ _).mp hclear
  rw [hpub, publish_seqWord]
  have hidx : 2 * epoch + 2 = 2 * (epoch + 1) := by omega
  rw [hidx]
  exact ⟨rfl, (seqWord_even_iff (2 * (epoch + 1))).mpr (by omega)⟩

structure ClearObservation where
  firstSeq : Word
  lastSeq : Word
  currentSeq : Word
  dataWord : Word

def ClearObservation.outcome (a : ClearObservation) : ClearOutcome :=
  clearSharedAttempt a.firstSeq a.lastSeq a.currentSeq a.dataWord

def clearSharedRun (observations : List ClearObservation) : Option ClearObservation :=
  (observations.take 4).find? (fun a => a.outcome != .retry)

def RetryContract : Prop :=
  ∀ observations,
    (clearSharedRun observations = none ↔ ∀ a ∈ observations.take 4, a.outcome = .retry) ∧
    ∀ selected, clearSharedRun observations = some selected ↔
      ∃ before after, observations.take 4 = before ++ selected :: after ∧
        (∀ a ∈ before, a.outcome = .retry) ∧ selected.outcome ≠ .retry

theorem retryContract_proven : RetryContract := by
  intro observations
  refine ⟨by simp [clearSharedRun, List.find?_eq_none], ?_⟩
  intro selected
  rw [clearSharedRun, List.find?_eq_some_iff_append]
  constructor
  · rintro ⟨hit, before, after, shape, earlier⟩
    exact ⟨before, after, shape, by simpa using earlier, by simpa using hit⟩
  · rintro ⟨before, after, shape, earlier, hit⟩
    exact ⟨by simpa using hit, before, after, shape, by simpa using earlier⟩

def afterAttempt (entry : SlotState) (outcome : ClearOutcome) : SlotState :=
  match outcome with
  | .cleared seq => ⟨seq, 0, 0, 0⟩
  | .retry | .skippedEmpty => entry

def SharedAttemptContract : Prop :=
  (∀ first last current raw,
    clearSharedAttempt first last current raw = .skippedEmpty ↔
      first &&& 1 = 0 ∧ raw = 0 ∧ first = last) ∧
  (∀ first last current raw next,
    clearSharedAttempt first last current raw = .cleared next ↔
      first &&& 1 = 0 ∧ raw ≠ 0 ∧ current = first ∧ next = publish first) ∧
  ∀ entry, entry.seq &&& 1 = 0 →
    let attempt := clearSharedAttempt entry.seq entry.seq entry.seq entry.data
    let result := afterAttempt entry attempt
    result.data = 0#64 ∧
    (entry.data = 0#64 → result = entry) ∧
    (entry.data ≠ 0#64 → result = ⟨publish entry.seq, 0, 0, 0⟩)

theorem sharedAttemptContract_proven : SharedAttemptContract := by
  refine ⟨clearSharedAttempt_skippedEmpty_iff, clearSharedAttempt_cleared_iff, ?_⟩
  intro entry even
  dsimp only
  by_cases empty : entry.data = 0#64
  · have skipped := (clearSharedAttempt_skippedEmpty_iff entry.seq entry.seq entry.seq entry.data).mpr
      ⟨even, empty, rfl⟩
    rw [skipped]
    simp [afterAttempt, empty]
  · have cleared := (clearSharedAttempt_cleared_iff entry.seq entry.seq entry.seq entry.data
      (publish entry.seq)).mpr ⟨even, empty, rfl, rfl⟩
    simp [cleared, afterAttempt, empty]

structure ClearEpoch {Event : Type} {memory : Memory Event}
    (writers : Writers memory) (k : Nat) : Prop where
  positive : 0 < k
  published : writers.published k
  player_zero : memory.writeValue (.payload .player) k = 0
  opponent_zero : memory.writeValue (.payload .opponent) k = 0
  metadata_zero : memory.writeValue (.payload .metadata) k = 0

theorem clear_shared_lookup_no_tear {Event : Type} {memory : Memory Event}
    {epoch finalIndex k : Nat} (writers : Writers memory)
    (hclear : ClearEpoch writers k)
    (reader : Reader memory epoch finalIndex)
    {pe oe de : Nat} (pr : FieldRead reader .player pe)
    (op : FieldRead reader .opponent oe) (dr : FieldRead reader .metadata de)
    (qp qo result : Word) (noWrap : finalIndex < 2 * epoch + modulus)
    (accepted : lookupDecision (memory.readValue reader.first) (memory.readValue reader.last)
      ⟨memory.readValue pr.event, memory.readValue op.event, memory.readValue dr.event⟩ qp qo =
        some result) :
    memory.writeValue (.payload .player) epoch = qp ∧
    memory.writeValue (.payload .opponent) epoch = qo ∧
    result = memory.writeValue (.payload .metadata) epoch ∧
    (epoch = k → qp = 0 ∧ qo = 0 ∧ result = 0) := by
  obtain ⟨hp, ho, hd⟩ := lookup_consistent writers reader pr op dr qp qo result noWrap accepted
  refine ⟨hp, ho, hd, ?_⟩
  intro heq
  subst heq
  exact ⟨hp.symm.trans hclear.player_zero, ho.symm.trans hclear.opponent_zero,
    hd.trans hclear.metadata_zero⟩

theorem clear_shared_occupied_snapshot_no_tear {Event : Type} {memory : Memory Event}
    {epoch finalIndex k : Nat} (writers : Writers memory)
    (hclear : ClearEpoch writers k)
    (reader : Reader memory epoch finalIndex)
    {pe oe de : Nat} (pr : FieldRead reader .player pe)
    (op : FieldRead reader .opponent oe) (dr : FieldRead reader .metadata de)
    (isEmpty : Word → Bool) (hEmptyZero : isEmpty 0 = true) (result : Snapshot)
    (occupied : isEmpty (memory.readValue dr.event) = false)
    (noWrap : finalIndex < 2 * epoch + modulus)
    (accepted : snapshotDecision (memory.readValue reader.first) (memory.readValue reader.last)
      ⟨memory.readValue pr.event, memory.readValue op.event, memory.readValue dr.event⟩ isEmpty =
        some result) :
    result = ⟨memory.writeValue (.payload .player) epoch,
      memory.writeValue (.payload .opponent) epoch, seqWord (2 * writers.baseEpoch + 2 * epoch),
      memory.writeValue (.payload .metadata) epoch⟩ ∧ epoch ≠ k := by
  have hsnap := occupied_snapshot_consistent writers reader pr op dr isEmpty result occupied noWrap accepted
  refine ⟨hsnap, ?_⟩
  intro heq
  subst heq
  obtain ⟨_, same, _⟩ := (snapshot_success_iff _ _ _ _ _).mp accepted
  rw [reader_first_value writers reader, reader_last_value writers reader] at same
  have vd := validated_field_value writers reader dr noWrap same
  rw [vd, hclear.metadata_zero] at occupied
  rw [hEmptyZero] at occupied
  cases occupied

theorem clear_shared_empty_snapshot_no_tear {Event : Type} {memory : Memory Event}
    {epoch finalIndex : Nat} (writers : Writers memory)
    (reader : Reader memory epoch finalIndex)
    {de : Nat} (dr : FieldRead reader .metadata de) (unusedPlayer unusedOpponent : Word)
    (isEmpty : Word → Bool) (result : Snapshot)
    (empty : isEmpty (memory.readValue dr.event) = true)
    (noWrap : finalIndex < 2 * epoch + modulus)
    (accepted : snapshotDecision (memory.readValue reader.first) (memory.readValue reader.last)
      ⟨unusedPlayer, unusedOpponent, memory.readValue dr.event⟩ isEmpty = some result) :
    result = ⟨0, 0, seqWord (2 * writers.baseEpoch + 2 * epoch),
      memory.writeValue (.payload .metadata) epoch⟩ :=
  empty_snapshot_consistent writers reader dr unusedPlayer unusedOpponent isEmpty result empty noWrap accepted

theorem clear_shared_fast_skip_sound {Event : Type} {memory : Memory Event}
    {epoch finalIndex : Nat} (writers : Writers memory)
    (skipReader : Reader memory epoch finalIndex)
    {de : Nat} (dr : FieldRead skipReader .metadata de)
    (noWrap : finalIndex < 2 * epoch + modulus)
    (currentSeq : Word)
    (hskip : clearSharedAttempt (memory.readValue skipReader.first)
      (memory.readValue skipReader.last) currentSeq (memory.readValue dr.event) = .skippedEmpty) :
    memory.writeValue (.payload .metadata) epoch = 0 := by
  obtain ⟨_, hzero, hsame⟩ := (clearSharedAttempt_skippedEmpty_iff _ _ _ _).mp hskip
  rw [reader_first_value writers skipReader, reader_last_value writers skipReader] at hsame
  have vd := validated_field_value writers skipReader dr noWrap hsame
  exact vd.symm.trans hzero

theorem post_clear_reader_sees_empty_snapshot {Event : Type} {memory : Memory Event}
    {epoch finalIndex k : Nat} (writers : Writers memory)
    (hclear : ClearEpoch writers k)
    (reader : Reader memory epoch finalIndex)
    {de : Nat} (dr : FieldRead reader .metadata de)
    (unusedPlayer unusedOpponent : Word)
    (afterClear : memory.hb (memory.write .sequence (2 * k)) reader.first)
    (noLaterWrite : finalIndex ≤ 2 * k) :
    epoch = k ∧
    memory.readValue dr.event = 0 ∧
    snapshotDecision (memory.readValue reader.first) (memory.readValue reader.last)
      ⟨unusedPlayer, unusedOpponent, memory.readValue dr.event⟩ (fun d => d == 0) =
      some ⟨0, 0, seqWord (2 * writers.baseEpoch + 2 * k), 0⟩ := by
  have hge : 2 * k ≤ 2 * epoch := memory.write_read afterClear reader.first_rf
  have hmono : 2 * epoch ≤ finalIndex := reader.monotone
  have hepoch : epoch = k := by omega
  have hfinal : finalIndex = 2 * k := by omega
  have hnoWrap : finalIndex < 2 * epoch + modulus := by unfold modulus; omega
  have hsame : seqWord (2 * writers.baseEpoch + 2 * epoch) =
      seqWord (2 * writers.baseEpoch + finalIndex) := by rw [hepoch, hfinal]
  have vd : memory.readValue dr.event = 0 := by
    rw [validated_field_value writers reader dr hnoWrap hsame, hepoch, hclear.metadata_zero]
  refine ⟨hepoch, vd, ?_⟩
  rw [reader_first_value writers reader, reader_last_value writers reader, vd, hepoch, hfinal]
  have heven : seqWord (2 * writers.baseEpoch + 2 * k) &&& 1 = 0 :=
    (seqWord_even_iff _).mpr (by omega)
  simp [snapshotDecision]
  exact heven

def PublishedClearContract : Prop :=
  ∀ {Event : Type} {memory : Memory Event}
    {epoch finalIndex k : Nat} (writers : Writers memory)
    (_hclear : ClearEpoch writers k)
    (reader : Reader memory epoch finalIndex)
    {pe oe de : Nat} (pr : FieldRead reader .player pe)
    (op : FieldRead reader .opponent oe) (dr : FieldRead reader .metadata de)
    (qp qo : Word) (bigEndian : Bool),
    memory.hb (memory.write .sequence (2 * k)) reader.first →
    finalIndex ≤ 2 * k →
    LookupCut.readOccupied bigEndian (memory.readValue reader.first) (memory.readValue reader.last)
      ⟨memory.readValue pr.event, memory.readValue op.event, memory.readValue dr.event⟩ qp qo =
      none

theorem post_clear_reader_lookup_misses : PublishedClearContract := by
  intro Event memory epoch finalIndex k writers hclear reader pe oe de pr op dr qp qo bigEndian afterClear noLaterWrite
  have hge : 2 * k ≤ 2 * epoch := memory.write_read afterClear reader.first_rf
  have hmono : 2 * epoch ≤ finalIndex := reader.monotone
  have hepoch : epoch = k := by omega
  have hnoWrap : finalIndex < 2 * epoch + modulus := by unfold modulus; omega
  cases result : LookupCut.readOccupied bigEndian (memory.readValue reader.first)
    (memory.readValue reader.last)
    ⟨memory.readValue pr.event, memory.readValue op.event, memory.readValue dr.event⟩ qp qo with
  | none => rfl
  | some entry =>
    obtain ⟨_, _, decoded, occupied⟩ := LookupCut.publishedReadContract_proven
      writers reader pr op dr qp qo bigEndian entry hnoWrap result
    rw [hepoch, hclear.metadata_zero] at decoded
    have zero : (Packing.unpack bigEndian 0).bound = 0#8 := by cases bigEndian <;> rfl
    exact (occupied (decoded ▸ zero)).elim

example : clearSharedAttempt 1 1 1 9 = .retry := by decide
example : clearSharedAttempt 2 2 2 0 = .skippedEmpty := by decide
example : clearSharedAttempt 2 4 2 0 = .retry := by decide
example : clearSharedAttempt 2 2 2 9 = .cleared 4 := by decide
example : clearSharedAttempt 2 2 3 9 = .retry := by decide
example : clearSharedAttempt 2 2 4 9 = .retry := by decide

end NeuralReversi.TranspositionTable.Clear

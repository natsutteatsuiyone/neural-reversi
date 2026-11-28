module

@[expose] public section

namespace NeuralReversi.TranspositionTable.SeqLock

abbrev Word := BitVec 64

def modulus : Nat := 18446744073709551616

def seqWord (index : Nat) : Word := BitVec.ofNat 64 index

theorem seqWord_local_injective (first last : Nat)
    (ordered : first ≤ last) (short : last < first + modulus)
    (equal : seqWord first = seqWord last) : first = last := by
  have residues := congrArg BitVec.toNat equal
  change first % 18446744073709551616 = last % 18446744073709551616 at residues
  unfold modulus at short
  omega

theorem seqWord_even_iff (index : Nat) :
    seqWord index &&& 1 = 0 ↔ index % 2 = 0 := by
  have bits : (seqWord index &&& 1).toNat = (index % 18446744073709551616) % 2 := by
    simp [seqWord, BitVec.toNat_and, Nat.and_one_is_mod]
  constructor
  · intro even
    have zero := congrArg BitVec.toNat even
    rw [bits] at zero
    change (index % 18446744073709551616) % 2 = 0 at zero
    omega
  · intro even
    apply BitVec.eq_of_toNat_eq
    rw [bits]
    change (index % 18446744073709551616) % 2 = 0
    omega

theorem even_index_epoch (index : Nat) (even : seqWord index &&& 1 = 0) :
    index = 2 * (index / 2) := by
  have parity := (seqWord_even_iff index).mp even
  omega

def claim (current expected : Word) : Bool × Word :=
  if current = expected then (true, expected ||| 1) else (false, current)

theorem claim_success_iff (current expected : Word) :
    (claim current expected).1 = true ↔ current = expected := by
  by_cases equal : current = expected <;> simp [claim, equal]

theorem successful_claim_fresh (snapshotIndex currentIndex : Nat)
    (ordered : snapshotIndex ≤ currentIndex)
    (noWrap : currentIndex < snapshotIndex + modulus)
    (success : (claim (seqWord currentIndex) (seqWord snapshotIndex)).1 = true) :
    snapshotIndex = currentIndex := by
  exact seqWord_local_injective snapshotIndex currentIndex ordered noWrap
    (((claim_success_iff _ _).mp success).symm)

theorem claim_failure_frame (current expected : Word) (different : current ≠ expected) :
    claim current expected = (false, current) := by
  simp [claim, different]

theorem claimed_is_odd (expected : Word) : (expected ||| 1) &&& 1 ≠ 0 := by
  intro equal
  have bit := congrArg (fun w : Word => w.getLsbD 0) equal
  simp at bit

theorem competing_claim_fails (ownerExpected contenderExpected : Word)
    (even : contenderExpected &&& 1 = 0) :
    claim (ownerExpected ||| 1) contenderExpected = (false, ownerExpected ||| 1) := by
  apply claim_failure_frame
  intro equal
  have impossible := claimed_is_odd ownerExpected
  apply impossible
  rw [equal]
  exact even

def publish (expected : Word) : Word := expected + 2

theorem publish_seqWord (index : Nat) : publish (seqWord index) = seqWord (index + 2) := by
  apply BitVec.eq_of_toNat_eq
  simp [publish, seqWord, BitVec.toNat_add, Nat.add_mod]

theorem publish_even (epoch : Nat) : publish (seqWord (2 * epoch)) &&& 1 = 0 := by
  rw [publish_seqWord, seqWord_even_iff]
  omega

inductive Field where
  | player | opponent | metadata
  deriving DecidableEq, Repr

inductive Location where
  | sequence
  | payload (field : Field)
  deriving DecidableEq, Repr

structure Memory (Event : Type) where
  sb : Event → Event → Prop
  hb : Event → Event → Prop
  rf : Event → Event → Prop
  write : Location → Nat → Event
  writeValue : Location → Nat → Word
  readValue : Event → Word
  rf_value : ∀ {loc i r}, rf (write loc i) r → readValue r = writeValue loc i
  release : Event → Prop
  acquire : Event → Prop
  releaseFence : Event → Prop
  acquireFence : Event → Prop
  sb_hb : ∀ {a b}, sb a b → hb a b
  sb_trans : ∀ {a b c}, sb a b → sb b c → sb a c
  hb_trans : ∀ {a b c}, hb a b → hb b c → hb a c
  hb_irrefl : ∀ a, ¬ hb a a
  release_acquire : ∀ {w r}, release w → acquire r → rf w r → hb w r
  fence_fence : ∀ {a w r b}, releaseFence a → acquireFence b →
    sb a w → rf w r → sb r b → hb a b
  write_write : ∀ {loc i j}, hb (write loc i) (write loc j) → i < j
  write_read : ∀ {loc i j r}, hb (write loc i) r → rf (write loc j) r → i ≤ j
  read_read : ∀ {loc i j a b}, hb a b → rf (write loc i) a →
    rf (write loc j) b → i ≤ j

structure Writers {Event : Type} (memory : Memory Event) where
  baseEpoch : Nat
  begun : Nat → Prop
  published : Nat → Prop
  fence : Nat → Event
  sequence_value : ∀ i r, memory.rf (memory.write .sequence i) r →
    memory.writeValue .sequence i = seqWord (2 * baseEpoch + i)
  publication_read : ∀ k r, 0 < k →
    memory.rf (memory.write .sequence (2 * k)) r → published k
  payload_read : ∀ k f r, 0 < k →
    memory.rf (memory.write (.payload f) k) r → begun k
  publish_release : ∀ k, published k → memory.release (memory.write .sequence (2 * k))
  payload_before_publish : ∀ k f, published k →
    memory.sb (memory.write (.payload f) k) (memory.write .sequence (2 * k))
  payload_fences : ∀ k f r, memory.rf (memory.write (.payload f) k) r → begun k →
    memory.releaseFence (fence k) ∧
    memory.sb (memory.write .sequence (2 * k - 1)) (fence k) ∧
    memory.sb (fence k) (memory.write (.payload f) k)

structure RawReader {Event : Type} (memory : Memory Event) (firstIndex finalIndex : Nat) where
  first : Event
  fence : Event
  last : Event
  first_acquire : memory.acquire first
  fence_acquire : memory.acquireFence fence
  first_before_fence : memory.sb first fence
  fence_before_last : memory.sb fence last
  first_rf : memory.rf (memory.write .sequence firstIndex) first
  last_rf : memory.rf (memory.write .sequence finalIndex) last

abbrev Reader {Event : Type} (memory : Memory Event) (epoch finalIndex : Nat) :=
  RawReader memory (2 * epoch) finalIndex

theorem RawReader.evenIndex {memory : Memory Event}
    (raw : RawReader memory firstIndex finalIndex) (writers : Writers memory)
    (even : memory.readValue raw.first &&& 1 = 0) : firstIndex = 2 * (firstIndex / 2) := by
  have value := (memory.rf_value raw.first_rf).trans (writers.sequence_value _ _ raw.first_rf)
  rw [value] at even
  have parity := (seqWord_even_iff _).mp even
  omega

def RawReader.asStable {memory : Memory Event}
    (raw : RawReader memory firstIndex finalIndex) (writers : Writers memory)
    (even : memory.readValue raw.first &&& 1 = 0) : Reader memory (firstIndex / 2) finalIndex where
  first := raw.first
  fence := raw.fence
  last := raw.last
  first_acquire := raw.first_acquire
  fence_acquire := raw.fence_acquire
  first_before_fence := raw.first_before_fence
  fence_before_last := raw.fence_before_last
  first_rf := by rw [← raw.evenIndex writers even]; exact raw.first_rf
  last_rf := raw.last_rf

structure FieldRead {Event : Type} {memory : Memory Event} {firstIndex finalIndex : Nat}
    (reader : RawReader memory firstIndex finalIndex) (field : Field) (sourceEpoch : Nat) where
  event : Event
  after_first : memory.sb reader.first event
  before_fence : memory.sb event reader.fence
  reads_from : memory.rf (memory.write (.payload field) sourceEpoch) event

def FieldRead.asStable {memory : Memory Event} {raw : RawReader memory firstIndex finalIndex}
    (read : FieldRead raw field sourceEpoch) (writers : Writers memory)
    (even : memory.readValue raw.first &&& 1 = 0) : FieldRead (raw.asStable writers even) field sourceEpoch :=
  ⟨read.event, read.after_first, read.before_fence, read.reads_from⟩

theorem RawReader.monotone {memory : Memory Event} (reader : RawReader memory firstIndex finalIndex) :
    firstIndex ≤ finalIndex := by
  exact memory.read_read
    (memory.hb_trans (memory.sb_hb reader.first_before_fence)
      (memory.sb_hb reader.fence_before_last)) reader.first_rf reader.last_rf

theorem RawReader.validatedIndex {memory : Memory Event}
    (raw : RawReader memory firstIndex finalIndex) (writers : Writers memory)
    (noWrap : finalIndex < firstIndex + modulus)
    (validated : memory.readValue raw.first = memory.readValue raw.last) : firstIndex = finalIndex := by
  rw [memory.rf_value raw.first_rf, writers.sequence_value _ _ raw.first_rf,
    memory.rf_value raw.last_rf, writers.sequence_value _ _ raw.last_rf] at validated
  have same := seqWord_local_injective _ _ (by have := raw.monotone; omega) (by omega) validated
  omega

theorem Reader.publicationAcquired {memory : Memory Event} {epoch finalIndex : Nat}
    (reader : Reader memory epoch finalIndex) (writers : Writers memory) (positive : 0 < epoch) :
    memory.hb (memory.write .sequence (2 * epoch)) reader.first := by
  have active := writers.publication_read epoch reader.first positive reader.first_rf
  exact memory.release_acquire (writers.publish_release epoch active) reader.first_acquire reader.first_rf

theorem field_not_older {Event : Type} {memory : Memory Event} {epoch finalIndex : Nat}
    (writers : Writers memory) (reader : Reader memory epoch finalIndex)
    {field : Field} {sourceEpoch : Nat} (read : FieldRead reader field sourceEpoch) :
    epoch ≤ sourceEpoch := by
  by_cases zero : epoch = 0
  · omega
  · have positive : 0 < epoch := by omega
    have active := writers.publication_read epoch reader.first positive reader.first_rf
    have sync := reader.publicationAcquired writers positive
    have visible := memory.hb_trans
      (memory.sb_hb (writers.payload_before_publish epoch field active))
      (memory.hb_trans sync (memory.sb_hb read.after_first))
    exact memory.write_read visible read.reads_from

theorem field_claim_visible {Event : Type} {memory : Memory Event} {epoch finalIndex : Nat}
    (writers : Writers memory) (reader : Reader memory epoch finalIndex)
    {field : Field} {sourceEpoch : Nat} (read : FieldRead reader field sourceEpoch)
    (positive : 0 < sourceEpoch) : 2 * sourceEpoch - 1 ≤ finalIndex := by
  have active := writers.payload_read sourceEpoch field read.event positive read.reads_from
  obtain ⟨release, claimBefore, storeAfter⟩ :=
    writers.payload_fences sourceEpoch field read.event read.reads_from active
  have sync := memory.fence_fence release reader.fence_acquire storeAfter read.reads_from read.before_fence
  have visible := memory.hb_trans
    (memory.sb_hb claimBefore)
    (memory.hb_trans sync (memory.sb_hb reader.fence_before_last))
  exact memory.write_read visible reader.last_rf

theorem same_epoch_of_validated (baseEpoch epoch finalIndex sourceEpoch : Nat)
    (ordered : 2 * epoch ≤ finalIndex)
    (noWrap : finalIndex < 2 * epoch + modulus)
    (validated : seqWord (2 * baseEpoch + 2 * epoch) = seqWord (2 * baseEpoch + finalIndex))
    (lower : epoch ≤ sourceEpoch)
    (upper : 0 < sourceEpoch → 2 * sourceEpoch - 1 ≤ finalIndex) :
    sourceEpoch = epoch := by
  have same := seqWord_local_injective _ _ (by omega) (by omega) validated
  by_cases zero : sourceEpoch = 0
  · omega
  · have visible := upper (by omega)
    omega

theorem validated_field_same_epoch {Event : Type} {memory : Memory Event}
    {epoch finalIndex : Nat} (writers : Writers memory)
    (reader : Reader memory epoch finalIndex) {field : Field} {sourceEpoch : Nat}
    (read : FieldRead reader field sourceEpoch)
    (noWrap : finalIndex < 2 * epoch + modulus)
    (validated : seqWord (2 * writers.baseEpoch + 2 * epoch) =
      seqWord (2 * writers.baseEpoch + finalIndex)) : sourceEpoch = epoch := by
  exact same_epoch_of_validated writers.baseEpoch epoch finalIndex sourceEpoch
    reader.monotone noWrap validated
    (field_not_older writers reader read) (field_claim_visible writers reader read)

theorem publication_budget (epoch later : Nat) (ordered : epoch ≤ later)
    (budget : later - epoch < 9223372036854775808) :
    2 * later < 2 * epoch + modulus := by
  have delta := Nat.sub_add_cancel ordered
  unfold modulus
  omega

theorem alias_requires_full_cycle (epoch later : Nat) (ordered : epoch ≤ later)
    (different : epoch ≠ later) (equal : seqWord (2 * epoch) = seqWord (2 * later)) :
    9223372036854775808 ≤ later - epoch := by
  by_cases enough : 9223372036854775808 ≤ later - epoch
  · exact enough
  · have short := publication_budget epoch later ordered (by omega)
    have same := seqWord_local_injective (2 * epoch) (2 * later) (by omega) short equal
    omega


structure Payload where
  player : Word
  opponent : Word
  metadata : Word
  deriving DecidableEq, Repr

structure Snapshot where
  player : Word
  opponent : Word
  sequence : Word
  metadata : Word
  deriving DecidableEq, Repr

theorem full_key_eq (lp lo rp ro : Word) :
    ((lp ^^^ rp) ||| (lo ^^^ ro)) = 0 ↔ lp = rp ∧ lo = ro := by
  constructor
  · intro equal
    constructor
    all_goals
      apply BitVec.eq_of_getLsbD_eq
      intro i _
      have bit := congrArg (fun w : Word => w.getLsbD i) equal
      cases hp : lp.getLsbD i <;> cases ho : lo.getLsbD i <;>
        cases hq : rp.getLsbD i <;> cases hr : ro.getLsbD i <;> simp_all
  · rintro ⟨rfl, rfl⟩
    simp

def lookupDecision (first last : Word) (sample : Payload) (qp qo : Word) : Option Word :=
  if first &&& 1 ≠ 0 then none
  else if sample.player = qp ∧ sample.opponent = qo then
    if first = last then some sample.metadata else none
  else none

theorem lookup_success_iff (first last : Word) (sample : Payload) (qp qo result : Word) :
    lookupDecision first last sample qp qo = some result ↔
      first &&& 1 = 0 ∧ sample.player = qp ∧ sample.opponent = qo ∧
      first = last ∧ sample.metadata = result := by
  by_cases even : first &&& 1 = 0 <;>
    by_cases player : sample.player = qp <;>
    by_cases opponent : sample.opponent = qo <;>
    by_cases stable : first = last <;>
    simp_all [lookupDecision]

def snapshotValue (first : Word) (sample : Payload) (isEmpty : Word → Bool) : Snapshot :=
  if isEmpty sample.metadata then ⟨0, 0, first, sample.metadata⟩
  else ⟨sample.player, sample.opponent, first, sample.metadata⟩

def snapshotDecision (first last : Word) (sample : Payload)
    (isEmpty : Word → Bool) : Option Snapshot :=
  if first &&& 1 ≠ 0 then none
  else if isEmpty sample.metadata then
    if first ≠ last then none else some ⟨0, 0, first, sample.metadata⟩
  else if first ≠ last then none
  else some ⟨sample.player, sample.opponent, first, sample.metadata⟩

theorem snapshot_success_iff (first last : Word) (sample : Payload)
    (isEmpty : Word → Bool) (result : Snapshot) :
    snapshotDecision first last sample isEmpty = some result ↔
      first &&& 1 = 0 ∧ first = last ∧ result = snapshotValue first sample isEmpty := by
  by_cases even : first &&& 1 = 0 <;> by_cases stable : first = last <;>
    cases empty : isEmpty sample.metadata <;>
    simp_all [snapshotDecision, snapshotValue] <;> exact eq_comm

theorem reader_first_value {Event : Type} {memory : Memory Event} {epoch finalIndex : Nat}
    (writers : Writers memory) (reader : Reader memory epoch finalIndex) :
    memory.readValue reader.first = seqWord (2 * writers.baseEpoch + 2 * epoch) := by
  exact (memory.rf_value reader.first_rf).trans (writers.sequence_value _ _ reader.first_rf)

theorem reader_last_value {Event : Type} {memory : Memory Event} {epoch finalIndex : Nat}
    (writers : Writers memory) (reader : Reader memory epoch finalIndex) :
    memory.readValue reader.last = seqWord (2 * writers.baseEpoch + finalIndex) := by
  exact (memory.rf_value reader.last_rf).trans (writers.sequence_value _ _ reader.last_rf)

theorem validated_field_value {Event : Type} {memory : Memory Event} {epoch finalIndex : Nat}
    (writers : Writers memory) (reader : Reader memory epoch finalIndex)
    {field : Field} {sourceEpoch : Nat} (read : FieldRead reader field sourceEpoch)
    (noWrap : finalIndex < 2 * epoch + modulus)
    (validated : seqWord (2 * writers.baseEpoch + 2 * epoch) =
      seqWord (2 * writers.baseEpoch + finalIndex)) :
    memory.readValue read.event = memory.writeValue (.payload field) epoch := by
  have same := validated_field_same_epoch writers reader read noWrap validated
  calc
    memory.readValue read.event = memory.writeValue (.payload field) sourceEpoch :=
      memory.rf_value read.reads_from
    _ = memory.writeValue (.payload field) epoch := congrArg _ same

theorem lookup_consistent {Event : Type} {memory : Memory Event} {epoch finalIndex : Nat}
    (writers : Writers memory) (reader : Reader memory epoch finalIndex)
    {pe oe de : Nat} (pr : FieldRead reader .player pe)
    (op : FieldRead reader .opponent oe) (dr : FieldRead reader .metadata de)
    (qp qo result : Word) (noWrap : finalIndex < 2 * epoch + modulus)
    (accepted : lookupDecision (memory.readValue reader.first) (memory.readValue reader.last)
      ⟨memory.readValue pr.event, memory.readValue op.event, memory.readValue dr.event⟩ qp qo =
        some result) :
    memory.writeValue (.payload .player) epoch = qp ∧
    memory.writeValue (.payload .opponent) epoch = qo ∧
    result = memory.writeValue (.payload .metadata) epoch := by
  obtain ⟨_, keyP, keyO, same, data⟩ :=
    (lookup_success_iff _ _ _ _ _ _).mp accepted
  rw [reader_first_value writers reader, reader_last_value writers reader] at same
  have vp := validated_field_value writers reader pr noWrap same
  have vo := validated_field_value writers reader op noWrap same
  have vd := validated_field_value writers reader dr noWrap same
  exact ⟨vp.symm.trans keyP, vo.symm.trans keyO, data.symm.trans vd⟩

theorem occupied_snapshot_consistent {Event : Type} {memory : Memory Event}
    {epoch finalIndex : Nat} (writers : Writers memory) (reader : Reader memory epoch finalIndex)
    {pe oe de : Nat} (pr : FieldRead reader .player pe)
    (op : FieldRead reader .opponent oe) (dr : FieldRead reader .metadata de)
    (isEmpty : Word → Bool) (result : Snapshot)
    (occupied : isEmpty (memory.readValue dr.event) = false)
    (noWrap : finalIndex < 2 * epoch + modulus)
    (accepted : snapshotDecision (memory.readValue reader.first) (memory.readValue reader.last)
      ⟨memory.readValue pr.event, memory.readValue op.event, memory.readValue dr.event⟩ isEmpty =
        some result) :
    result = ⟨memory.writeValue (.payload .player) epoch,
      memory.writeValue (.payload .opponent) epoch, seqWord (2 * writers.baseEpoch + 2 * epoch),
      memory.writeValue (.payload .metadata) epoch⟩ := by
  obtain ⟨_, same, output⟩ := (snapshot_success_iff _ _ _ _ _).mp accepted
  rw [reader_first_value writers reader, reader_last_value writers reader] at same
  have vp := validated_field_value writers reader pr noWrap same
  have vo := validated_field_value writers reader op noWrap same
  have vd := validated_field_value writers reader dr noWrap same
  simp only [snapshotValue, occupied, Bool.false_eq_true, ↓reduceIte] at output
  simpa only [reader_first_value writers reader, vp, vo, vd] using output

theorem empty_snapshot_consistent {Event : Type} {memory : Memory Event}
    {epoch finalIndex : Nat} (writers : Writers memory) (reader : Reader memory epoch finalIndex)
    {de : Nat} (dr : FieldRead reader .metadata de) (unusedPlayer unusedOpponent : Word)
    (isEmpty : Word → Bool) (result : Snapshot)
    (empty : isEmpty (memory.readValue dr.event) = true)
    (noWrap : finalIndex < 2 * epoch + modulus)
    (accepted : snapshotDecision (memory.readValue reader.first) (memory.readValue reader.last)
      ⟨unusedPlayer, unusedOpponent, memory.readValue dr.event⟩ isEmpty = some result) :
    result = ⟨0, 0, seqWord (2 * writers.baseEpoch + 2 * epoch),
      memory.writeValue (.payload .metadata) epoch⟩ := by
  obtain ⟨_, same, output⟩ := (snapshot_success_iff _ _ _ _ _).mp accepted
  rw [reader_first_value writers reader, reader_last_value writers reader] at same
  have vd := validated_field_value writers reader dr noWrap same
  simp only [snapshotValue, empty, ↓reduceIte] at output
  simpa only [reader_first_value writers reader, vd] using output

example : lookupDecision 1 1 ⟨7, 8, 9⟩ 7 8 = none := by decide
example : lookupDecision 0 0 ⟨7, 8, 9⟩ 7 10 = none := by decide
example : lookupDecision 0 2 ⟨7, 8, 9⟩ 7 8 = none := by decide
example : lookupDecision 0 0 ⟨7, 8, 9⟩ 7 8 = some 9 := by decide
example : snapshotDecision 0 0 ⟨7, 8, 0⟩ (fun d => d == 0) = some ⟨0, 0, 0, 0⟩ := by decide
example : snapshotDecision 2 2 ⟨7, 8, 9⟩ (fun d => d == 0) = some ⟨7, 8, 2, 9⟩ := by decide

example : seqWord 0 = seqWord 18446744073709551616 := by decide
example : seqWord 18446744073709551614 + 2 = seqWord 18446744073709551616 := by decide
example : claim (0 ||| 1) 0 = (false, 1) := by decide
example : claim (0 ||| 1) 2 = (false, 1) := by decide

example : seqWord (2 * 1) = seqWord 2 ∧ 2 < 2 * 1 + modulus ∧
    (0 < 0 → 2 * 0 - 1 ≤ 2) ∧ ¬ (1 ≤ 0) := by decide
example : seqWord (2 * 0) = seqWord 0 ∧ 0 < 2 * 0 + modulus ∧
    0 ≤ 1 ∧ ¬ (2 * 1 - 1 ≤ 0) := by decide
example : seqWord 0 = seqWord modulus ∧ 0 ≤ 0 ∧ 0 ≤ 1 ∧
    (0 < 0 → 2 * 0 - 1 ≤ modulus) ∧ (2 * 1 - 1 ≤ modulus) ∧
    ¬ (modulus < modulus) := by decide

end NeuralReversi.TranspositionTable.SeqLock

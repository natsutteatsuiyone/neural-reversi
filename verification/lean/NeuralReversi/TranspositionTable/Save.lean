import NeuralReversi.TranspositionTable.Packing
import NeuralReversi.Bitboard.MovesSpec
import NeuralReversi.TranspositionTable.Snapshot

namespace NeuralReversi.TranspositionTable.Save

open Packing

abbrev Key := BitVec 64 × BitVec 64

def proposal (key storedKey : Key) (data stored : EntryData) : Option EntryData :=
  if data.bound = boundByte .exact then some data
  else if stored.bound = 0 then some data
  else
    let same := sameBoardBits key.1 key.2 storedKey.1 storedKey.2
    let depth := data.depth.setWidth 32
    if same = false ∨ stored.depth.toNat ≤ (depth + 2).toNat ∨
        (selectivityRank stored.selectivity < data.selectivity.toNat ∧
          stored.depth.toNat ≤ (depth + 4).toNat) ∨ data.generation ≠ stored.generation then
      if data.bestMove ≠ noMove ∨ same = false then some data
      else some (withBestMoveOf data stored)
    else none

def Reject (key storedKey : Key) (data stored : EntryData) : Prop :=
  data.bound ≠ boundByte .exact ∧ stored.bound ≠ 0 ∧ key = storedKey ∧
    data.generation = stored.generation ∧
    (if min stored.selectivity.toNat 5 < data.selectivity.toNat then 4 else 2) <
      stored.depth.toNat - data.depth.toNat

instance (key storedKey : Key) (data stored : EntryData) : Decidable (Reject key storedKey data stored) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

def payload (key storedKey : Key) (data stored : EntryData) : EntryData :=
  if data.bound ≠ boundByte .exact ∧ stored.bound ≠ 0 ∧ key = storedKey ∧
      data.bestMove = noMove then { data with bestMove := stored.bestMove }
  else data

private theorem widened_add (d : BitVec 8) (k : BitVec 32) (h : k.toNat ≤ 4) :
    (d.setWidth 32 + k).toNat = d.toNat + k.toNat := by
  have hd : d.toNat < 256 := d.toFin.isLt
  simp only [BitVec.toNat_add, depth_widen]
  apply Nat.mod_eq_of_lt
  omega

private theorem decision (key storedKey : Key) (data stored : EntryData) :
    proposal key storedKey data stored =
      if Reject key storedKey data stored then none else some (payload key storedKey data stored) := by
  have same : sameBoardBits key.1 key.2 storedKey.1 storedKey.2 = true ↔ key = storedKey := by
    rw [sameBoardBits_exact, Prod.ext_iff]
  have different : sameBoardBits key.1 key.2 storedKey.1 storedKey.2 = false ↔ key ≠ storedKey := by
    simp [← same]
  simp only [proposal, Reject, payload, different, widened_add _ 2 (by decide),
    widened_add _ 4 (by decide), selectivityRank_eq_min]
  by_cases exactBound : data.bound = boundByte .exact
  · simp [exactBound]
  by_cases empty : stored.bound = 0
  · simp [empty]
  by_cases sameKey : key = storedKey
  · simp only [exactBound, empty, sameKey, true_and, ↓reduceIte, withBestMoveOf]
    split <;> rename_i gate
    · split <;> split <;> simp_all <;> intro gen <;> simp [gen] at gate <;> omega
    · split <;> simp_all <;> omega
  · simp [exactBound, sameKey]

def legal (key : Key) (move : BitVec 8) : Prop :=
  Bitboard.Moves.LegalMove key.1 key.2 move.toNat

def Contract : Prop :=
  ∀ key storedKey data stored,
    (proposal key storedKey data stored =
      if Reject key storedKey data stored then none else some (payload key storedKey data stored)) ∧
    ((data.bestMove ≠ noMove → legal key data.bestMove) →
      (stored.bound ≠ 0 → stored.bestMove ≠ noMove → legal storedKey stored.bestMove) →
      ∀ written, proposal key storedKey data stored = some written →
        written.bestMove ≠ noMove → legal key written.bestMove)

theorem contract_proven : Contract := by
  intro key storedKey data stored
  refine ⟨decision .., ?_⟩
  intro incoming old written result present
  rw [decision] at result
  split at result
  · contradiction
  · cases Option.some.inj result
    unfold payload at *
    split at * <;> simp_all

structure Observation where
  first : SeqLock.Word
  last : SeqLock.Word
  sample : SeqLock.Payload
  current : SeqLock.Word

inductive Outcome where
  | retry
  | rejected
  | written (seq : SeqLock.Word) (entry : EntryData)
  deriving DecidableEq, Repr

def claimWrite (current observed : SeqLock.Word) (data : EntryData) : Outcome :=
  if (SeqLock.claim current observed).1 then .written (SeqLock.publish observed) data else .retry

def attempt (bigEndian : Bool) (key : Key) (data : EntryData) (observation : Observation) : Outcome :=
  if data.bound = boundByte .exact then
    if observation.first &&& 1 = 0 then claimWrite observation.current observation.first data else .retry
  else match Snapshot.read bigEndian observation.first observation.last observation.sample with
    | none => .retry
    | some slot => match proposal key (slot.player, slot.opponent) data (unpack bigEndian slot.metadata) with
      | none => .rejected
      | some write => claimWrite observation.current slot.sequence write

theorem claimWrite_iff (current observed seq : SeqLock.Word) (data entry : EntryData) :
    claimWrite current observed data = .written seq entry ↔
      current = observed ∧ SeqLock.publish observed = seq ∧ data = entry := by
  by_cases fresh : current = observed <;> simp [claimWrite, SeqLock.claim_success_iff, fresh]

def ExactAttemptContract : Prop :=
  ∀ bigEndian key data observation, data.bound = boundByte .exact →
    attempt bigEndian key data observation =
      if observation.first &&& 1 = 0 ∧ observation.current = observation.first then
        .written (SeqLock.publish observation.first) data else .retry

theorem exactAttemptContract_proven : ExactAttemptContract := by
  intro bigEndian key data observation exactBound
  by_cases fresh : observation.current = observation.first <;>
    simp [attempt, exactBound, fresh, claimWrite, SeqLock.claim_success_iff]

def OtherAttemptContract : Prop :=
  ∀ bigEndian key data obs, data.bound ≠ boundByte .exact →
    let sample := Snapshot.read bigEndian obs.first obs.last obs.sample
    (attempt bigEndian key data obs = .rejected ↔
      ∃ slot, sample = some slot ∧ Reject key (slot.player, slot.opponent) data (unpack bigEndian slot.metadata)) ∧
    ∀ seq entry, attempt bigEndian key data obs = .written seq entry ↔
      ∃ slot, sample = some slot ∧ obs.current = slot.sequence ∧ SeqLock.publish slot.sequence = seq ∧
        proposal key (slot.player, slot.opponent) data (unpack bigEndian slot.metadata) = some entry

theorem otherAttemptContract_proven : OtherAttemptContract := by
  intro bigEndian key data obs notExact
  dsimp only
  rw [attempt, ite_eq_right notExact]
  cases snapshot : Snapshot.read bigEndian obs.first obs.last obs.sample with
  | none => simp
  | some slot =>
    let stored := unpack bigEndian slot.metadata
    let storedKey := (slot.player, slot.opponent)
    change (_ ↔ ∃ observed, some slot = some observed ∧ _) ∧ _
    have rejected : proposal key storedKey data stored = none ↔ Reject key storedKey data stored := by
      rw [decision]
      split <;> simp_all
    cases proposed : proposal key storedKey data stored with
    | none => simp [stored, storedKey, proposed, (rejected.mp proposed)]
    | some write =>
      have allowed : ¬ Reject key storedKey data stored := by
        intro refuse
        rw [rejected.mpr refuse] at proposed
        contradiction
      by_cases fresh : obs.current = slot.sequence <;>
        simp [stored, storedKey, proposed, allowed, claimWrite, SeqLock.claim_success_iff, fresh]

theorem attemptProposal {bigEndian : Bool} {key : Key} {data entry : EntryData}
    {observed : Observation} {seq : SeqLock.Word}
    (saved : attempt bigEndian key data observed = .written seq entry) :
    ∃ previousKey previous, proposal key previousKey data previous = some entry := by
  by_cases exactBound : data.bound = boundByte .exact
  · rw [exactAttemptContract_proven _ _ _ _ exactBound] at saved
    split at saved
    · cases saved
      exact ⟨key, data, by simp [proposal, exactBound]⟩
    · contradiction
  · obtain ⟨slot, _, _, _, written⟩ :=
      ((otherAttemptContract_proven _ _ _ _ exactBound).2 seq entry).mp saved
    exact ⟨(slot.player, slot.opponent), unpack bigEndian slot.metadata, written⟩

def run (bigEndian : Bool) (key : Key) (data : EntryData) (observations : List Observation) : Option Observation :=
  (observations.take 4).find? fun obs => attempt bigEndian key data obs != .retry

def RetryContract : Prop :=
  ∀ bigEndian key data observations,
    (run bigEndian key data observations = none ↔
      ∀ obs ∈ observations.take 4, attempt bigEndian key data obs = .retry) ∧
    ∀ selected, run bigEndian key data observations = some selected ↔
      ∃ before after, observations.take 4 = before ++ selected :: after ∧
        (∀ obs ∈ before, attempt bigEndian key data obs = .retry) ∧
        attempt bigEndian key data selected ≠ .retry

theorem retryContract_proven : RetryContract := by
  intro bigEndian key data observations
  refine ⟨by simp [run, List.find?_eq_none], ?_⟩
  intro selected
  rw [run, List.find?_eq_some_iff_append]
  constructor
  · rintro ⟨hit, before, after, shape, earlier⟩
    exact ⟨before, after, shape, by simpa using earlier, by simpa using hit⟩
  · rintro ⟨before, after, shape, earlier, hit⟩
    exact ⟨by simpa using hit, before, after, shape, by simpa using earlier⟩

def FreshWriteContract : Prop :=
  ∀ bigEndian key data obs seq entry observedIndex currentIndex,
    obs.first = SeqLock.seqWord observedIndex → obs.current = SeqLock.seqWord currentIndex →
    observedIndex ≤ currentIndex → currentIndex < observedIndex + SeqLock.modulus →
    attempt bigEndian key data obs = .written seq entry → observedIndex = currentIndex

theorem freshWriteContract_proven : FreshWriteContract := by
  intro bigEndian key data obs seq entry observedIndex currentIndex observed current ordered noWrap saved
  have same : obs.current = obs.first := by
    by_cases exactBound : data.bound = boundByte .exact
    · rw [exactAttemptContract_proven _ _ _ _ exactBound] at saved
      split at saved
      · rename_i success
        exact success.2
      · cases saved
    · obtain ⟨slot, read, claimed, _, _⟩ := (otherAttemptContract_proven _ _ _ _ exactBound).2 _ _ |>.mp saved
      have description := (Snapshot.readContract_proven _ _ _ _ _).mp read
      exact claimed.trans description.2.2.2.1
  rw [observed, current] at same
  exact SeqLock.seqWord_local_injective _ _ ordered noWrap same.symm

private def sampleEntry (move : BitVec 8) : EntryData := ⟨0, move, 1, 10, 0, 0, 0⟩

example :
    let first : Observation := ⟨2, 2, ⟨7, 8, pack false (sampleEntry 10)⟩, 4⟩
    let second : Observation := ⟨4, 4, ⟨7, 8, pack false (sampleEntry 20)⟩, 4⟩
    let data := sampleEntry noMove
    (run false (7, 8) data [first, second]).map (attempt false (7, 8) data) =
      some (.written 6 (sampleEntry 20)) := by decide

end NeuralReversi.TranspositionTable.Save

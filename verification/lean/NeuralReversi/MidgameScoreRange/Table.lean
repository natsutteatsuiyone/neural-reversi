import NeuralReversi.MidgameScoreRange.Domain
import NeuralReversi.EvalCache.Layout
import NeuralReversi.PvsParallel.Cutoffs

namespace NeuralReversi.MidgameScoreRange

open MidgameAspiration (FiniteScore)
open TranspositionTable (EntryData Score)
open TranspositionTable.Packing (newData pack unpack penalized)
open PvsParallel.Cutoffs (History Lookup)

def FiniteHistory (history : History) : Prop :=
  ∀ b entry, (b, entry) ∈ history → FiniteScore entry.score.toInt

def slotEntry (entries : List EntryData) (slot : Nat) : Except String EntryData :=
  entries[slot]?.elim (.error "TT slot out of bounds") Except.ok

def SlotContract : Prop :=
  ∀ (entries : List EntryData) (slot : Nat),
    (slotEntry entries slot = .error "TT slot out of bounds" ↔ entries.length ≤ slot) ∧
    (∀ entry, slotEntry entries slot = .ok entry ↔ entries[slot]? = some entry)

theorem slotContract_proven : SlotContract := by
  intro entries slot
  cases h : entries[slot]? with
  | none =>
    have outside := List.getElem?_eq_none_iff.mp h
    simp [slotEntry, outside]
  | some found =>
    have inside := (List.getElem?_eq_some_iff.mp h).choose
    simp [slotEntry, h, show ¬ entries.length ≤ slot by omega]

def checkedStore (entries : List EntryData) (slot : Nat) (key previousKey : TranspositionTable.Save.Key)
    (score : Score) (move : Option (Fin 64)) (bound : TranspositionTable.Bound)
    (depth : TranspositionTable.Depth) (sel : TranspositionTable.Selectivity) (gen : BitVec 8)
    (endgame pv : Bool) : Except String (Option EntryData) := do
  let previous ← slotEntry entries slot
  if isFinite score.raw then
    return TranspositionTable.Save.proposal key previousKey
      (newData score move bound depth sel gen endgame pv) previous
  return none

def StoreContract : Prop :=
  ∀ entries slot key previousKey score move bound depth sel gen endgame pv,
    let observed := checkedStore entries slot key previousKey score move bound depth sel gen endgame pv
    (entries.length ≤ slot → observed = .error "TT slot out of bounds") ∧
    (slot < entries.length → ¬ FiniteScore score.raw.toInt → observed = .ok none) ∧
    (∀ written, observed = .ok (some written) →
      FiniteScore score.raw.toInt ∧ written.score.toInt = score.raw.toInt ∧
        ∀ bigEndian, FiniteScore (unpack bigEndian (pack bigEndian written)).score.toInt)

theorem storeContract_proven : StoreContract := by
  intro entries slot key previousKey score move bound depth sel gen endgame pv
  have checked := slotContract_proven entries slot
  dsimp
  refine ⟨fun outside => ?_, fun inside bad => ?_, fun written found => ?_⟩
  · simp [checkedStore, checked.1.mpr outside, bind, Except.bind]
  · have invalid : isFinite score.raw = false := by
      cases gate : isFinite score.raw
      · rfl
      · exact (bad ((gateContract_proven _).mp gate)).elim
    have selected := (checked.2 _).mpr (List.getElem?_eq_getElem inside)
    simp [checkedStore, selected, invalid, bind, Except.bind, pure, Except.pure]
  · cases address : slotEntry entries slot with
    | error err => simp [checkedStore, address, bind, Except.bind] at found
    | ok previous =>
      cases gate : isFinite score.raw with
      | false => simp [checkedStore, address, gate, bind, Except.bind, pure, Except.pure] at found
      | true =>
        have finite := (gateContract_proven _).mp gate
        let incoming := newData score move bound depth sel gen endgame pv
        have saved : TranspositionTable.Save.proposal key previousKey incoming previous = some written := by
          simpa [checkedStore, address, gate, bind, Except.bind, pure, Except.pure] using found
        have decision := (TranspositionTable.Save.contract_proven key previousKey incoming previous).1
        rw [decision] at saved
        split at saved
        · contradiction
        have equal := Option.some.inj saved
        subst written
        have preserved : (TranspositionTable.Save.payload key previousKey incoming previous).score =
            incoming.score := by
          unfold TranspositionTable.Save.payload
          split <;> rfl
        have exactScore : incoming.score.toInt = score.raw.toInt :=
          TranspositionTable.StoredBound.narrow_score score finite.1 finite.2
        refine ⟨finite, by rw [preserved, exactScore], fun bigEndian => ?_⟩
        rwa [TranspositionTable.Packing.unpack_pack, preserved, exactScore]

inductive TTRead (history : History) (board : PvsBoardTree.Position) (entry : EntryData) : Prop
  | lookup (matched : Lookup history board entry) : TTRead history board entry
  | probe (attempts : List TranspositionTable.ProbeRetry.RawAttempt) (slot : Fin 2)
      (matched : PvsParallel.Cutoffs.ProbeMatches history board attempts slot entry)
      (hit : TranspositionTable.ProbeRetry.probe
        (attempts.map (TranspositionTable.ProbeRetry.observe (PvsParallel.Cutoffs.key board))) =
          .hit slot entry) : TTRead history board entry

theorem TTRead.source (read : TTRead history board entry) : (board, entry) ∈ history := by
  cases read with
  | lookup matched => exact matched.source
  | probe attempts slot matched hit =>
    obtain ⟨attempt, present, payload, occupied⟩ :=
      TranspositionTable.ProbeRetry.contract_proven.2 _ attempts slot entry hit
    exact matched attempt present payload occupied

def ReadContract : Prop :=
  ∀ (history : History) board entry, FiniteHistory history → TTRead history board entry →
    FiniteScore entry.score.toInt ∧
    FiniteScore (penalized entry).score.toInt ∧
    (∀ checked, ∃ score, ScoreRefresh.fromRaw checked (BitVec.ofInt 32 entry.score.toInt) = .ok score ∧
      score.raw.toInt = entry.score.toInt)

theorem readContract_proven : ReadContract := by
  intro history board entry finite matched
  have source := finite board entry matched.source
  refine ⟨source, source, fun checked => ?_⟩
  refine ⟨⟨BitVec.ofInt 32 entry.score.toInt⟩, ?_, ?_⟩
  · have sign : (BitVec.ofInt 32 entry.score.toInt).toInt = entry.score.toInt := by
      rw [BitVec.toInt_ofInt, Int.bmod_eq_of_le]
      · have := source.1; omega
      · have := source.2; omega
    have noSentinel : ¬ (entry.score.toInt < -32767 ∨ entry.score.toInt > 32767) := by
      obtain ⟨lo, hi⟩ := source
      omega
    simp [ScoreRefresh.fromRaw, sign, noSentinel]
  · change (BitVec.ofInt 32 entry.score.toInt).toInt = entry.score.toInt
    rw [BitVec.toInt_ofInt, Int.bmod_eq_of_le]
    · have := source.1; omega
    · have := source.2; omega

theorem history_push {history : History} {board : PvsBoardTree.Position} {entry : EntryData}
    (finite : FiniteHistory history) (range : FiniteScore entry.score.toInt) :
    FiniteHistory (history ++ [(board, entry)]) := by
  intro b value member
  rcases List.mem_append.mp member with old | new
  · exact finite b value old
  · obtain ⟨rfl, rfl⟩ := Prod.mk.inj (List.mem_singleton.mp new)
    exact range

inductive Published (history : History) : EntryData → Prop
  | cleared : Published history ⟨0, 0, 0, 0, 0, 0, 0⟩
  | stored {entries slot key previousKey score move bound depth sel gen endgame pv entry}
      (accepted : checkedStore entries slot key previousKey score move bound depth sel gen endgame pv =
        .ok (some entry)) : Published history entry
  | penalty {board prior} (found : (board, prior) ∈ history) : Published history (penalized prior)

inductive Produced : History → Prop
  | empty : Produced []
  | append {history board entry} (prior : Produced history) (published : Published history entry) :
      Produced (history ++ [(board, entry)])

def HistoryContract : Prop := ∀ history, Produced history → FiniteHistory history

theorem historyContract_proven : HistoryContract := by
  intro history produced
  induction produced with
  | empty => simp [FiniteHistory]
  | append _ published earlier =>
    apply history_push earlier
    cases published with
    | cleared => exact finite_zero
    | stored accepted =>
      obtain ⟨finite, equal, _⟩ := (storeContract_proven ..).2.2 _ accepted
      rwa [equal]
    | penalty cached =>
      have prior := earlier _ _ cached
      exact prior

inductive EvalPublished : BitVec 64 → Prop
  | empty : EvalPublished EvalCache.emptyWord
  | written (key : BitVec 64) (score : BitVec 32)
      (clamped : -16383 ≤ score.toInt ∧ score.toInt ≤ 16383) :
      EvalPublished (EvalCache.pack key score)

def EvalCacheContract : Prop :=
  ∀ key word score, EvalPublished word → EvalCache.read key word = some score →
    -16383 ≤ score ∧ score ≤ 16383

theorem evalCacheContract_proven : EvalCacheContract := by
  intro key word score published hit
  cases published with
  | empty =>
    rw [EvalCache.read_empty] at hit
    cases hit
  | written original source clamped =>
    have valid : EvalCache.Admitted source.toInt := by
      unfold EvalCache.Admitted
      omega
    rw [((EvalCache.encodingContract_proven original source).2.1 valid).2 key] at hit
    split at hit
    · cases hit
      exact clamped
    · cases hit

example : slotEntry [] 0 = .error "TT slot out of bounds" := by rfl

example : (match checkedStore [⟨0, 0, 0, 0, 0, 0, 0⟩] 0 (0, 0) (0, 0)
    ⟨32767⟩ none .exact 12 ⟨5, by decide⟩ 0 false false with
    | .ok none => true
    | _ => false) = true := by decide

end NeuralReversi.MidgameScoreRange

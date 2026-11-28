module
public import NeuralReversi.Threading.SplitShutdown

@[expose] public section

namespace NeuralReversi.Threading.SearchAbort

open AtomicWord (Event Write Read ReadWindow SequencedBefore)
open CutoffVisibility (History HappensBefore)
open SplitCompletion (RetainedThrough)

def generation (word : BitVec 64) : BitVec 32 :=
  word.extractLsb' 32 32

def requested (word : BitVec 64) : BitVec 32 :=
  word.extractLsb' 0 32

def highMask : BitVec 64 := BitVec.allOnes 32 ++ (0 : BitVec 32)

def lowMask : BitVec 64 := (0 : BitVec 32) ++ BitVec.allOnes 32

def begin (word : BitVec 64) : BitVec 64 :=
  (generation word + 1) ++ generation word

def request (word : BitVec 64) : BitVec 64 :=
  (word &&& highMask) ||| (word >>> (32 : Nat))

def requestFor (word : BitVec 64) (target : BitVec 32) : Option (BitVec 64) :=
  if generation word = target then some ((word &&& highMask) ||| ((0 : BitVec 32) ++ target))
  else none

def Stopped (word : BitVec 64) : Prop :=
  (word >>> (32 : Nat)) = word &&& lowMask

instance (word : BitVec 64) : Decidable (Stopped word) :=
  inferInstanceAs (Decidable (_ = _))

private theorem halves (word : BitVec 64) : generation word ++ requested word = word :=
  BitVec.extractLsb'_append_extractLsb'

private theorem high_word (word : BitVec 64) :
    word >>> (32 : Nat) = (0 : BitVec 32) ++ generation word := by
  rw [BitVec.ushiftRight_eq_extractLsb'_of_lt (by decide)]
  rfl

private theorem packed_high_mask (epoch marker : BitVec 32) :
    (epoch ++ marker) &&& highMask = epoch ++ (0 : BitVec 32) := by
  change ((epoch ++ marker) &&& (BitVec.allOnes 32 ++ 0#32)) = (epoch ++ 0#32)
  rw [BitVec.and_append, BitVec.and_allOnes, BitVec.and_zero]

private theorem packed_low_mask (epoch marker : BitVec 32) :
    (epoch ++ marker) &&& lowMask = (0 : BitVec 32) ++ marker := by
  change ((epoch ++ marker) &&& (0#32 ++ BitVec.allOnes 32)) = (0#32 ++ marker)
  rw [BitVec.and_append, BitVec.and_zero, BitVec.and_allOnes]

private theorem packed_request (epoch marker : BitVec 32) :
    request (epoch ++ marker) = epoch ++ epoch := by
  unfold request
  rw [packed_high_mask, high_word, generation, BitVec.extractLsb'_append_eq_left,
    BitVec.or_append]
  change ((epoch ||| 0#32) ++ (0#32 ||| epoch)) = epoch ++ epoch
  rw [BitVec.or_zero, BitVec.zero_or]

private theorem packed_stopped (epoch marker : BitVec 32) :
    Stopped (epoch ++ marker) ↔ epoch = marker := by
  unfold Stopped
  rw [high_word, packed_low_mask, generation, BitVec.extractLsb'_append_eq_left]
  constructor
  · intro same
    have low := congrArg requested same
    unfold requested at low
    rwa [BitVec.extractLsb'_append_eq_right, BitVec.extractLsb'_append_eq_right] at low
  · intro same
    simp [same]

theorem stopped_iff (word : BitVec 64) : Stopped word ↔ generation word = requested word :=
  calc
    Stopped word ↔ Stopped (generation word ++ requested word) := by rw [halves]
    _ ↔ _ := packed_stopped (generation word) (requested word)

theorem request_stops (word : BitVec 64) : Stopped (request word) := by
  rw [← halves word, packed_request]
  exact (packed_stopped _ _).2 rfl

def GenerationWordContract : Prop :=
  ∀ word : BitVec 64,
    generation (begin word) = generation word + 1 ∧ requested (begin word) = generation word ∧
    generation (request word) = generation word ∧ Stopped (request word) ∧
    (∀ target, if generation word = target then requestFor word target = some (request word)
      else requestFor word target = none) ∧
    ¬ Stopped (begin word)

theorem generationWordContract_proven : GenerationWordContract := by
  intro word
  have stops : request word = generation word ++ generation word := calc
    request word = request (generation word ++ requested word) := congrArg request (halves word).symm
    _ = _ := packed_request (generation word) (requested word)
  refine ⟨BitVec.extractLsb'_append_eq_left, BitVec.extractLsb'_append_eq_right, ?_,
    request_stops word, ?_, ?_⟩
  · rw [stops]
    exact BitVec.extractLsb'_append_eq_left
  · intro target
    by_cases same : generation word = target
    · simp [requestFor, request, high_word, ← same]
    · simp [requestFor, same]
  · rw [begin, packed_stopped]
    intro same
    exact (by decide : (1 : BitVec 32) ≠ 0) (BitVec.add_right_eq_self.mp same)

structure FailedSplitCheck (history : History) {writes : List Write} (read : Read writes)
    (panicAbort : Fin writes.length) (last : Nat) : Prop where
  retained : ∃ wait : Event, RetainedThrough history (writes.get panicAbort).event wait ∧
    SequencedBefore wait read.event
  beforeNext : ∀ i : Fin writes.length, i.val = last + 1 →
    HappensBefore history read.event (writes.get i).event
  writeRead : ∀ i : Fin writes.length, HappensBefore history (writes.get i).event read.event →
    i.val ≤ read.source.val
  readWrite : ∀ i : Fin writes.length, HappensBefore history read.event (writes.get i).event →
    read.source.val < i.val
  requests : ∀ i : Fin writes.length, panicAbort.val ≤ i.val → i.val ≤ last →
    ∃ previous : Fin writes.length, previous.val + 1 = i.val ∧
      (writes.get i).value = request (writes.get previous).value

def ObserveFailedSplitContract : Prop :=
  ∀ (history : History) (writes : List Write) (read : Read writes)
    (panicAbort : Fin writes.length) (last : Nat),
    FailedSplitCheck history read panicAbort last → Stopped (writes.get read.source).value

theorem observeFailedSplitContract_proven : ObserveFailedSplitContract := by
  intro history writes read panicAbort last check
  obtain ⟨wait, kept, tested⟩ := check.retained
  have visible := SplitCompletion.retainedAccessContract_proven history
    (writes.get panicAbort).event wait read.event kept tested
  have window : ReadWindow read (HappensBefore history) panicAbort last :=
    ⟨visible, check.beforeNext, check.writeRead, check.readWrite⟩
  obtain ⟨prior, _, request⟩ := check.requests read.source window.bounds.1 window.bounds.2
  rw [request]
  exact request_stops (writes.get prior).value

def ResultBoundaryContract : Prop :=
  ∀ (history : History) (messages : List Write) (sent : Fin messages.length)
    (received : Read messages) (check wait nextBegin : Event),
    RetainedThrough history check wait →
    SequencedBefore wait (messages.get sent).event →
    SplitShutdown.MessageRead history sent received →
    HappensBefore history received.event nextBegin →
    HappensBefore history check nextBegin

theorem resultBoundaryContract_proven : ResultBoundaryContract := by
  intro history messages sent received check wait nextBegin retained beforeSend matchResult next
  have order := SplitCompletion.retainedAccessContract_proven history check wait
    (messages.get sent).event retained beforeSend
  have message := (SplitShutdown.messageHandoffContract_proven history messages sent received matchResult).2
  exact .trans order (.trans message next)

def checkedResult (word : BitVec 64) (candidate : α) : Option α :=
  if Stopped word then none else some candidate

def FailedSplitPublicationContract : Prop :=
  ∀ (history : History) (writes : List Write) (ownerRead rootRead : Read writes)
    (panicAbort : Fin writes.length) (last : Nat) (Score Snapshot : Type)
    (score : Score) (candidate : Snapshot) (previous : Option Snapshot),
    FailedSplitCheck history ownerRead panicAbort last →
    FailedSplitCheck history rootRead panicAbort last →
    checkedResult (writes.get ownerRead.source).value score = none ∧
    (checkedResult (writes.get rootRead.source).value candidate).or previous = previous

theorem failedSplitPublicationContract_proven : FailedSplitPublicationContract := by
  intro history writes ownerRead rootRead panicAbort last Score Snapshot score candidate previous owner root
  have reject := observeFailedSplitContract_proven history writes ownerRead panicAbort last owner
  have preserve := observeFailedSplitContract_proven history writes rootRead panicAbort last root
  refine ⟨ite_eq_left reject, ?_⟩
  unfold checkedResult
  rw [ite_eq_left preserve]
  rfl

end NeuralReversi.Threading.SearchAbort

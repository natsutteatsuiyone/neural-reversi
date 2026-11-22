import NeuralReversi.EvalCache.Probe

namespace NeuralReversi.EvalCache

theorem self_read (key : Key) (score : BitVec 32) (ok : Admitted score.toInt) :
    read key (pack key score) = some score.toInt := by
  have finds := ((encodingContract_proven key score).2.1 ok).2 key
  simpa using finds

def replaceWay (entries : Bucket) (way : Way) (word : Word) : Bucket :=
  fun i => if i = way then word else entries i

def InsertVisibleContract : Prop :=
  ∀ key entries replacement way score,
    select key entries replacement = .miss way → Admitted (BitVec.toInt score) →
    select key (replaceWay entries way (pack key score)) replacement = .hit way score.toInt

theorem insertVisibleContract_proven : InsertVisibleContract := by
  intro key entries replacement way score missed admitted
  have before := probeContract_proven key entries replacement
  rw [missed] at before
  let updated := replaceWay entries way (pack key score)
  have inserted : read key (updated way) = some score.toInt := by
    simpa [updated, replaceWay] using self_read key score admitted
  have after := probeContract_proven key updated replacement
  cases result : select key updated replacement with
  | miss chosen =>
    rw [result] at after
    change (∀ i : Way, read key (updated i) = none) ∧ _ at after
    have contradiction := inserted.symm.trans (after.1 way)
    cases contradiction
  | hit chosen cached =>
    rw [result] at after
    change First (hitFlags key updated) chosen ∧ read key (updated chosen) = some cached at after
    have sameWay : chosen = way := by
      by_cases same : chosen = way
      · exact same
      · have prior : read key (updated chosen) = none := by
          change (∀ i : Way, read key (entries i) = none) ∧ _ at before
          simpa [updated, replaceWay, same] using before.1 chosen
        have contradiction := after.2.symm.trans prior
        cases contradiction
    subst chosen
    exact congrArg (ProbeResult.hit way) (Option.some.inj (after.2.symm.trans inserted))

abbrev Memory := Nat → Bucket

def storeAt (memory : Memory) (index : Nat) (way : Way) (word : Word) : Memory :=
  fun i => if i = index then replaceWay (memory i) way word else memory i

def clear : Memory := fun _ _ => emptyWord

def access (sizeLog2 : Nat) (memory : Memory) (key : Key) (replacement : Way)
    (evaluate : Unit → BitVec 32) : Option Int × Memory :=
  let index := bucket sizeLog2 key
  match select key (memory index) replacement with
  | .hit _ score => (some score, memory)
  | .miss way =>
    let score := evaluate ()
    match encode key score with
    | none => (none, memory)
    | some data => (some score.toInt, storeAt memory index way data)

def AccessContract : Prop :=
  ∀ sizeLog2 memory key replacement,
    let index := bucket sizeLog2 key
    let result := select key (memory index) replacement
    (∀ way score, result = .hit way score → ∀ evaluate,
      access sizeLog2 memory key replacement evaluate = (some score, memory)) ∧
    (∀ way, result = .miss way → ∀ evaluate,
      let score := evaluate ()
      let observed := access sizeLog2 memory key replacement evaluate
      (¬ Admitted score.toInt → observed = (none, memory)) ∧
      (Admitted score.toInt → observed.1 = some score.toInt ∧
        observed.2 index way = pack key score ∧
        ∀ i j, (i, j) ≠ (index, way) → observed.2 i j = memory i j))

theorem accessContract_proven : AccessContract := by
  intro sizeLog2 memory key replacement
  dsimp
  constructor
  · intro way score hit evaluate
    simp [access, hit]
  · intro way missed evaluate
    constructor
    · intro refused
      simp [access, missed, encode, refused]
    · intro admitted
      simp only [access, missed, encode, admitted, ite_true]
      refine ⟨by trivial, by simp [storeAt, replaceWay], fun i j different => ?_⟩
      by_cases same : i = bucket sizeLog2 key
      · have way : j ≠ way := fun h => different (Prod.ext same h)
        simp [storeAt, replaceWay, same, way]
      · simp [storeAt, same]

def ClearContract : Prop :=
  ∀ (key : Key) (replacement : Way) (index : Nat),
    select key (clear index) replacement = .miss 0

theorem clearContract_proven : ClearContract := by
  intro key replacement index
  have result := probeContract_proven key (clear index) replacement
  cases chosen : select key (clear index) replacement with
  | hit way score =>
    rw [chosen] at result
    have hit : read key emptyWord = some score := result.2
    rw [read_empty] at hit
    contradiction
  | miss way =>
    rw [chosen] at result
    rcases result.2 with empty | ⟨full, _⟩
    · apply congrArg ProbeResult.miss
      apply Fin.ext
      by_cases zero : way.val = 0
      · exact zero
      · have first := empty.2 (0 : Way) (Nat.pos_of_ne_zero zero)
        simp [emptyFlags, clear] at first
    · exact (full 0 rfl).elim

abbrev Publication := Nat × Way × Word

def WrittenFrom (sizeLog2 : Nat) (produces : Key → BitVec 32 → Prop)
    (entry : Publication) : Prop :=
  entry.2.2 = emptyWord ∨
    ∃ key score, produces key score ∧ bucket sizeLog2 key = entry.1 ∧
      Admitted score.toInt ∧ entry.2.2 = pack key score

def WriteContract : Prop :=
  ∀ sizeLog2 produces key score word way index,
    WrittenFrom sizeLog2 produces (index, way, emptyWord) ∧
    (produces key score → encode key score = some word →
      WrittenFrom sizeLog2 produces (bucket sizeLog2 key, way, word))

theorem writeContract_proven : WriteContract := by
  intro sizeLog2 produces key score word way index
  refine ⟨Or.inl rfl, fun value stored => Or.inr ?_⟩
  unfold encode at stored
  split at stored
  · rename_i valid
    exact ⟨key, score, value, rfl, valid, (Option.some.inj stored).symm⟩
  · cases stored

def AtomicReads (index : Nat) (history : List Publication) (observed : Bucket) : Prop :=
  ∀ way, (index, way, observed way) ∈ history

def AtomicHitContract : Prop :=
  ∀ sizeLog2 produces history observed key replacement way score,
    (∀ publication ∈ history, WrittenFrom sizeLog2 produces publication) →
    AtomicReads (bucket sizeLog2 key) history observed →
    select key observed replacement = .hit way score →
    ∃ original source, produces original source ∧
      (bucket sizeLog2 key, way, pack original source) ∈ history ∧
      score = source.toInt ∧ (18 ≤ sizeLog2 → original = key)

theorem atomicHitContract_proven : AtomicHitContract := by
  intro sizeLog2 produces history observed key replacement way score published reads hit
  have first := probeContract_proven key observed replacement
  rw [hit] at first
  change First (hitFlags key observed) way ∧ read key (observed way) = some score at first
  have source := published _ (reads way)
  rcases source with empty | ⟨original, raw, produced, sameBucket, admitted, word⟩
  · change observed way = emptyWord at empty
    have proved := first.2
    rw [empty, read_empty] at proved
    cases proved
  · have read := first.2
    change observed way = pack original raw at word
    rw [word, ((encodingContract_proven original raw).2.1 admitted).2 key] at read
    split at read
    · rename_i sameTag
      have located := (keyContract_proven sizeLog2 original key)
      exact ⟨original, raw, produced, word ▸ reads way, (Option.some.inj read).symm,
        fun enough => (located enough).1 ⟨sameBucket, sameTag.symm⟩⟩
    · cases read

def DeterministicHitContract : Prop :=
  ∀ sizeLog2 (evaluate : Key → BitVec 32) history observed key replacement way score,
    18 ≤ sizeLog2 →
    (∀ publication ∈ history, WrittenFrom sizeLog2 (fun k v => v = evaluate k) publication) →
    AtomicReads (bucket sizeLog2 key) history observed →
    select key observed replacement = .hit way score →
    score = (evaluate key).toInt

theorem deterministicHitContract_proven : DeterministicHitContract := by
  intro sizeLog2 evaluate history observed key replacement way score big valid reads hit
  obtain ⟨_, _, produced, _, equal, keyEqual⟩ :=
    atomicHitContract_proven sizeLog2 _ history observed key replacement way score valid reads hit
  rw [equal, produced, keyEqual big]

end NeuralReversi.EvalCache

module
public import NeuralReversi.Eval.SmallNetwork.Layout

@[expose] public section

namespace NeuralReversi.Eval.WeightStream

def inputFields (width : Nat) : List Nat := [2 * width, 2 * width * InputLayer.totalRows]
def linearFields (outputs paddedInputs : Nat) : List Nat :=
  [4 * outputs, outputs * paddedInputs]
def outputFields (paddedInputs : Nat) : List Nat := [4, 2 * paddedInputs]
def stackFields : List Nat := linearFields 16 288 ++ linearFields 64 32 ++ outputFields 320

def mainFields : List Nat :=
  inputFields 256 ++ (List.replicate 6 (inputFields 128)).flatten ++
    (List.replicate 60 stackFields).flatten

def smallFields : List Nat :=
  (List.replicate 3 (inputFields 128)).flatten ++ (List.replicate 30 (outputFields 128)).flatten

def phaseStart (phase : Nat) : Nat :=
  (inputFields 256).sum + phase * (inputFields 128).sum

def stackStart (ply : Nat) : Nat := phaseStart 6 + ply * stackFields.sum

private theorem repeated_sum (n : Nat) (fields : List Nat) :
    (List.replicate n fields).flatten.sum = n * fields.sum := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.replicate_succ, List.flatten_cons, List.sum_append, ih, Nat.succ_mul, Nat.add_comm]

def MainLayoutContract : Prop :=
  mainFields.sum = 609599984 ∧
    (inputFields 256).sum = phaseStart 0 ∧
    stackStart 60 = mainFields.sum ∧
    (∀ phase, phase < 6 →
      phaseStart phase + (inputFields 128).sum = phaseStart (phase + 1) ∧
      phaseStart (phase + 1) ≤ stackStart 0) ∧
    (∀ ply, ply < 60 →
      stackStart ply + stackFields.sum = stackStart (ply + 1) ∧
      stackStart (ply + 1) ≤ mainFields.sum)

theorem mainLayoutContract_proven : MainLayoutContract := by
  have size : mainFields.sum = 609599984 := by
    simp only [mainFields, List.sum_append, repeated_sum]
    rfl
  refine ⟨size, rfl, size.symm, ?_, ?_⟩
  · intro phase valid
    change 152285696 + phase * 76142848 + 76142848 =
      152285696 + (phase + 1) * 76142848 ∧
      152285696 + (phase + 1) * 76142848 ≤ 609142784
    omega
  · intro ply valid
    rw [size]
    change 609142784 + ply * 7620 + 7620 = 609142784 + (ply + 1) * 7620 ∧
      609142784 + (ply + 1) * 7620 ≤ 609599984
    omega

def SmallLayoutContract : Prop := smallFields.sum = SmallNetwork.modelBytes

theorem smallLayoutContract_proven : SmallLayoutContract := by
  unfold SmallLayoutContract
  simp only [smallFields, List.sum_append, repeated_sum]
  rfl

def readFields : List Nat → Nat → Option Nat
  | [], remaining => some remaining
  | field :: rest, remaining =>
    if field ≤ remaining then readFields rest (remaining - field) else none

def FieldConsumptionContract : Prop :=
  ∀ fields remaining leftover,
    readFields fields remaining = some leftover ↔ remaining = fields.sum + leftover

theorem fieldConsumptionContract_proven : FieldConsumptionContract := by
  intro fields
  induction fields with
  | nil => simp [readFields]
  | cons field rest ih =>
    intro remaining leftover
    simp only [readFields, List.sum_cons]
    split
    · rw [ih]
      omega
    · simp
      omega

def readSlices {α : Type} : List Nat → List α → Option (List (List α) × List α)
  | [], bytes => some ([], bytes)
  | width :: widths, bytes =>
    if width ≤ bytes.length then
      (readSlices widths (bytes.drop width)).map fun (fields, tail) =>
        (bytes.take width :: fields, tail)
    else none

def SliceContract : Prop :=
  ∀ (α : Type) widths (bytes : List α) groups tail,
    readSlices widths bytes = some (groups, tail) ↔
      groups.map List.length = widths ∧ groups.flatten ++ tail = bytes

theorem sliceContract_proven : SliceContract := by
  intro α widths
  induction widths with
  | nil =>
    intro bytes groups tail
    cases groups <;> simp [readSlices, eq_comm]
  | cons width widths ih =>
    intro bytes groups tail
    constructor
    · intro read
      unfold readSlices at read
      split at read
      · rename_i fits
        cases parts : readSlices widths (bytes.drop width) with
        | none => simp [parts] at read
        | some result =>
          obtain ⟨rest, extra⟩ := result
          simp only [parts, Option.map_some, Option.some.injEq, Prod.mk.injEq] at read
          obtain ⟨rfl, rfl⟩ := read
          obtain ⟨lengths, contents⟩ := (ih _ _ _).mp parts
          constructor
          · simp [List.length_take, Nat.min_eq_left fits, lengths]
          · simpa only [List.flatten_cons, List.append_assoc, contents] using
              List.take_append_drop width bytes
      · simp at read
    · rintro ⟨lengths, rfl⟩
      cases groups with
      | nil => simp at lengths
      | cons first rest =>
        simp only [List.map_cons, List.cons.injEq] at lengths
        obtain ⟨rfl, lengths⟩ := lengths
        have readRest : readSlices widths (rest.flatten ++ tail) = some (rest, tail) :=
          (ih _ _ _).mpr ⟨lengths, rfl⟩
        simp [readSlices, List.flatten_cons, List.append_assoc, readRest]

def readAcrossChunks {α : Type} : List (List α) → Nat → Option (List α × List (List α))
  | [], size => if size = 0 then some ([], []) else none
  | chunk :: chunks, size =>
    if size ≤ chunk.length then some (chunk.take size, chunk.drop size :: chunks)
    else (readAcrossChunks chunks (size - chunk.length)).map fun (head, tail) =>
      (chunk ++ head, tail)

def ChunkContract : Prop :=
  ∀ (α : Type) (chunks : List (List α)) size,
    (readAcrossChunks chunks size).map (fun (head, tail) => (head, tail.flatten)) =
      if size ≤ chunks.flatten.length then
        some (chunks.flatten.take size, chunks.flatten.drop size)
      else none

theorem chunkContract_proven : ChunkContract := by
  intro α chunks
  induction chunks with
  | nil =>
    intro size
    simp [readAcrossChunks]
  | cons chunk chunks ih =>
    intro size
    have bridge := ih (size - chunk.length)
    simp only [readAcrossChunks, List.flatten_cons]
    by_cases fits : size ≤ chunk.length
    · have total : size ≤ (chunk ++ chunks.flatten).length := by
        simp only [List.length_append]
        omega
      rw [ite_eq_left fits, ite_eq_left total, List.take_append_of_le_length fits,
        List.drop_append_of_le_length fits]
      rfl
    · have nextFits : (size ≤ (chunk ++ chunks.flatten).length) ↔
          (size - chunk.length ≤ chunks.flatten.length) := by
        simp only [List.length_append]
        omega
      simp only [ite_eq_right fits, nextFits, List.take_append, List.drop_append,
        List.take_of_length_le (by omega : chunk.length ≤ size),
        List.drop_of_length_le (by omega : chunk.length ≤ size), List.nil_append]
      by_cases enough : size - chunk.length ≤ chunks.flatten.length
      · rw [ite_eq_left enough] at bridge ⊢
        cases h : readAcrossChunks chunks (size - chunk.length) with
        | none => simp [h] at bridge
        | some data =>
          obtain ⟨head, tail⟩ := data
          simp only [h, Option.map_some, Option.some.injEq, Prod.mk.injEq] at bridge
          simp [bridge.1, bridge.2]
      · rw [ite_eq_right enough] at bridge ⊢
        cases h : readAcrossChunks chunks (size - chunk.length) with
        | none => rfl
        | some _ => simp [h] at bridge

def littleNat : List (BitVec 8) → Nat
  | [] => 0
  | byte :: rest => byte.toNat + 256 * littleNat rest

def littleWord (bytes : List (BitVec 8)) : BitVec (8 * bytes.length) :=
  BitVec.ofNat _ (littleNat bytes)

private theorem littleNat_bound (bytes : List (BitVec 8)) :
    littleNat bytes < 256 ^ bytes.length := by
  induction bytes with
  | nil => decide
  | cons byte rest ih =>
    have byteBound := byte.isLt
    simp only [littleNat, List.length_cons, Nat.pow_succ]
    omega

private theorem littleNat_byte (bytes : List (BitVec 8)) (i : Nat) (valid : i < bytes.length) :
    littleNat bytes / 256 ^ i % 256 = (bytes[i]'valid).toNat := by
  induction bytes generalizing i with
  | nil => simp at valid
  | cons byte rest ih =>
    have byteBound := byte.isLt
    cases i with
    | zero => simp [littleNat]; omega
    | succ i =>
      simp only [littleNat, List.getElem_cons_succ]
      rw [Nat.pow_succ', ← Nat.div_div_eq_div_mul,
        Nat.add_mul_div_left _ _ (by decide)]
      have tiny : byte.toNat / 256 = 0 := Nat.div_eq_of_lt byteBound
      -- Split off the low byte before shifting by the remaining byte count.
      simpa [tiny] using ih i (by simpa using Nat.lt_of_succ_lt_succ valid)

def LittleEndianContract : Prop :=
  ∀ (bytes : List (BitVec 8)),
    (littleWord bytes).toNat = littleNat bytes ∧
    ∀ (i : Nat) (valid : i < bytes.length),
      (littleWord bytes).toNat / 256 ^ i % 256 = (bytes[i]'valid).toNat

theorem littleEndianContract_proven : LittleEndianContract := by
  intro bytes
  have range : littleNat bytes < 2 ^ (8 * bytes.length) := by
    rw [Nat.pow_mul]
    exact littleNat_bound bytes
  have exactWord : (littleWord bytes).toNat = littleNat bytes := by
    simp [littleWord, Nat.mod_eq_of_lt range]
  exact ⟨exactWord, fun i valid => exactWord.symm ▸ littleNat_byte bytes i valid⟩

inductive ReadError where
  | unexpectedEof
  | invalidData
  | permissionDenied
  | other
  deriving DecidableEq

inductive Probe where
  | eof
  | byte
  | failed (error : ReadError)
  | interrupted (next : Probe)

def checkEnd : Probe → Except ReadError Unit
  | .eof => .ok ()
  | .byte => .error .invalidData
  | .failed error => .error error
  | .interrupted next => checkEnd next

def interrupt : Nat → Probe → Probe
  | 0, probe => probe
  | n + 1, probe => .interrupted (interrupt n probe)

def EofProbeContract : Prop :=
  ∀ retries,
    checkEnd (interrupt retries .eof) = .ok () ∧
    checkEnd (interrupt retries .byte) = .error .invalidData ∧
    (∀ error, checkEnd (interrupt retries (.failed error)) = .error error)

theorem eofProbeContract_proven : EofProbeContract := by
  intro retries
  induction retries with
  | zero => exact ⟨rfl, rfl, fun _ => rfl⟩
  | succ _ ih => exact ih

def decoderEnd (terminal : Option ReadError) : Probe :=
  match terminal with
  | none => .eof
  | some error => .failed error

def load (fields : List Nat) (bytes retries : Nat)
    (terminal : Option ReadError) : Except ReadError Unit :=
  match readFields fields bytes with
  | none => .error (terminal.getD .unexpectedEof)
  | some leftover =>
    checkEnd (interrupt retries (if leftover = 0 then decoderEnd terminal else .byte))

def LoadContract : Prop :=
  ∀ fields bytes retries terminal,
    (load fields bytes retries terminal = .ok () ↔ bytes = fields.sum ∧ terminal = none) ∧
    (bytes < fields.sum → load fields bytes retries terminal = .error (terminal.getD .unexpectedEof)) ∧
    (fields.sum < bytes → load fields bytes retries terminal = .error .invalidData) ∧
    (∀ error, load fields fields.sum retries (some error) = .error error)

theorem loadContract_proven : LoadContract := by
  intro fields bytes retries terminal
  obtain ⟨eof, byte, failed⟩ := eofProbeContract_proven retries
  have exactRead : readFields fields fields.sum = some 0 :=
    (fieldConsumptionContract_proven _ _ _).mpr (Nat.add_zero _).symm
  refine ⟨?_, ?_, ?_, ?_⟩
  · rcases Nat.lt_trichotomy bytes fields.sum with short | size | long
    · have noRead : readFields fields bytes = none := by
        cases h : readFields fields bytes with
        | none => rfl
        | some left =>
          have := (fieldConsumptionContract_proven _ _ _).mp h
          omega
      simp [load, noRead, Nat.ne_of_lt short]
    · subst bytes
      cases terminal <;> simp [load, exactRead, decoderEnd, eof, failed]
    · have hasRead : readFields fields bytes = some (bytes - fields.sum) :=
        (fieldConsumptionContract_proven _ _ _).mpr (by omega)
      simp [load, hasRead, show bytes - fields.sum ≠ 0 by omega, byte, Nat.ne_of_gt long]
  · intro short
    cases h : readFields fields bytes with
    | none => simp [load, h]
    | some _ =>
      have := (fieldConsumptionContract_proven _ _ _).mp h
      omega
  · intro long
    have hasRead : readFields fields bytes = some (bytes - fields.sum) :=
      (fieldConsumptionContract_proven _ _ _).mpr (by omega)
    simp [load, hasRead, show bytes - fields.sum ≠ 0 by omega, byte]
  · intro error
    simp [load, exactRead, decoderEnd, failed]

def consumed (fields : List Nat) (bytes : Nat) : Nat :=
  match readFields fields bytes with
  | none => bytes
  | some leftover => fields.sum + min leftover 1

def ConsumedContract : Prop :=
  ∀ fields bytes, consumed fields bytes = min bytes (fields.sum + 1)

theorem consumedContract_proven : ConsumedContract := by
  intro fields bytes
  unfold consumed
  split
  · rename_i noRead
    by_cases fits : fields.sum ≤ bytes
    · have read := (fieldConsumptionContract_proven fields bytes (bytes - fields.sum)).mpr (by omega)
      simp [noRead] at read
    · omega
  · rename_i leftover read
    have size := (fieldConsumptionContract_proven fields bytes leftover).mp read
    omega

end NeuralReversi.Eval.WeightStream

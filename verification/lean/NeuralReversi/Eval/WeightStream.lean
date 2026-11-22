import NeuralReversi.Eval.SmallNetwork.Layout

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

inductive ReadError where
  | unexpectedEof
  | invalidData
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

end NeuralReversi.Eval.WeightStream

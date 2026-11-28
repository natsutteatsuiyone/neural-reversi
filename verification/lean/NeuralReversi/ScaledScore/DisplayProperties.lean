module
public import NeuralReversi.ScaledScore.DisplayModel
public import NeuralReversi.ScaledScore.FloatProperties

@[expose] public section

namespace NeuralReversi.ScaledScore

inductive DecimalEncoding : Nat → List Char → Prop where
  | digit (d : Nat) (hd : d < 10) : DecimalEncoding d [Char.ofNat (48 + d)]
  | append (n d : Nat) (hn : 0 < n) (hd : d < 10) (chars : List Char)
      (prior : DecimalEncoding n chars) :
      DecimalEncoding (10 * n + d) (chars ++ [Char.ofNat (48 + d)])

theorem decimal_digit (n : Nat) (h : n < 10) : n.digitChar = Char.ofNat (48 + n) := by
  rw [← Nat.toNat_digitChar_of_lt_ten h]
  exact (Char.ofNat_toNat _).symm

theorem repr_decimal (n : Nat) : DecimalEncoding n n.repr.toList := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    by_cases small : n < 10
    · rw [Nat.repr_of_lt small, decimal_digit n small]
      simpa only [String.toList_singleton] using DecimalEncoding.digit n small
    · have large : 10 ≤ n := by omega
      have positive : 0 < n / 10 := (Nat.div_pos_iff).mpr ⟨by decide, large⟩
      have decrease : n / 10 < n := Nat.div_lt_self (by omega) (by decide)
      have digit : n % 10 < 10 := Nat.mod_lt n (by decide)
      have decomp : 10 * (n / 10) + n % 10 = n := by omega
      rw [Nat.repr_of_ge large]
      have result := DecimalEncoding.append (n / 10) (n % 10) positive digit
        (n / 10).repr.toList (ih (n / 10) decrease)
      simpa only [decomp, String.toList_append, String.toList_singleton,
        decimal_digit (n % 10) digit] using result

theorem twoDigits_chars (n : Nat) (h : n < 100) :
    (twoDigits n).toList = [Char.ofNat (48 + n / 10), Char.ofNat (48 + n % 10)] := by
  unfold twoDigits
  split <;> rename_i hn
  · rw [Nat.repr_of_lt hn]
    have div : n / 10 = 0 := by omega
    have mod : n % 10 = n := Nat.mod_eq_of_lt hn
    simp [div, mod, decimal_digit n hn]
  · have large : 10 ≤ n := by omega
    have div : n / 10 < 10 := by omega
    rw [Nat.repr_of_ge large, Nat.repr_of_lt div]
    simp [decimal_digit (n / 10) div, decimal_digit (n % 10) (Nat.mod_lt n (by decide))]

theorem magnitudeChunks_text (cents : Nat) :
    (String.join (magnitudeChunks cents)).toList =
      (cents / 100).repr.toList ++ ['.'] ++ (twoDigits (cents % 100)).toList := by
  unfold magnitudeChunks
  split <;> rename_i hc
  · subst cents; simp [twoDigits, String.join_cons, String.join_nil]
  · split <;> rename_i h10
    · have div : cents / 100 = 0 := by omega
      have mod : cents % 100 = cents := Nat.mod_eq_of_lt (by omega)
      simp [div, mod, twoDigits, h10, String.join_cons, String.join_nil]
    · split <;> rename_i h100
      · have div : cents / 100 = 0 := by omega
        have mod : cents % 100 = cents := Nat.mod_eq_of_lt h100
        simp [div, mod, twoDigits, h10, String.join_cons, String.join_nil]
      · simp [String.join_cons, String.join_nil, List.append_assoc]

def FixedDecimalText (raw : Word) (cents : Nat) (text : List Char) : Prop :=
  ∃ integral : List Char, DecimalEncoding (cents / 100) integral ∧
    text = (if raw.toInt < 0 then ['-'] else []) ++ integral ++
      ['.', Char.ofNat (48 + cents % 100 / 10), Char.ofNat (48 + cents % 10)]

theorem displayChunks_text (raw : Word) :
    FixedDecimalText raw (displayCents raw) (String.join (displayChunks raw)).toList := by
  refine ⟨(displayCents raw / 100).repr.toList, repr_decimal _, ?_⟩
  have mod : displayCents raw % 100 % 10 = displayCents raw % 10 := by omega
  have digits := twoDigits_chars (displayCents raw % 100) (Nat.mod_lt _ (by decide))
  unfold displayChunks
  rw [String.join_append, String.toList_append, magnitudeChunks_text, digits, mod]
  split <;> simp [String.join_cons, String.join_nil]

def PrecisionTwoChunks (cents : Nat) (chunks : List String) : Prop :=
  if cents = 0 then chunks = ["0.", "00"]
  else if cents < 10 then ∃ digit : String,
    DecimalEncoding cents digit.toList ∧ chunks = ["0.", "0", digit]
  else if cents < 100 then ∃ digits : String,
    DecimalEncoding cents digits.toList ∧ chunks = ["0.", digits]
  else ∃ whole fraction : String,
    DecimalEncoding (cents / 100) whole.toList ∧
      fraction.toList = [Char.ofNat (48 + cents % 100 / 10), Char.ofNat (48 + cents % 10)] ∧
      chunks = [whole, ".", fraction]

theorem magnitudeChunks_spec (cents : Nat) : PrecisionTwoChunks cents (magnitudeChunks cents) := by
  unfold PrecisionTwoChunks magnitudeChunks
  split
  · rfl
  · split
    · exact ⟨cents.repr, repr_decimal _, rfl⟩
    · split
      · exact ⟨cents.repr, repr_decimal _, rfl⟩
      · refine ⟨(cents / 100).repr, twoDigits (cents % 100), repr_decimal _, ?_, rfl⟩
        have mod : cents % 100 % 10 = cents % 10 := by omega
        simpa only [mod] using twoDigits_chars (cents % 100) (Nat.mod_lt _ (by decide))

def DisplayChunkContract : Prop :=
  ∀ raw : Word, ∃ chunks : List String, PrecisionTwoChunks (displayCents raw) chunks ∧
    displayChunks raw = (if raw.toInt < 0 then ["-"] else []) ++ chunks

theorem displayChunkContract_proven : DisplayChunkContract := by
  intro raw
  exact ⟨magnitudeChunks (displayCents raw), magnitudeChunks_spec _, rfl⟩

inductive WriteExecution {State Payload : Type}
    (write : State → String → WriteResult State Payload) :
    State → List String → WriteResult State Payload → Prop where
  | done (state : State) : WriteExecution write state [] (.ok state)
  | next (state next : State) (chunk : String) (rest : List String)
      (result : WriteResult State Payload) (step : write state chunk = .ok next)
      (tail : WriteExecution write next rest result) :
      WriteExecution write state (chunk :: rest) result
  | error (state next : State) (chunk : String) (rest : List String)
      (step : write state chunk = .error next) :
      WriteExecution write state (chunk :: rest) (.error next)
  | panic (state next : State) (chunk : String) (rest : List String) (payload : Payload)
      (step : write state chunk = .panic payload next) :
      WriteExecution write state (chunk :: rest) (.panic payload next)

theorem writeChunks_execution {State Payload : Type}
    (write : State → String → WriteResult State Payload) (chunks : List String) :
    ∀ state, WriteExecution write state chunks (writeChunks write state chunks) := by
  induction chunks with
  | nil => intro state; exact .done state
  | cons chunk rest ih =>
    intro state
    simp only [writeChunks]
    split <;> rename_i next step
    · exact .next _ _ _ _ _ step (ih next)
    · exact .error _ _ _ _ step
    · exact .panic _ _ _ _ _ step

theorem writeExecution_unique {State Payload : Type}
    {write : State → String → WriteResult State Payload} {state : State}
    {chunks : List String} {result : WriteResult State Payload}
    (execution : WriteExecution write state chunks result) :
    result = writeChunks write state chunks := by
  induction execution with
  | done => rfl
  | next _ _ _ _ _ step _ ih => simp only [writeChunks, step]; exact ih
  | error _ _ _ _ step => simp only [writeChunks, step]
  | panic _ _ _ _ _ step => simp only [writeChunks, step]

def DisplayTextContract : Prop :=
  ∀ raw : Word, ∃ rounded : Nat,
    RepresentsConversionResult raw rounded ∧ NearestEvenMagnitude raw.toInt.natAbs rounded ∧
      (∀ candidate : Int,
        ((rounded : Int) * 25 - (displayCents raw : Int) * 64).natAbs ≤
          ((rounded : Int) * 25 - candidate * 64).natAbs) ∧
      (2 * ((rounded : Int) * 25 - (displayCents raw : Int) * 64).natAbs = 64 →
        displayCents raw % 2 = 0) ∧
      FixedDecimalText raw (displayCents raw) (String.join (displayChunks raw)).toList

theorem displayTextContract_proven : DisplayTextContract := by
  intro raw
  let rounded := roundedMagnitude raw.toInt.natAbs
  refine ⟨rounded, model_represents_rounded raw, roundedMagnitude_nearest _, ?_, ?_,
    displayChunks_text raw⟩
  · simpa [displayCents, distance, rounded] using round_nearest (rounded * 25) 6
  · intro tie
    apply round_tie_even_from_error (rounded * 25) 6
    simpa [displayCents, distance, rounded] using tie

theorem displayCents_half_error (raw : Word) :
    2 * ((roundedMagnitude raw.toInt.natAbs : Int) * 25 -
      (displayCents raw : Int) * 64).natAbs ≤ 64 := by
  simpa [displayCents, distance] using rounding_half_error (roundedMagnitude raw.toInt.natAbs * 25) 6

theorem cents_exact_integer (whole : Nat) :
    roundEvenQuotient (whole * 256 * 25) 6 = whole * 100 := by
  have hmul : whole * 256 * 25 = whole * 100 * 64 := by omega
  simp only [roundEvenQuotient]
  rw [hmul]
  simp

def DisplayWriterContract : Prop :=
  ∀ (State Payload : Type) (write : State → String → WriteResult State Payload)
    (state : State) (raw : Word) (options : FormatOptions),
    (display raw options write state).2 = options ∧
      ∀ result, (display raw options write state).1 = result ↔
        WriteExecution write state (displayChunks raw) result

theorem displayWriterContract_proven : DisplayWriterContract := by
  intro State Payload write state raw options
  refine ⟨rfl, ?_⟩
  intro result
  constructor
  · intro h; rw [← h]; exact writeChunks_execution write _ state
  · intro h; exact (writeExecution_unique h).symm

end NeuralReversi.ScaledScore

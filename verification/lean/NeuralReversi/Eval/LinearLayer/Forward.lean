module
public import NeuralReversi.Eval.LinearLayer.Layout
public import NeuralReversi.Eval.LinearLayer.Dot

@[expose] public section

namespace NeuralReversi.Eval.LinearLayer

def biasPrefix {α : Type} (count : Nat) (biases : List α) : List α :=
  biases.take count

def BiasPrefixContract : Prop :=
  ∀ (α : Type) (count : Nat) (biases : List α), count ≤ biases.length →
    (biasPrefix count biases).length = count ∧
      ∀ i, i < count → (biasPrefix count biases)[i]? = biases[i]?

theorem biasPrefixContract_proven : BiasPrefixContract := by
  intro α count biases fits
  unfold biasPrefix
  exact ⟨by rw [List.length_take, Nat.min_eq_left fits],
    fun _ hi => by rw [List.getElem?_take, ite_eq_left hi]⟩

def loadedWeight (inputs chunks outputs : Nat) (disk : Nat → Int) (slot : Nat) : Int :=
  let chunk := slot / 4 / outputs
  let row := slot / 4 % outputs
  let lane := slot % 4
  if 4 * chunk + lane < inputs then disk (diskSlot chunks row chunk lane) else 0

private theorem loadedPackedWeight (inputs chunks outputs : Nat) (disk : Nat → Int)
    (chunk row lane : Nat) (rowOk : row < outputs) (laneOk : lane < 4) :
    loadedWeight inputs chunks outputs disk (packedSlot outputs chunk row lane) =
      if 4 * chunk + lane < inputs then disk (diskSlot chunks row chunk lane) else 0 := by
  obtain ⟨hc, hr, hl⟩ := packedSlotsContract_proven.2 outputs chunk row lane rowOk laneOk
  unfold loadedWeight
  rw [hc, hr, hl]

private theorem diskRowColumn (chunks row chunk lane : Nat) :
    diskSlot chunks row chunk lane = row * (4 * chunks) + (4 * chunk + lane) := by
  simp only [diskSlot, Nat.add_mul, Nat.mul_assoc]
  rw [Nat.mul_comm 4 chunks, Nat.mul_comm 4 chunk, Nat.add_assoc]

def LoadedWeightsContract : Prop :=
  ∀ inputs chunks outputs disk chunk row lane,
    chunk < chunks → row < outputs → lane < 4 →
    let raw := diskSlot chunks row chunk lane
    let slot := rustWeightIndex raw (4 * chunks) outputs
    loadedWeight inputs chunks outputs disk slot =
      if raw % (4 * chunks) < inputs then disk raw else 0

theorem loadedWeightsContract_proven : LoadedWeightsContract := by
  intro inputs chunks outputs disk chunk row lane chunkOk rowOk laneOk
  dsimp
  rw [weightLayoutContract_proven chunks outputs row chunk lane chunkOk laneOk,
    loadedPackedWeight inputs chunks outputs disk chunk row lane rowOk laneOk]
  have col : diskSlot chunks row chunk lane % (4 * chunks) = 4 * chunk + lane := by
    have h : 4 * chunk + lane < 4 * chunks := by omega
    rw [diskRowColumn, Nat.mul_comm row (4 * chunks), Nat.mul_add_mod, Nat.mod_eq_of_lt h]
  rw [col]

inductive DotKernel where
  | mixed
  | x86Emulated
  | sdot
  | widened

def kernelDot : DotKernel → (Nat → Int) → (Nat → Int) → Int
  | .mixed => byteDot
  | .x86Emulated => x86EmulatedDot
  | .sdot => sdotEmulatedDot
  | .widened => widenedDot

def sumPrefix (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumPrefix f n + f n

private theorem sumPrefix_zeroTail (f : Nat → Int) (n m : Nat)
    (hle : n ≤ m) (hz : ∀ i, n ≤ i → f i = 0) :
    sumPrefix f m = sumPrefix f n := by
  induction m with
  | zero =>
    have : n = 0 := by omega
    rw [this]
  | succ m ih =>
    by_cases heq : n = m + 1
    · rw [heq]
    · rw [sumPrefix, ih (by omega), hz m (by omega), Int.add_zero]

private theorem sumPrefix_congr (n : Nat) (f g : Nat → Int)
    (h : ∀ i, i < n → f i = g i) : sumPrefix f n = sumPrefix g n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [sumPrefix, sumPrefix, ih (fun i hi => h i (by omega)), h n (by omega)]

private theorem chunkedSum (f : Nat → Int) (chunks : Nat) :
    sumPrefix (fun chunk => byteDot (fun lane => f (4 * chunk + lane)) (fun _ => 1)) chunks =
      sumPrefix f (4 * chunks) := by
  induction chunks with
  | zero => rfl
  | succ chunks ih =>
    rw [sumPrefix, ih]
    have h : 4 * (chunks + 1) = 4 * chunks + 1 + 1 + 1 + 1 := by omega
    rw [h, sumPrefix, sumPrefix, sumPrefix, sumPrefix]
    unfold byteDot
    simp only [Int.mul_one]
    rw [show 4 * chunks + 0 = 4 * chunks by omega,
      show 4 * chunks + 2 = 4 * chunks + 1 + 1 by omega,
      show 4 * chunks + 3 = 4 * chunks + 1 + 1 + 1 by omega]
    omega

def simdForward (kernel : DotKernel) (inputs chunks outputs processed row : Nat)
    (bias : Int) (input disk : Nat → Int) : BitVec 32 :=
  BitVec.ofInt 32 (bias + sumPrefix (fun chunk =>
    kernelDot kernel (fun lane => input (4 * chunk + lane))
      (fun lane => loadedWeight inputs chunks outputs disk (packedSlot outputs chunk row lane)))
    processed)

def scalarForward (inputs chunks row : Nat) (bias : Int) (input disk : Nat → Int) : BitVec 32 :=
  BitVec.ofInt 32 (bias + sumPrefix (fun i => input i * disk (row * (4 * chunks) + i)) inputs)

def DenseForwardContract : Prop :=
  ∀ kernel inputs chunks outputs processed row bias (input disk : Nat → Int),
    row < outputs → processed ≤ chunks → inputs ≤ 4 * processed →
    (∀ i, i < 4 * processed → 0 ≤ input i ∧ input i ≤ 255) →
    (∀ i, i < 4 * chunks * outputs → -128 ≤ disk i ∧ disk i ≤ 127) →
    simdForward kernel inputs chunks outputs processed row bias input disk =
      scalarForward inputs chunks row bias input disk

theorem denseForwardContract_proven : DenseForwardContract := by
  intro kernel inputs chunks outputs processed row bias input disk rowOk procOk inOk ha hb
  let paddedTerm (i : Nat) :=
    if i < inputs then input i * disk (row * (4 * chunks) + i) else 0
  have paddedTerm_def (i : Nat) :
      paddedTerm i = if i < inputs then input i * disk (row * (4 * chunks) + i) else 0 := rfl
  unfold simdForward scalarForward
  congr 2
  calc
    _ = sumPrefix
        (fun chunk => byteDot (fun lane => paddedTerm (4 * chunk + lane)) (fun _ => 1))
        processed := by
      apply sumPrefix_congr
      intro chunk hc
      let w (lane : Nat) :=
        loadedWeight inputs chunks outputs disk (packedSlot outputs chunk row lane)
      have weight (lane : Nat) (hl : lane < 4) :
          w lane = if 4 * chunk + lane < inputs then disk (diskSlot chunks row chunk lane) else 0 := by
        exact loadedPackedWeight inputs chunks outputs disk chunk row lane rowOk hl
      have inputOk lane (hl : lane < 4) := ha (4 * chunk + lane) (by omega)
      have weightOk (lane : Nat) (hl : lane < 4) : -128 ≤ w lane ∧ w lane ≤ 127 := by
        rw [weight lane hl]
        split
        · apply hb
          rw [diskRowColumn]
          have h := Nat.mul_le_mul_right (4 * chunks) (show row + 1 ≤ outputs by omega)
          rw [Nat.add_one_mul, Nat.mul_comm outputs] at h
          omega
        · decide
      have hw := mixedDotContract_proven _ _ inputOk weightOk
      have hkernel : kernelDot kernel (fun lane => input (4 * chunk + lane)) w =
          byteDot (fun lane => input (4 * chunk + lane)) w := by
        cases kernel
        · rfl
        · exact hw.1
        · exact hw.2.1
        · exact hw.2.2.1
      have term (lane : Nat) (hl : lane < 4) :
          input (4 * chunk + lane) * w lane = paddedTerm (4 * chunk + lane) := by
        rw [weight lane hl, diskRowColumn, paddedTerm_def]
        split <;> simp_all
      change kernelDot kernel _ w = _
      rw [hkernel]
      unfold byteDot
      simp only [Int.mul_one, term 0 (by decide), term 1 (by decide), term 2 (by decide),
        term 3 (by decide)]
    _ = sumPrefix paddedTerm (4 * processed) := chunkedSum paddedTerm processed
    _ = sumPrefix paddedTerm inputs :=
      sumPrefix_zeroTail paddedTerm inputs (4 * processed) inOk (fun _ hi => ite_eq_right (by omega))
    _ = _ := sumPrefix_congr inputs _ _ (fun _ hi => ite_eq_left hi)

abbrev Word := BitVec 32

def reduce : List Word → Word
  | [] => 0
  | word :: words => word + reduce words

private theorem reduce_modify (words : List Word) (lane : Nat) (delta : Word)
    (inBounds : lane < words.length) :
    reduce (words.modify lane (· + delta)) = reduce words + delta := by
  induction words generalizing lane with
  | nil => simp at inBounds
  | cons word words ih =>
    cases lane with
    | zero =>
      simp [List.modify, reduce, BitVec.add_assoc, BitVec.add_comm delta]
    | succ lane =>
      have h : lane < words.length := by simpa using inBounds
      change word + reduce (words.modify lane (· + delta)) = (word + reduce words) + delta
      rw [ih lane h, BitVec.add_assoc]

def partialLane (processed unroll chunk : Nat) : Nat :=
  if chunk < processed / unroll * unroll then chunk % unroll else 0

def partials (processed unroll : Nat) (delta : Nat → Word) : Nat → List Word
  | 0 => List.replicate unroll 0
  | chunk + 1 => (partials processed unroll delta chunk).modify
      (partialLane processed unroll chunk) (· + delta chunk)

private theorem partials_length (processed unroll : Nat) (delta : Nat → Word) (n : Nat) :
    (partials processed unroll delta n).length = unroll := by
  induction n with
  | zero => simp [partials]
  | succ n ih => simp [partials, List.length_modify, ih]

private theorem reduce_zeros (n : Nat) : reduce (List.replicate n 0) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [List.replicate_succ, reduce, ih]
    rfl

def PartialReductionContract : Prop :=
  ∀ processed unroll (delta : Nat → Int) (bias : Int),
    0 < unroll →
    BitVec.ofInt 32 bias +
        reduce (partials processed unroll (fun i => BitVec.ofInt 32 (delta i)) processed) =
      BitVec.ofInt 32 (bias + sumPrefix delta processed)

theorem partialReductionContract_proven : PartialReductionContract := by
  intro processed unroll delta bias positive
  have laneBound (chunk : Nat) : partialLane processed unroll chunk < unroll := by
    unfold partialLane
    split
    · exact Nat.mod_lt _ positive
    · exact positive
  have sums (n : Nat) :
      reduce (partials processed unroll (fun i => BitVec.ofInt 32 (delta i)) n) =
        BitVec.ofInt 32 (sumPrefix delta n) := by
    induction n with
    | zero =>
      rw [partials, reduce_zeros, sumPrefix]
      rfl
    | succ n ih =>
      rw [partials, reduce_modify _ _ _ (by simpa only [partials_length] using laneBound n),
        ih, sumPrefix, BitVec.ofInt_add]
  rw [sums, BitVec.ofInt_add]

def SdotBlockContract : Prop :=
  ∀ (low high : Nat → Int) (count : Nat) (acc : Int),
    BitVec.ofInt 32 (acc + sumPrefix low count) - BitVec.ofInt 32 (sumPrefix high count) =
      BitVec.ofInt 32 (acc + sumPrefix (fun i => low i - high i) count)

theorem sdotBlockContract_proven : SdotBlockContract := by
  intro low high count acc
  have sums : sumPrefix (fun i => low i - high i) count =
      sumPrefix low count - sumPrefix high count := by
    induction count with
    | zero => rfl
    | succ count ih =>
      rw [sumPrefix, sumPrefix, sumPrefix, ih]
      omega
  rw [BitVec.sub_eq_add_neg, ← BitVec.ofInt_neg, ← BitVec.ofInt_add, sums]
  congr 1
  omega

end NeuralReversi.Eval.LinearLayer

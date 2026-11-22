import NeuralReversi.MoveList.Initialization

namespace NeuralReversi.MoveList.Retain
open Initialization

def copySlot (mem : Memory α) (dst src : Nat) : Memory α :=
  fun i => if i = dst then mem src else mem i

theorem copySlot_eq_write (mem : Memory α) (dst src : Nat) (v : α)
    (readable : mem src = some v) : copySlot mem dst src = write mem dst v := by
  funext i
  simp [copySlot, write, readable]

structure Outcome (α : Type) where
  memory : Memory α
  len : Nat

def compact (keep : Nat → Bool) : Nat → Nat → Nat → Memory α → Outcome α
  | 0, _, dst, mem => ⟨mem, dst⟩
  | n + 1, read, dst, mem =>
      if keep read then compact keep n (read + 1) (dst + 1) (copySlot mem dst read)
      else compact keep n (read + 1) dst mem

def selected (values : Nat → α) (keep : Nat → Bool) : Nat → Nat → List α
  | 0, _ => []
  | n + 1, read =>
      if keep read then values read :: selected values keep n (read + 1)
      else selected values keep n (read + 1)

def Readable (mem : Memory α) (values : Nat → α) (read n : Nat) : Prop :=
  ∀ i, read ≤ i → i < read + n → mem i = some (values i)

theorem copy_preserves_unread (mem : Memory α) (values : Nat → α)
    (read dst n : Nat) (order : dst ≤ read) (init : Readable mem values read (n + 1)) :
    Readable (copySlot mem dst read) values (read + 1) n := by
  intro i lo hi
  have ne : i ≠ dst := by omega
  simpa [copySlot, ne] using init i (by omega) (by omega)

def SafeTrace (values : Nat → α) (keep : Nat → Bool) :
    Nat → Nat → Nat → Memory α → Prop
  | 0, _, _, _ => True
  | n + 1, read, dst, mem =>
      dst ≤ read ∧ mem read = some (values read) ∧
      (if keep read then SafeTrace values keep n (read + 1) (dst + 1) (copySlot mem dst read)
       else SafeTrace values keep n (read + 1) dst mem)

theorem compact_safe (values : Nat → α) (keep : Nat → Bool) (n read dst : Nat)
    (mem : Memory α) (order : dst ≤ read) (init : Readable mem values read n) :
    SafeTrace values keep n read dst mem := by
  induction n generalizing read dst mem with
  | zero => trivial
  | succ n ih =>
    refine ⟨order, init read (by omega) (by omega), ?_⟩
    split
    · exact ih _ _ _ (by omega) (copy_preserves_unread mem values read dst n order init)
    · apply ih _ _ _ (by omega)
      intro i lo hi
      exact init i (by omega) (by omega)

theorem compact_exact (values : Nat → α) (keep : Nat → Bool) (n read dst : Nat)
    (mem : Memory α) (order : dst ≤ read) (init : Readable mem values read n) :
    let out := compact keep n read dst mem
    let expected := selected values keep n read
    out.len = dst + expected.length ∧
    (∀ j, j < expected.length → out.memory (dst + j) = expected[j]?) ∧
    (∀ i, i < dst → out.memory i = mem i) ∧
    (∀ i, read + n ≤ i → out.memory i = mem i) := by
  induction n generalizing read dst mem with
  | zero => simp [compact, selected]
  | succ n ih =>
    by_cases accept : keep read = true
    · have resultIH := ih (read + 1) (dst + 1) (copySlot mem dst read) (by omega)
        (copy_preserves_unread mem values read dst n order init)
      dsimp only at resultIH
      simp only [compact, selected, accept, ↓reduceIte, List.length_cons]
      refine ⟨by omega, ?_, ?_, ?_⟩
      · intro j hj
        cases j with
        | zero =>
          have first := resultIH.2.2.1 dst (by omega)
          have original := init read (by omega) (by omega)
          simpa [copySlot, original] using first
        | succ j =>
          have tail := resultIH.2.1 j (by omega)
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using tail
      · intro i hi
        have before := resultIH.2.2.1 i (by omega)
        have ne : i ≠ dst := by omega
        simpa [copySlot, ne] using before
      · intro i hi
        have after := resultIH.2.2.2 i (by omega)
        have ne : i ≠ dst := by omega
        simpa [copySlot, ne] using after
    · have tailInit : Readable mem values (read + 1) n := by
        intro i lo hi; exact init i (by omega) (by omega)
      have resultIH := ih (read + 1) dst mem (by omega) tailInit
      simpa [compact, selected, accept, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using resultIH

theorem selected_length_le (values : Nat → α) (keep : Nat → Bool) (n read : Nat) :
    (selected values keep n read).length ≤ n := by
  induction n generalizing read with
  | zero => simp [selected]
  | succ n ih =>
    have tail := ih (read + 1)
    by_cases h : keep read = true
    · simp only [selected, h, ↓reduceIte, List.length_cons]
      omega
    · simp [selected, h]
      exact Nat.le_trans tail (by omega)

theorem retain_contract (values : Nat → α) (keep : Nat → Bool) (n : Nat)
    (mem : Memory α) (init : Readable mem values 0 n) (capacity : n ≤ 35) :
    let out := compact keep n 0 0 mem
    SafeTrace values keep n 0 0 mem ∧
    out.len = (selected values keep n 0).length ∧ out.len ≤ n ∧ out.len ≤ 35 ∧
    (∀ i, i < out.len → out.memory i = (selected values keep n 0)[i]?) ∧
    (∀ i, n ≤ i → out.memory i = mem i) := by
  have exactness := compact_exact values keep n 0 0 mem (by omega) init
  have bound := selected_length_le values keep n 0
  dsimp only at exactness ⊢
  refine ⟨compact_safe values keep n 0 0 mem (by omega) init, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using exactness.1
  · omega
  · omega
  · intro i hi
    simpa using exactness.2.1 i (by omega)
  · simpa using exactness.2.2.2

def liveSlot (out : Outcome α) (i : Nat) : Option α :=
  if i < out.len then out.memory i else none

theorem liveSlot_exact (values : Nat → α) (keep : Nat → Bool) (n : Nat)
    (mem : Memory α) (init : Readable mem values 0 n) (i : Nat) :
    liveSlot (compact keep n 0 0 mem) i = (selected values keep n 0)[i]? := by
  have spec := compact_exact values keep n 0 0 mem (by omega) init
  dsimp only at spec
  have length : (compact keep n 0 0 mem).len = (selected values keep n 0).length := by
    simpa using spec.1
  unfold liveSlot
  rw [length]
  by_cases hi : i < (selected values keep n 0).length
  · simp only [hi, ↓reduceIte]
    simpa using spec.2.1 i hi
  · have absent : (selected values keep n 0)[i]? = none :=
      List.getElem?_eq_none_iff.mpr (by omega)
    simp only [hi, ↓reduceIte, absent]

theorem retained_prefix_initialized (values : Nat → α) (keep : Nat → Bool) (n : Nat)
    (mem : Memory α) (init : Readable mem values 0 n) (i : Nat)
    (live : i < (compact keep n 0 0 mem).len) :
    ∃ v, (compact keep n 0 0 mem).memory i = some v := by
  have spec := compact_exact values keep n 0 0 mem (by omega) init
  dsimp only at spec
  have bound : i < (selected values keep n 0).length := by omega
  refine ⟨(selected values keep n 0)[i], ?_⟩
  have exactSlot := spec.2.1 i bound
  simpa only [Nat.zero_add, List.getElem?_eq_getElem bound] using exactSlot

example : (compact (fun i => decide (i ≠ 0)) 3 0 0 (fun i => some (10 + i))).len = 2 := rfl
example : (compact (fun i => decide (i ≠ 0)) 3 0 0 (fun i => some (10 + i))).memory 0 = some 11 := rfl
example : (compact (fun i => decide (i ≠ 0)) 3 0 0 (fun i => some (10 + i))).memory 1 = some 12 := rfl
example : (compact (fun _ => false) 3 0 0 (fun i => some i)).memory 0 = some 0 := rfl
example : (compact (fun _ => false) 3 0 0 (fun i => some i)).len = 0 := rfl
example : copySlot (fun i => some i) 1 0 1 ≠ some 1 := by decide

end NeuralReversi.MoveList.Retain

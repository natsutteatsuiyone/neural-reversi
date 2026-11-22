import NeuralReversi.Bitboard.Model

namespace NeuralReversi.PrincipalVariation

abbrev Slot := Option Bitboard.Square
abbrev Buffer := List Slot
abbrev Stack := List Buffer

def capacity : Nat := 61

def visible : Buffer → List Bitboard.Square
  | some square :: tail => square :: visible tail
  | _ => []

def clear (buffer : Buffer) : Buffer := none :: buffer.drop 1

def prepare (stack : Stack) (ply : Nat) : Stack :=
  (stack.modify ply clear).modify (ply + 1) clear

def copyPrefix : Nat → Buffer → Buffer
  | fuel + 1, some square :: tail => some square :: copyPrefix fuel tail
  | _, _ => [none]

def prepend (square : Bitboard.Square) (child current : Buffer) : Buffer :=
  let written := some square :: copyPrefix (capacity - 2) child
  written ++ current.drop written.length

theorem copyPrefix_spec (fuel : Nat) (child : Buffer) :
    copyPrefix fuel child = ((visible child).take fuel).map some ++ [none] := by
  induction fuel generalizing child with
  | zero => rfl
  | succ fuel ih =>
    cases child with
    | nil => rfl
    | cons entry tail => cases entry <;> simp [copyPrefix, visible, ih]

theorem visible_written (line : List Bitboard.Square) (suffix : Buffer) :
    visible (line.map some ++ none :: suffix) = line := by
  induction line <;> simp_all [visible]

def BufferContract : Prop :=
  ∀ (square : Bitboard.Square) (child current : Buffer),
    current.length = capacity →
    let next := prepend square child current
    let continued := (visible child).take (capacity - 2)
    let line := square :: continued
    next = line.map some ++ none :: current.drop (line.length + 1) ∧
    next.length = capacity ∧
    visible next = line ∧
    (line.length < capacity ∧ next[line.length]? = some none) ∧
    ((visible child).length ≤ capacity - 2 → visible next = square :: visible child)

theorem bufferContract_proven : BufferContract := by
  intro square child current size next continued line
  have layout : next = line.map some ++ none :: current.drop (line.length + 1) := by
    simp [next, prepend, copyPrefix_spec, line, continued, Nat.add_assoc]
  have bound : line.length + 1 ≤ capacity := by
    have length := List.length_take (l := visible child) (i := capacity - 2)
    simp only [line, continued, List.length_cons]
    unfold capacity at *
    omega
  have terminator : next[line.length]? = some none := by
    rw [layout, List.getElem?_append_right (by simp)]
    simp
  refine ⟨layout, ?_, ?_, ⟨by omega, terminator⟩, ?_⟩
  · simp only [layout, List.length_append, List.length_map, List.length_cons, List.length_drop, size]
    omega
  · rw [layout, visible_written]
  · intro short
    rw [layout, visible_written]
    simp [line, continued, List.take_of_length_le short]

def PrepareContract : Prop :=
  ∀ (stack : Stack) (ply : Nat),
    (prepare stack ply).length = stack.length ∧
    (∀ i, i < stack.length → (i = ply ∨ i = ply + 1) →
      ∃ row, (prepare stack ply)[i]? = some row ∧ visible row = []) ∧
    (∀ i, i ≠ ply → i ≠ ply + 1 → (prepare stack ply)[i]? = stack[i]?)

theorem prepareContract_proven : PrepareContract := by
  intro stack ply
  refine ⟨by simp [prepare], ?_, ?_⟩
  · intro i inside cleared
    rcases cleared with same | next
    · subst i
      have justCurrent : (prepare stack ply)[ply]? = some (clear stack[ply]) := by
        simp [prepare, inside]
      exact ⟨_, justCurrent, rfl⟩
    · subst i
      have justChild : (prepare stack ply)[ply + 1]? = some (clear stack[ply + 1]) := by
        simp [prepare, inside]
      exact ⟨_, justChild, rfl⟩
  · intro i other unrelated
    simp [prepare, Ne.symm other, Ne.symm unrelated]

example : let a : Bitboard.Square := ⟨0, by decide⟩
    let child := List.replicate 61 (some a)
    (prepend a child child).length = 61 ∧ (prepend a child child)[60]? = some none := by decide

end NeuralReversi.PrincipalVariation

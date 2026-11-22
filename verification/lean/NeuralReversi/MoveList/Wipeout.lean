import NeuralReversi.MoveList.Retain

namespace NeuralReversi.MoveList.Wipeout
open Initialization Retain

structure Move where
  square : NeuralReversi.Square.Square
  flipped : Bitboard.Bits
  deriving DecidableEq, Repr

def isMatch (opponent : Bitboard.Bits) (mv : Move) : Bool := mv.flipped == opponent

def scan (opponent : Bitboard.Bits) (mem : Memory Move) : Nat → Nat → Option Move
  | 0, _ => none
  | n + 1, start =>
      match mem start with
      | some mv => if isMatch opponent mv then some mv else scan opponent mem n (start + 1)
      | none => scan opponent mem n (start + 1)

theorem scan_matches_list (opponent : Bitboard.Bits) (xs : List Move)
    (mem : Memory Move) (start : Nat)
    (initialized : ∀ i, i < xs.length → mem (start + i) = xs[i]?) :
    scan opponent mem xs.length start = xs.find? (isMatch opponent) := by
  induction xs generalizing start with
  | nil => rfl
  | cons head tail ih =>
    have first : mem start = some head := by simpa using initialized 0 (by simp)
    have tailPrefix : ∀ i, i < tail.length → mem ((start + 1) + i) = tail[i]? := by
      intro i hi
      have h := initialized (i + 1) (by simpa using hi)
      simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using h
    simp only [List.length_cons, scan, first, List.find?_cons]
    cases h : isMatch opponent head <;> simp [ih (start + 1) tailPrefix]

def rebuild (opponent : Bitboard.Bits) (keep : Nat → Bool) (n : Nat) (mem : Memory Move) :
    Option NeuralReversi.Square.Square :=
  let out := compact keep n 0 0 mem
  (scan opponent out.memory out.len 0).map Move.square

theorem compact_scan_exact (opponent : Bitboard.Bits) (values : Nat → Move)
    (keep : Nat → Bool) (n : Nat) (mem : Memory Move) (init : Readable mem values 0 n) :
    let out := compact keep n 0 0 mem
    scan opponent out.memory out.len 0 = (selected values keep n 0).find? (isMatch opponent) := by
  have spec := compact_exact values keep n 0 0 mem (by omega) init
  dsimp only at spec ⊢
  have len : (compact keep n 0 0 mem).len = (selected values keep n 0).length := by
    simpa using spec.1
  rw [len]
  apply scan_matches_list
  intro i hi
  exact spec.2.1 i hi

theorem rebuild_exact (opponent : Bitboard.Bits) (values : Nat → Move)
    (keep : Nat → Bool) (n : Nat) (mem : Memory Move) (init : Readable mem values 0 n) :
    rebuild opponent keep n mem =
      ((selected values keep n 0).find? (isMatch opponent)).map Move.square := by
  unfold rebuild
  dsimp only
  rw [compact_scan_exact opponent values keep n mem init]

theorem rebuild_none_iff (opponent : Bitboard.Bits) (values : Nat → Move)
    (keep : Nat → Bool) (n : Nat) (mem : Memory Move) (init : Readable mem values 0 n) :
    rebuild opponent keep n mem = none ↔
      ∀ mv ∈ selected values keep n 0, mv.flipped ≠ opponent := by
  rw [rebuild_exact opponent values keep n mem init, Option.map_eq_none_iff, List.find?_eq_none]
  simp [isMatch]

theorem rebuild_some_iff (opponent : Bitboard.Bits) (values : Nat → Move)
    (keep : Nat → Bool) (n : Nat) (mem : Memory Move) (init : Readable mem values 0 n)
    (sq : NeuralReversi.Square.Square) :
    rebuild opponent keep n mem = some sq ↔
      ∃ mv before after, selected values keep n 0 = before ++ mv :: after ∧
        mv.flipped = opponent ∧ mv.square = sq ∧
        ∀ previous ∈ before, previous.flipped ≠ opponent := by
  rw [rebuild_exact opponent values keep n mem init, Option.map_eq_some_iff]
  constructor
  · rintro ⟨mv, found, square⟩
    obtain ⟨hit, before, after, shape, earlier⟩ := List.find?_eq_some_iff_append.mp found
    refine ⟨mv, before, after, shape, ?_, square, ?_⟩
    · simpa [isMatch] using hit
    · intro previous member
      simpa [isMatch] using earlier previous member
  · rintro ⟨mv, before, after, shape, hit, square, earlier⟩
    refine ⟨mv, List.find?_eq_some_iff_append.mpr ?_, square⟩
    refine ⟨by simpa [isMatch] using hit, before, after, shape, ?_⟩
    intro previous member
    simpa [isMatch] using earlier previous member

def generationCache (opponent : Bitboard.Bits) : List Move →
    Option NeuralReversi.Square.Square → Option NeuralReversi.Square.Square
  | [], cache => cache
  | mv :: rest, cache =>
      generationCache opponent rest (if isMatch opponent mv then some mv.square else cache)

private def firstMove : Move := ⟨⟨0, by decide⟩, 1⟩
private def lastMove : Move := ⟨⟨1, by decide⟩, 1⟩
private def original (i : Nat) : Move := if i = 0 then firstMove else lastMove
private def originalMemory : Memory Move := fun i => some (original i)

example : generationCache 1 [firstMove, lastMove] none = some lastMove.square := rfl
example : rebuild 1 (fun _ => true) 2 originalMemory = some firstMove.square := rfl
example : rebuild 1 (fun i => decide (i = 0)) 2 originalMemory = some firstMove.square := rfl
example : rebuild 1 (fun _ => false) 2 originalMemory = none := rfl
example : (compact (fun _ => false) 2 0 0 originalMemory).memory 0 = some firstMove := rfl

end NeuralReversi.MoveList.Wipeout

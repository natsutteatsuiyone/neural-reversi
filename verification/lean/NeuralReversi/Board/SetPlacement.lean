module
public import NeuralReversi.Board.Properties

@[expose] public section

namespace NeuralReversi.Board.SetPlacement
open Bitboard

def placedMask (b : Board) (mask : Bits) : Bits := mask &&& ~~~b.opponent

def placeMask (b : Board) (flipped mask : Bits) : Board :=
  ⟨b.opponent &&& ~~~flipped, (b.player ||| placedMask b mask) ||| flipped⟩

theorem overlap_exact (b : Board) (flipped mask : Bits) :
    (placeMask b flipped mask).player &&& (placeMask b flipped mask).opponent =
      (b.player &&& b.opponent) &&& ~~~flipped := by
  ext i hi
  simp only [placeMask, placedMask, BitVec.getElem_and, BitVec.getElem_or,
    BitVec.getElem_not]
  cases hp : b.player[i] <;> cases ho : b.opponent[i] <;>
    cases hf : flipped[i] <;> cases hm : mask[i] <;> simp_all

theorem wellFormed_preserved (b : Board) (flipped mask : Bits) (wf : wellFormed b) :
    wellFormed (placeMask b flipped mask) := by
  unfold wellFormed at *
  rw [overlap_exact, wf]
  simp

theorem occupancy_exact (b : Board) (flipped mask : Bits) :
    (placeMask b flipped mask).player ||| (placeMask b flipped mask).opponent =
      ((b.player ||| b.opponent) ||| placedMask b mask) ||| flipped := by
  ext i hi
  simp only [placeMask, placedMask, BitVec.getElem_and, BitVec.getElem_or,
    BitVec.getElem_not]
  cases hp : b.player[i] <;> cases ho : b.opponent[i] <;>
    cases hf : flipped[i] <;> cases hm : mask[i] <;> simp_all

theorem occupancy_mask_exact (b : Board) (flipped mask : Bits) :
    (placeMask b flipped mask).player ||| (placeMask b flipped mask).opponent =
      ((b.player ||| b.opponent) ||| mask) ||| flipped := by
  ext i hi
  simp only [placeMask, placedMask, BitVec.getElem_and, BitVec.getElem_or,
    BitVec.getElem_not]
  cases hp : b.player[i] <;> cases ho : b.opponent[i] <;>
    cases hf : flipped[i] <;> cases hm : mask[i] <;> simp_all

theorem occupancy_monotone (b : Board) (flipped mask : Bits) :
    (b.player ||| b.opponent) &&&
      ~~~((placeMask b flipped mask).player ||| (placeMask b flipped mask).opponent) = 0 := by
  rw [occupancy_mask_exact]
  ext i hi
  simp only [BitVec.getElem_and, BitVec.getElem_or, BitVec.getElem_not]
  cases hp : b.player[i] <;> cases ho : b.opponent[i] <;>
    cases hf : flipped[i] <;> cases hm : mask[i] <;> simp_all

theorem equals_xor_of_valid (b : Board) (flipped mask : Bits)
    (wf : wellFormed b) (subset : flipped &&& ~~~b.opponent = 0)
    (empty : mask &&& (b.player ||| b.opponent) = 0) :
    placeMask b flipped mask = ⟨b.opponent ^^^ flipped, b.player ^^^ flipped ^^^ mask⟩ := by
  unfold placeMask
  congr 1
  all_goals
    ext i hi
    have hd := congrArg (fun value : Bits => value[i]) wf
    have hf := congrArg (fun value : Bits => value[i]) subset
    have he := congrArg (fun value : Bits => value[i]) empty
    simp only [wellFormed, placedMask, BitVec.getElem_and, BitVec.getElem_or,
      BitVec.getElem_not, BitVec.getElem_xor] at *
    cases hp : b.player[i] <;> cases ho : b.opponent[i] <;>
      cases hflip : flipped[i] <;> cases hm : mask[i] <;> simp_all

def place (b : Board) (flipped : Bits) (sq : Square) : Board :=
  placeMask b flipped (squareMask sq)

def placeAny (debugAssertions overflowChecks : Bool) (b : Board) (flipped : Bits)
    (sq : AnySquare) : Except String Board :=
  (squareBitboard debugAssertions overflowChecks sq).map (placeMask b flipped)

theorem placeAny_playable (debugAssertions overflowChecks : Bool) (b : Board)
    (flipped : Bits) (sq : Square) :
    placeAny debugAssertions overflowChecks b flipped sq =
      .ok (place b flipped sq) := by
  rw [placeAny, (squareMaskContract_proven sq debugAssertions overflowChecks).2.1]
  rfl

theorem placeAny_error (debugAssertions overflowChecks : Bool) (b : Board)
    (flipped : Bits) (sq : AnySquare) (reason : String)
    (failed : squareBitboard debugAssertions overflowChecks sq = .error reason) :
    placeAny debugAssertions overflowChecks b flipped sq = .error reason := by
  simp only [placeAny, failed, Except.map]

theorem placeAny_success_iff (debugAssertions overflowChecks : Bool) (b : Board)
    (flipped : Bits) (sq : AnySquare) (result : Board) :
    placeAny debugAssertions overflowChecks b flipped sq = .ok result ↔
      ∃ mask, squareBitboard debugAssertions overflowChecks sq = .ok mask ∧
        result = placeMask b flipped mask := by
  cases conversion : squareBitboard debugAssertions overflowChecks sq with
  | error reason => simp [placeAny, conversion, Except.map]
  | ok mask => simp [placeAny, conversion, Except.map, eq_comm]

theorem placeAny_wellFormed (debugAssertions overflowChecks : Bool) (b : Board)
    (flipped : Bits) (sq : AnySquare) (result : Board) (wf : wellFormed b)
    (accepted : placeAny debugAssertions overflowChecks b flipped sq = .ok result) :
    wellFormed result := by
  obtain ⟨mask, _, rfl⟩ := (placeAny_success_iff debugAssertions overflowChecks b flipped sq result).mp accepted
  exact wellFormed_preserved b flipped mask wf

example : place ⟨1, 0⟩ 0 ⟨0, by decide⟩ = ⟨0, 1⟩ := by rfl

example : place ⟨0, 1⟩ 0 ⟨0, by decide⟩ = ⟨1, 0⟩ ∧
    wellFormed (place ⟨0, 1⟩ 0 ⟨0, by decide⟩) := by
  constructor
  · rfl
  · unfold wellFormed
    decide

example : ¬ wellFormed (placeMask ⟨1, 1⟩ 0 0) := by
  unfold wellFormed
  decide

example : placeMask ⟨0, 0⟩ 3 4 = ⟨0, 7⟩ := by rfl

example : place ⟨1, 2⟩ 2 ⟨2, by decide⟩ = NeuralReversi.Board.place ⟨1, 2⟩ 2 ⟨2, by decide⟩ := by rfl

end NeuralReversi.Board.SetPlacement

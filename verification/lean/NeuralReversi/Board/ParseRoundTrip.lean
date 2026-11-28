module
public import NeuralReversi.Board.ParseProperties
public import NeuralReversi.Board.RenderingProperties

@[expose] public section

namespace NeuralReversi.Board
open Bitboard

theorem board_fields_ext (a b : Board) (player : a.player = b.player)
    (opponent : a.opponent = b.opponent) : a = b := by
  cases a
  cases b
  simp_all

def visibleBoard (b : Board) : Board := ⟨b.player, b.opponent &&& ~~~b.player⟩

def bareBoardText (b : Board) (side : Disc.PlayerColor) : String :=
  String.ofList (List.ofFn fun i : Fin 64 => Disc.toChar (cellDisc b i.val side))

def stripRowSeparators (text : String) : String :=
  String.ofList (text.toList.filter (· != '\n'))

theorem toChar_not_newline (side : Disc.Disc) : Disc.toChar side ≠ '\n' := by
  cases side <;> decide

theorem toChar_validCell (side : Disc.Disc) : validCell (Disc.toChar side) = true := by
  cases side <;> decide

theorem bareBoardText_size (b : Board) (side : Disc.PlayerColor) :
    (bareBoardText b side).toList.length = 64 := by simp [bareBoardText]

theorem bareBoardText_valid (b : Board) (side : Disc.PlayerColor) :
    ∀ c ∈ (bareBoardText b side).toList, validCell c = true := by
  intro c member
  simp only [bareBoardText, String.toList_ofList, List.mem_ofFn] at member
  obtain ⟨i, rfl⟩ := member
  exact toChar_validCell _

theorem parsed_bareBoardText (b : Board) (side : Disc.PlayerColor) :
    parsedBoard side ⟨0, 0⟩ (Square.iter.zip (bareBoardText b side).toList) = visibleBoard b := by
  apply board_fields_ext
  · apply BitVec.eq_of_getLsbD_eq
    intro i hi
    simp only [parsedBoard, BitVec.ofNat_eq_ofNat, BitVec.zero_or]
    rw [bitsFor_zip_bit _ (bareBoardText_size b side) _ ⟨i, hi⟩]
    simp only [bareBoardText, String.toList_ofList, List.getElem_ofFn, visibleBoard]
    cases side <;> cases hp : b.player.getLsbD i <;> cases ho : b.opponent.getLsbD i <;>
      simp_all [cellDisc, Disc.PlayerColor.opposite, Disc.PlayerColor.toChar, Disc.PlayerColor.toDisc, Disc.toChar]
  · apply BitVec.eq_of_getLsbD_eq
    intro i hi
    simp only [parsedBoard, BitVec.ofNat_eq_ofNat, BitVec.zero_or]
    rw [bitsFor_zip_bit _ (bareBoardText_size b side) _ ⟨i, hi⟩]
    simp only [bareBoardText, String.toList_ofList, List.getElem_ofFn, visibleBoard,
      BitVec.getLsbD_and, BitVec.getLsbD_not]
    cases side <;> cases hp : b.player.getLsbD i <;> cases ho : b.opponent.getLsbD i <;>
      simp_all [cellDisc, Disc.PlayerColor.opposite, Disc.PlayerColor.toChar, Disc.PlayerColor.toDisc, Disc.toChar]

theorem fromString_bareBoardText (checked : Bool) (b : Board) (side : Disc.PlayerColor) :
    fromString checked (bareBoardText b side) side = .ok (visibleBoard b) := by
  have size := bareBoardText_size b side
  have representable : RepresentableText (bareBoardText b side) := by
    unfold RepresentableText
    rw [size]
    decide
  have noError := (firstInvalid_none_iff _ size).mpr (bareBoardText_valid b side)
  rw [fromString_exact checked _ side representable]
  simp only [size, Nat.lt_irrefl, ↓reduceIte, noError]
  rw [parsed_bareBoardText b side]

theorem visibleBoard_wellFormed (b : Board) : wellFormed (visibleBoard b) := by
  change b.player &&& (b.opponent &&& ~~~b.player) = 0#64
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_not, BitVec.getLsbD_zero]
  cases b.player.getLsbD i <;> cases b.opponent.getLsbD i <;> simp_all

theorem visibleBoard_eq (b : Board) (valid : wellFormed b) : visibleBoard b = b := by
  apply board_fields_ext
  · rfl
  · apply BitVec.eq_of_getLsbD_eq
    intro i hi
    have apart := congrArg (fun bits : Bits => bits.getLsbD i) valid
    simp only [BitVec.getLsbD_and, BitVec.ofNat_eq_ofNat, BitVec.getLsbD_zero] at apart
    simp only [visibleBoard, BitVec.getLsbD_and, BitVec.getLsbD_not]
    cases hp : b.player.getLsbD i <;> cases ho : b.opponent.getLsbD i <;> simp_all

theorem strip_renderingSpec (b : Board) (side : Disc.PlayerColor) :
    stripRowSeparators (String.ofList (renderingSpec b side)) = bareBoardText b side := by
  simp [stripRowSeparators, renderingSpec, bareBoardText, List.ofFn_succ, List.ofFn_zero,
    toChar_not_newline]

def BoardTextRoundTripContract : Prop :=
  ∀ (checked : Bool) (b : Board) (side : Disc.PlayerColor),
    (toStringAsBoard checked b side).bind
      (fun text => (fromString checked (stripRowSeparators text) side).mapError fun _ => "parse failure") =
      .ok (visibleBoard b) ∧
    wellFormed (visibleBoard b) ∧ (wellFormed b → visibleBoard b = b)

theorem boardTextRoundTripContract_proven : BoardTextRoundTripContract := by
  intro checked b side
  refine ⟨?_, visibleBoard_wellFormed b, visibleBoard_eq b⟩
  rw [toStringAsBoard_exact]
  simp only [Except.bind, strip_renderingSpec, fromString_bareBoardText, Except.mapError]

end NeuralReversi.Board

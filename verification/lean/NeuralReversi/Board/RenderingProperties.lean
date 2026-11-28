module
public import NeuralReversi.Board.CellProperties
public import Init.Data.List.OfFn

@[expose] public section

namespace NeuralReversi.Board
open Bitboard
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

def renderingSpec (b : Board) (side : Disc.PlayerColor) : List Char :=
  List.ofFn fun (p : Fin 71) =>
    if p.val % 9 = 8 then '\n'
    else Disc.toChar (cellDisc b (p.val / 9 * 8 + p.val % 9) side)

theorem discAt_lt (checked : Bool) (b : Board) (sq : AnySquare) (side : Disc.PlayerColor) :
    discAt checked b sq side = .ok (cellDisc b sq.val side) :=
  discAt_playable checked b sq side

theorem renderChars_exact (checked : Bool) (b : Board) (side : Disc.PlayerColor) :
    renderChars checked b side 0 Square.iter = .ok (renderingSpec b side) := by
  simp [Square.iter, List.finRange, List.ofFn_succ, List.ofFn_zero, renderingSpec,
    renderChars, discAt_lt, bind, Except.bind, pure, Except.pure]
  repeat constructor

theorem toStringAsBoard_exact (checked : Bool) (b : Board) (side : Disc.PlayerColor) :
    toStringAsBoard checked b side = .ok (String.ofList (renderingSpec b side)) := by
  rw [toStringAsBoard, renderChars_exact]
  rfl

def RenderingContract : Prop :=
  ∀ (checked : Bool) (b : Board) (side : Disc.PlayerColor),
    ∃ (text : String), toStringAsBoard checked b side = .ok text ∧
      ∃ size : text.toList.length = 71,
      ∀ (p : Fin 71), text.toList[p.val]'(by rw [size]; exact p.isLt) =
        if p.val % 9 = 8 then '\n'
        else Disc.toChar (cellDisc b (p.val / 9 * 8 + p.val % 9) side)

theorem renderingContract_proven : RenderingContract := by
  intro checked b side
  refine ⟨String.ofList (renderingSpec b side), toStringAsBoard_exact checked b side, ?_, ?_⟩
  · simp [renderingSpec]
  · intro p
    simp only [String.toList_ofList, renderingSpec, List.getElem_ofFn]

theorem renderingSpec_newline_iff (b : Board) (side : Disc.PlayerColor) (p : Fin 71) :
    (renderingSpec b side)[p.val]'(by simp only [renderingSpec, List.length_ofFn]; exact p.isLt) = '\n' ↔
      p.val % 9 = 8 := by
  simp only [renderingSpec, List.getElem_ofFn]
  split
  · simp_all
  · have noNewline (d : Disc.Disc) : Disc.toChar d ≠ '\n' := by cases d <;> decide
    simp_all

theorem renderingSpec_cell (b : Board) (side : Disc.PlayerColor) (i : Fin 64) :
    ∃ bound : i.val + i.val / 8 < (renderingSpec b side).length,
      (renderingSpec b side)[i.val + i.val / 8]'bound = Disc.toChar (cellDisc b i.val side) := by
  have hi := i.isLt
  have bound : i.val + i.val / 8 < 71 := by omega
  refine ⟨by simpa only [renderingSpec, List.length_ofFn] using bound, ?_⟩
  simp only [renderingSpec, List.getElem_ofFn]
  have newline : (i.val + i.val / 8) % 9 ≠ 8 := by omega
  have source : (i.val + i.val / 8) / 9 * 8 + (i.val + i.val / 8) % 9 = i.val := by omega
  rw [ite_eq_right newline, source]

end NeuralReversi.Board

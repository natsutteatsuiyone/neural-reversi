import NeuralReversi.Stability.EdgeCapture

namespace NeuralReversi.Stability.Semantics

def encodePair (line : Line) (c : Color) : Edge.State :=
  (encodeLine line c, encodeLine line (Opposite c))

theorem encodePair_disjoint (line : Line) (c : Color) :
    (encodePair line c).1 &&& (encodePair line c).2 = 0 :=
  encodeLine_disjoint line c

theorem encodePair_swap (line : Line) (c : Color) :
    encodePair line (Opposite c) = ((encodePair line c).2, (encodePair line c).1) := by
  cases c <;> rfl

theorem encodePair_empty (line : Line) (c : Color) (a : Fin 8)
    (empty : line a = none) :
    Edge.emptyAt (encodePair line c).1 (encodePair line c).2 a = true := by
  apply Edge.bits_emptyAt
  · simp [encodePair, empty, -BitVec.getLsbD_eq_getElem]
  · simp [encodePair, empty, -BitVec.getLsbD_eq_getElem]

theorem encodePair_moveSpec (line : Line) (c : Color) (a : Fin 8)
    (empty : line a = none) :
    encodePair (linePlace line c a) c =
      Edge.moveSpec (encodePair line c).1 (encodePair line c).2 a := by
  let p := (encodePair line c).1
  let o := (encodePair line c).2
  have valid : p &&& o = 0 := encodePair_disjoint line c
  have vac : Edge.emptyAt p o a = true := encodePair_empty line c a empty
  have after := Edge.moveSpec_disjoint p o a valid vac
  have decoded := Edge.moveSpec_line p o c a valid
  have before : decodeColor p o c = line := decodeColor_encode line c
  rw [before] at decoded
  apply Prod.ext
  · change encodeLine (linePlace line c a) c = _
    rw [← decoded]
    exact encodeLine_decode_player _ _ _ after
  · change encodeLine (linePlace line c a) (Opposite c) = _
    rw [← decoded]
    exact encodeLine_decode_opponent _ _ _ after

end NeuralReversi.Stability.Semantics

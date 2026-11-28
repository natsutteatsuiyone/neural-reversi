module
public import NeuralReversi.Stability.Geometry
public import NeuralReversi.Solve1.Projection

@[expose] public section

namespace NeuralReversi.Stability

def spread : Word → List Nat → Word
  | _, [] => 0
  | p, row :: rest =>
      (if p.getLsbD (8*row) then ((255 : Word) <<< (8*row)) else 0) ||| spread p rest

set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false
private theorem dilate_finite : ∀ n : Fin 256,
    Solve1.Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val) * 255 =
      spread (Solve1.Projection.scatter [0,8,16,24,32,40,48,56] (BitVec.ofNat 8 n.val))
        [0,1,2,3,4,5,6,7] := by decide

theorem dilate_eq_spread (p : Word) :
    (p &&& 0x0101010101010101) * 255 = spread p [0,1,2,3,4,5,6,7] := by
  let cells : List Nat := [0,8,16,24,32,40,48,56]
  have hp := Solve1.Projection.masked_eq_scatter cells p (by decide)
    0x0101010101010101 (by decide)
  have hh := dilate_finite ⟨(Solve1.Projection.gather cells p).toNat,
    (Solve1.Projection.gather cells p).isLt⟩
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq] at hh
  rw [← hp] at hh
  simpa [spread, -BitVec.getLsbD_eq_getElem] using hh

theorem getLsbD_cond (b : Bool) (x y : Word) (i : Nat) :
    (if b then x else y).getLsbD i = (if b then x.getLsbD i else y.getLsbD i) := by
  cases b <;> rfl

theorem getLsbD_ite (b : Prop) [Decidable b] (x y : Word) (i : Nat) :
    (if b then x else y).getLsbD i = (if b then x.getLsbD i else y.getLsbD i) := by
  split <;> rfl

theorem full_0_0 (disc : Word) :
    (field (getFullLines disc) 0).getLsbD 0 = lineBits disc 0 0 := by
  simp only [field, getFullLines, Id.run, bind, pure]
  rw [dilate_eq_spread]
  simp [spread, getLsbD_cond, getLsbD_ite, lineBits, lineCells, sameLine, List.range_succ,
    -BitVec.getLsbD_eq_getElem, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]
end NeuralReversi.Stability

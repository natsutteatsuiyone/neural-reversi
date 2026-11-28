module
public import NeuralReversi.Flip.Avx2.Model
public import NeuralReversi.Flip.Avx2.Prepared
public import NeuralReversi.Simd.X86Properties

@[expose] public section

namespace NeuralReversi.Flip.Avx2
open NeuralReversi.Simd.X86

def literalLane (leftMask rightMask player opponent : Word) (step : Nat) : Word :=
  let x := ~~~opponent &&& leftMask
  let lo := ((x + (-1#64)) ^^^ x) &&& leftMask
  let rp := player &&& rightMask
  let first := rp ||| (rp >>> step)
  let lf := ~~~player &&& lo
  let leftFlip := ~~~(if lf = lo then BitVec.allOnes 64 else 0) &&& lf
  let second := first ||| (first >>> (2*step))
  let third := second ||| (second >>> (4*step))
  let re := (~~~opponent &&& rightMask) ^^^ rp
  let rightFlip := ~~~third &&&
    (rightMask &&& (if re.toInt < rp.toInt then BitVec.allOnes 64 else 0))
  rightFlip ||| leftFlip

theorem literalLane_eq (leftMask rightMask player opponent : Word) (step : Nat) :
    literalLane leftMask rightMask player opponent step =
      leftWord leftMask player opponent ||| rightWord rightMask player opponent step := by
  unfold literalLane
  dsimp only
  have negOne : (-1#64) = BitVec.allOnes 64 := by decide
  simp only [negOne,add_allOnes_eq_sub_one]
  rw [cmpeq_erase,cmpgt_gate,← shadow_three_stages]
  rw [BitVec.or_comm]
  rfl

theorem mmFlipPrepared_eq_derived (pp oo : Vec256) (pos : Index64)
    (valid : pos.toNat < 66) :
    mmFlipPrepared pp oo pos valid = derivedPrepared ⟨pos.toNat,valid⟩ pp oo := by
  funext j
  rcases j with ⟨j,hj⟩
  have cases : j = 0 ∨ j = 1 := by omega
  rcases cases with rfl | rfl
  · change
      literalLane (mask66 ⟨pos.toNat,valid⟩ 0) (mask66 ⟨pos.toNat,valid⟩ 4) (pp 0) (oo 0) 1 |||
        literalLane (mask66 ⟨pos.toNat,valid⟩ 2) (mask66 ⟨pos.toNat,valid⟩ 6) (pp 2) (oo 2) 9 = _
    rw [literalLane_eq,literalLane_eq]
    rfl
  · change
      literalLane (mask66 ⟨pos.toNat,valid⟩ 1) (mask66 ⟨pos.toNat,valid⟩ 5) (pp 1) (oo 1) 8 |||
        literalLane (mask66 ⟨pos.toNat,valid⟩ 3) (mask66 ⟨pos.toNat,valid⟩ 7) (pp 3) (oo 3) 7 = _
    rw [literalLane_eq,literalLane_eq]
    rfl

theorem flip_eq_summary (sq : Square) (player opponent : Word) :
    flip sq player opponent = summary sq player opponent := by
  unfold flip flip1 newContext flipPrepared
  rw [mmFlipPrepared_eq_derived]
  dsimp only
  rw [shuffleEpi32_4e]
  change
    derivedPrepared ⟨(squareIndex sq).toNat,squareIndex_valid sq⟩ (broadcast player) (broadcast opponent) 0 |||
      derivedPrepared ⟨(squareIndex sq).toNat,squareIndex_valid sq⟩ (broadcast player) (broadcast opponent) 1 = _
  have index : (⟨(squareIndex sq).toNat,squareIndex_valid sq⟩ : Index) = publicIndex sq :=
    Fin.ext (squareIndex_value sq)
  rw [index]
  exact broadcast_public_reduction sq player opponent

end NeuralReversi.Flip.Avx2

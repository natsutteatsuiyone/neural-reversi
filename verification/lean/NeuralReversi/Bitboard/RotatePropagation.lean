module
public import NeuralReversi.Bitboard.RotateGeometry
public import NeuralReversi.Simd.TernaryProperties

@[expose] public section

namespace NeuralReversi.Bitboard.Moves
open NeuralReversi.Simd.X86

def rotateExpanded (h p : Bits) (d : Nat) : Bits :=
  let first := h &&& p.rotateLeft d
  let two := ternaryWord first h (first.rotateLeft d) 0xf8
  let pre := h &&& h.rotateLeft d
  let four := ternaryWord two pre (two.rotateLeft (2*d)) 0xf8
  ternaryWord four pre (four.rotateLeft (2*d)) 0xf8

theorem rotateExpanded_left (o mask p : Bits) (d : Nat)
    (dw : d < 64) (tw : 2*d < 64) (gaps : MaskGaps mask d)
    (preGaps : ∀ i : Fin 64, i.val < 2*d →
      (mask &&& (mask <<< d)).getLsbD i.val = false) :
    rotateExpanded (o &&& mask) p d = expandedLeft (o &&& mask) p d := by
  simp only [rotateExpanded,ternaryWord_f8,
    potential_rol_eq_shl o mask d dw gaps,
    maskedOpponent_rol_eq_shl o mask _ d dw gaps,
    maskedPre_rol_eq_shl o mask _ d tw preGaps,expandedLeft]

theorem rotateExpanded_right (o mask p : Bits) (d : Nat)
    (dp : 0 < d) (dw : d < 64) (tw : 2*d < 64) (gaps : MaskGaps mask d)
    (preGaps : ∀ i : Fin 64, 64-2*d ≤ i.val →
      (mask &&& (mask >>> d)).getLsbD i.val = false) :
    (let h := o &&& mask
     let first := h &&& p.rotateLeft (64-d)
     let two := ternaryWord first h (first.rotateLeft (64-d)) 0xf8
     let pre := h &&& h.rotateLeft (64-d)
     let four := ternaryWord two pre (two.rotateLeft (64-2*d)) 0xf8
     ternaryWord four pre (four.rotateLeft (64-2*d)) 0xf8) = expandedRight (o &&& mask) p d := by
  simp only [ternaryWord_f8,
    rol_negative_eq_ror _ d dp dw,
    rol_negative_eq_ror _ (2*d) (by omega) tw,
    potential_ror_eq_shr o mask d dw gaps,
    maskedOpponent_ror_eq_shr o mask _ d dw gaps,
    maskedPre_ror_eq_shr o mask _ d tw preGaps,expandedRight]

def rotateLane (h p : Bits) (d doubled : Nat) : Bits :=
  let first := h &&& p.rotateLeft d
  let two := ternaryWord first h (first.rotateLeft d) 0xf8
  let pre := h &&& h.rotateLeft d
  let four := ternaryWord two pre (two.rotateLeft doubled) 0xf8
  let six := ternaryWord four pre (four.rotateLeft doubled) 0xf8
  six.rotateLeft d

theorem rotateLane_left (o mask p : Bits) (d : Nat)
    (dw : d < 64) (tw : 2*d < 64) (gaps : MaskGaps mask d)
    (preGaps : ∀ i : Fin 64, i.val < 2*d →
      (mask &&& (mask <<< d)).getLsbD i.val = false) :
    rotateLane (o &&& mask) p d (2*d) = expandedLeft (o &&& mask) p d <<< d := by
  change (rotateExpanded (o &&& mask) p d).rotateLeft d = _
  rw [rotateExpanded_left o mask p d dw tw gaps preGaps,moves_rol_eq_shl o mask p d dw gaps]

theorem rotateLane_right (o mask p : Bits) (d : Nat)
    (dp : 0 < d) (dw : d < 64) (tw : 2*d < 64) (gaps : MaskGaps mask d)
    (preGaps : ∀ i : Fin 64, 64-2*d ≤ i.val →
      (mask &&& (mask >>> d)).getLsbD i.val = false) :
    rotateLane (o &&& mask) p (64-d) (64-2*d) = expandedRight (o &&& mask) p d >>> d := by
  have trace := rotateExpanded_right o mask p d dp dw tw gaps preGaps
  dsimp only at trace
  dsimp only [rotateLane]
  rw [trace,
    rol_negative_eq_ror _ d dp dw,moves_ror_eq_shr o mask p d dw gaps]


end NeuralReversi.Bitboard.Moves

module
public import NeuralReversi.Bitboard.Model

@[expose] public section

namespace NeuralReversi.Bitboard.Moves

def horizontalMask : Bits := 0x7e7e7e7e7e7e7e7e

def getMovesPortable (player opponent : Bits) : Bits :=
  let hOpp := opponent &&& horizontalMask
  let flip7 := hOpp &&& (player <<< (7 : Nat))
  let flip9 := hOpp &&& (player <<< (9 : Nat))
  let flip8 := opponent &&& (player <<< (8 : Nat))
  let flip1 := hOpp &&& (player <<< (1 : Nat))
  let flip7 := flip7 ||| (hOpp &&& (flip7 <<< (7 : Nat)))
  let flip9 := flip9 ||| (hOpp &&& (flip9 <<< (9 : Nat)))
  let flip8 := flip8 ||| (opponent &&& (flip8 <<< (8 : Nat)))
  let moves := hOpp + flip1
  let pre7 := hOpp &&& (hOpp <<< (7 : Nat))
  let pre9 := hOpp &&& (hOpp <<< (9 : Nat))
  let pre8 := opponent &&& (opponent <<< (8 : Nat))
  let flip7 := flip7 ||| (pre7 &&& (flip7 <<< (14 : Nat)))
  let flip9 := flip9 ||| (pre9 &&& (flip9 <<< (18 : Nat)))
  let flip8 := flip8 ||| (pre8 &&& (flip8 <<< (16 : Nat)))
  let flip7 := flip7 ||| (pre7 &&& (flip7 <<< (14 : Nat)))
  let flip9 := flip9 ||| (pre9 &&& (flip9 <<< (18 : Nat)))
  let flip8 := flip8 ||| (pre8 &&& (flip8 <<< (16 : Nat)))
  let moves := moves ||| ((flip7 <<< (7 : Nat)) |||
    (flip9 <<< (9 : Nat)) ||| (flip8 <<< (8 : Nat)))
  let flip7 := hOpp &&& (player >>> (7 : Nat))
  let flip9 := hOpp &&& (player >>> (9 : Nat))
  let flip8 := opponent &&& (player >>> (8 : Nat))
  let flip1 := hOpp &&& (player >>> (1 : Nat))
  let flip7 := flip7 ||| (hOpp &&& (flip7 >>> (7 : Nat)))
  let flip9 := flip9 ||| (hOpp &&& (flip9 >>> (9 : Nat)))
  let flip8 := flip8 ||| (opponent &&& (flip8 >>> (8 : Nat)))
  let flip1 := flip1 ||| (hOpp &&& (flip1 >>> (1 : Nat)))
  let pre7 := pre7 >>> (7 : Nat)
  let pre9 := pre9 >>> (9 : Nat)
  let pre8 := pre8 >>> (8 : Nat)
  let pre1 := hOpp &&& (hOpp >>> (1 : Nat))
  let flip7 := flip7 ||| (pre7 &&& (flip7 >>> (14 : Nat)))
  let flip9 := flip9 ||| (pre9 &&& (flip9 >>> (18 : Nat)))
  let flip8 := flip8 ||| (pre8 &&& (flip8 >>> (16 : Nat)))
  let flip1 := flip1 ||| (pre1 &&& (flip1 >>> (2 : Nat)))
  let flip7 := flip7 ||| (pre7 &&& (flip7 >>> (14 : Nat)))
  let flip9 := flip9 ||| (pre9 &&& (flip9 >>> (18 : Nat)))
  let flip8 := flip8 ||| (pre8 &&& (flip8 >>> (16 : Nat)))
  let flip1 := flip1 ||| (pre1 &&& (flip1 >>> (2 : Nat)))
  let moves := moves ||| ((flip7 >>> (7 : Nat)) |||
    (flip9 >>> (9 : Nat)) ||| (flip8 >>> (8 : Nat)) ||| (flip1 >>> (1 : Nat)))
  moves &&& ~~~(player ||| opponent)

end NeuralReversi.Bitboard.Moves

import NeuralReversi.Board.Move.Model
import NeuralReversi.Simd.Fixture

namespace NeuralReversi.Board.MoveFixture
open NeuralReversi.Simd.X86 (parseWord)

def wordResult : Except String Bitboard.Bits → String
  | .ok value => s!"ok:{value.toNat}"
  | .error message => s!"error:{message}"

def boardResult : Except String Board → String
  | .ok value => s!"ok:{value.player.toNat},{value.opponent.toNat}"
  | .error message => s!"error:{message}"

def request (fields : List String) : Except String String := do
  let [op,player,opponent,mask,square,backend] := fields |
    throw "expected player opponent mask square backend"
  let player ← parseWord player
  let opponent ← parseWord opponent
  let mask ← parseWord mask
  let some square := square.toNat? | throw "invalid square"
  let square : Bitboard.AnySquare ← if bound : square < 64 then pure ⟨square,bound⟩
    else throw "square outside typed domain"
  let config : Flip.Dispatch.X86Config ← match backend with
    | "0" => pure ⟨false,false,false⟩
    | "1" => pure ⟨false,false,true⟩
    | "2" => pure ⟨false,true,true⟩
    | "3" => pure ⟨true,true,true⟩
    | _ => throw "unsupported move backend"
  let checked := op.endsWith "checked"
  let input : Board := ⟨player,opponent⟩
  let flipped := Flip.Dispatch.flip config checked square player opponent
  let applied := Bitboard.applyMoveAny checked checked player mask square
  let placed := placeAny checked checked input mask square
  let moved := makeMove config checked checked input square
  let state := s!"{player.toNat},{opponent.toNat}"
  return s!"{wordResult flipped}|{wordResult applied}|{boardResult placed}|{boardResult moved}|{state}|{state}"

end NeuralReversi.Board.MoveFixture

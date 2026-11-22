import NeuralReversi.Board.MovesModel
import NeuralReversi.Simd.Fixture

namespace NeuralReversi.Board
open NeuralReversi.Simd.X86 (parseWord)

def wrapperRequest (fields : List String) : Except String String := do
  let [op,player,opponent,square,backend] := fields |
    throw "expected board word pair, square and backend code"
  let player ← parseWord player
  let opponent ← parseWord opponent
  let some square := square.toNat? | throw "invalid square"
  let square : Bitboard.AnySquare ← if bound : square < 64 then pure ⟨square,bound⟩
    else throw "square outside typed domain"
  let config : Bitboard.Moves.X86Config ← match backend with
    | "0" => pure ⟨false,false⟩
    | "1" => pure ⟨false,true⟩
    | "2" => pure ⟨true,true⟩
    | _ => throw "unsupported board-wrapper backend code"
  let checked := op.endsWith "checked"
  let b : Board := ⟨player,opponent⟩
  let bitboardPair := Bitboard.getMovesAndPotential config player opponent
  let boardPair := getMovesAndPotential config b
  let legal := match isLegalMove config checked checked b square with
    | .ok value => s!"ok:{if value then 1 else 0}"
    | .error message => s!"error:{message}"
  let state := s!"{b.player.toNat},{b.opponent.toNat}"
  return s!"{bitboardPair.1.toNat},{bitboardPair.2.toNat}|{boardPair.1.toNat},{boardPair.2.toNat}|{legal}|{state}|{state}"

def movesRequest (fields : List String) : Except String String := do
  let [_,player,opponent,reserved,backend] := fields |
    throw "expected board word pair, zero and backend code"
  let player ← parseWord player
  let opponent ← parseWord opponent
  let reserved ← parseWord reserved
  if reserved != 0 then throw "board moves reserves the fourth field as zero"
  let config : Bitboard.Moves.X86Config ← match backend with
    | "0" => pure ⟨false,false⟩
    | "1" => pure ⟨false,true⟩
    | "2" => pure ⟨true,true⟩
    | _ => throw "unsupported board-move backend code"
  let b : Board := ⟨player,opponent⟩
  let bitboardMoves := Bitboard.getMoves config player opponent
  let moves := getMoves config b
  let has := hasLegalMoves config b
  let over := isGameOver config b
  let state := s!"{b.player.toNat},{b.opponent.toNat}"
  return s!"{bitboardMoves.toNat}|{moves.toNat}|{if has then 1 else 0}|{if over then 1 else 0}|{state}|{state}"

end NeuralReversi.Board

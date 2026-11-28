module
public import NeuralReversi.Board.Model
public import NeuralReversi.Bitboard.RepresentationModel

@[expose] public section

namespace NeuralReversi.Board
open Bitboard

def defaultBoard : Board :=
  ⟨squareMask ⟨35, by decide⟩ ||| squareMask ⟨28, by decide⟩,
   squareMask ⟨27, by decide⟩ ||| squareMask ⟨36, by decide⟩⟩

def newBoard : Board := defaultBoard
def getPlayer (b : Board) : Value := ⟨b.player⟩
def getOpponent (b : Board) : Value := ⟨b.opponent⟩

end NeuralReversi.Board

module
public import NeuralReversi.Board.Model
public import NeuralReversi.Bitboard.DispatchModel
public import NeuralReversi.Bitboard.SquareModel

@[expose] public section

namespace NeuralReversi.Board

def getMoves (config : Bitboard.Moves.X86Config) (b : Board) : Bitboard.Bits :=
  Bitboard.getMoves config b.player b.opponent

def hasLegalMoves (config : Bitboard.Moves.X86Config) (b : Board) : Bool :=
  !(Bitboard.isEmpty (getMoves config b))

def isLegalMove (config : Bitboard.Moves.X86Config) (debugAssertions overflowChecks : Bool)
    (b : Board) (sq : Bitboard.AnySquare) : Except String Bool :=
  Bitboard.containsSquare debugAssertions overflowChecks (getMoves config b) sq

def getMovesAndPotential (config : Bitboard.Moves.X86Config) (b : Board) : Bitboard.Bits × Bitboard.Bits :=
  Bitboard.getMovesAndPotential config b.player b.opponent

def isGameOver (config : Bitboard.Moves.X86Config) (b : Board) : Bool :=
  if hasLegalMoves config b then false
  else !(hasLegalMoves config (switchPlayers b))

end NeuralReversi.Board

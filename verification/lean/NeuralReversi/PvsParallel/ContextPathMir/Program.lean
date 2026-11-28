module
public import NeuralReversi.SearchContextState
public import NeuralReversi.PvsParallel.ExactDepth
public import NeuralReversi.EndgameSearch.Cancellation

@[expose] public section

namespace NeuralReversi.PvsParallel.ContextPathMir

open PvsBoardTree (Position GeneratedSquare play pass)
open SearchContextState (State)

inductive Step where
  | move
  | pass
  deriving DecidableEq

inductive Callee where
  | move (pv : Bool)
  | pass
  deriving DecidableEq

def recursiveSite : Bool → Nat → Option Callee
  | false, 59 => some .pass
  | false, 257 => some (.move false)
  | false, 356 => some (.move false)
  | false, 365 => some (.move false)
  | false, 382 => some (.move true)
  | true, 26 => some (.move false)
  | true, 37 => some (.move true)
  | _, _ => none

def childSite (worker : Bool) (pc : Nat) : Option Step :=
  (recursiveSite worker pc).map fun
    | .move _ => .move
    | .pass => .pass

def start (root : Position) : State :=
  SearchContextState.withFeatures ⟨Board.emptyCount root.val⟩ false (SearchContextState.rebuild root.val)

def rootDepth (s : State) : ExactDepth.Depth := s.empties.count

def place (s : State) (sq : Bitboard.Square) (flips : Bitboard.Bits) : State :=
  SearchContextState.placement s sq flips

def restore (s : State) : State := SearchContextState.undo s

def passTurn (s : State) : State := SearchContextState.passTurn s

def unpass (s : State) : State := SearchContextState.undoPass s

def splitCopy (s : State) : State :=
  SearchContextState.withFeatures s.empties s.opponentTurn (s.slots (SearchContextState.ply s))

/-- `EndGameStrategy::shallow_search` builds its `Empties` from the board it receives. -/
def leafState (b : Position) : EndgameSearch.Cancellation.EmptyState :=
  ⟨Empties.encode (Board.emptySquares b.val), Board.emptyCount b.val⟩

end NeuralReversi.PvsParallel.ContextPathMir

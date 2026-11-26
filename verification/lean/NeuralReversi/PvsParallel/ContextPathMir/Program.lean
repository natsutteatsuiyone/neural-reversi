import NeuralReversi.SearchContextState
import NeuralReversi.PvsParallel.ExactDepth
import NeuralReversi.EndgameSearch.Cancellation

namespace NeuralReversi.PvsParallel.ContextPathMir

open PvsBoardTree (Position GeneratedSquare play pass)
open SearchContextState (State)

def scan : List Bitboard.Square := [0, 7, 56, 63, 2, 5, 16, 23, 40, 47, 58, 61, 18, 21, 42, 45, 3, 4, 24, 31, 32, 39, 59, 60, 19, 20, 26, 29, 34, 37, 43, 44, 11, 12, 25, 30, 33, 38, 51, 52, 10, 13, 17, 22, 41, 46, 50, 53, 1, 6, 8, 15, 48, 55, 57, 62, 9, 14, 49, 54, 27, 28, 35, 36]

def quad (sq : Bitboard.Square) : BitVec 8 :=
  (show Vector (BitVec 8) 64 from #v[1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8])[sq]

inductive Step where
  | move
  | pass
  deriving DecidableEq

inductive Callee where
  | move (pv : Bool)
  | pass
  deriving DecidableEq

def recursiveSite : Bool → Nat → Option Callee
  | false, 60 => some .pass
  | false, 246 => some (.move false)
  | false, 345 => some (.move false)
  | false, 354 => some (.move false)
  | false, 371 => some (.move true)
  | true, 26 => some (.move false)
  | true, 37 => some (.move true)
  | _, _ => none

def childSite (worker : Bool) (pc : Nat) : Option Step :=
  (recursiveSite worker pc).map fun
    | .move _ => .move
    | .pass => .pass

def start (root : Position) : State :=
  let bits := Board.emptySquares root.val
  let cells := scan.filter fun sq => bits.getLsbD sq.val
  SearchContextState.withFeatures
    ⟨EmptyListInit.newLinks bits, cells.foldl (fun a sq => a ^^^ quad sq) 0, BitVec.ofNat 32 cells.length⟩
    false (SearchContextState.rebuild root.val)

def rootDepth (s : State) : ExactDepth.Depth := s.empties.count

def place (s : State) (sq : Bitboard.Square) (flips : Bitboard.Bits) : State :=
  SearchContextState.placement s sq flips

def restore (s : State) (sq : Bitboard.Square) : State := SearchContextState.undo s sq

def passTurn (s : State) : State := SearchContextState.passTurn s

def unpass (s : State) : State := SearchContextState.undoPass s

def splitCopy (s : State) : State :=
  SearchContextState.withFeatures s.empties s.opponentTurn (s.slots (SearchContextState.ply s))

def leafState (s : State) : EndgameSearch.Cancellation.EmptyState :=
  ⟨s.empties.links, s.empties.parity, s.empties.count⟩

end NeuralReversi.PvsParallel.ContextPathMir

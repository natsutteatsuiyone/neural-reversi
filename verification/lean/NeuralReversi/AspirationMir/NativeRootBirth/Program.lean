module
public import NeuralReversi.PvsParallel.PvEntryMir
public import NeuralReversi.RootMoveStorage

@[expose] public section

namespace NeuralReversi.AspirationMir.NativeRootBirth

open PvsParallel

def capacity : Nat := 35
def negativeInfinity : Int := -32767

def fresh (sq : Bitboard.Square) : RootMoveStorage.Row :=
  ⟨sq, negativeInfinity, negativeInfinity, 0, []⟩

def boot (mc : Bitboard.Moves.X86Config) (fc : Cutoffs.FlipConfig) (b : PvsBoardTree.Position)
    (paired : Bool) (mem : MoveList.Initialization.Memory Move.Move) : Option (Nat × List RootMoveStorage.Row) := do
  let list := PvEntryMir.generated mc fc b paired mem
  let n ← list.len
  if n ≤ capacity then
    let rows ← (List.finRange n).mapM fun i => (list.slots i.val).map fun mv => fresh mv.sq
    return (0,rows)
  else none

end NeuralReversi.AspirationMir.NativeRootBirth

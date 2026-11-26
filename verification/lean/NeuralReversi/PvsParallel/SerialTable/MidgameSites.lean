import NeuralReversi.PvsParallel.SerialCalls

namespace NeuralReversi.PvsParallel.SerialTable.MidgameSites

open SerialCalls (Request)
open PvsBoardTree (Position)

/-- The caller supplies the current alpha and depth at the suspension point, after any
    internal iterative reduction. The pass branch precedes that reduction. -/
def passed (q : Request) : Request :=
  ⟨PvsBoardTree.pass q.board, q.depth, q.pv, !q.pv && !q.cut, -q.beta, -q.alpha⟩

def scout (q : Request) (next : Position) (reduction : ExactDepth.Depth) : Request :=
  ⟨next, q.depth - 1 - reduction, false, reduction > 0 || !q.cut, -(q.alpha + 1), -q.alpha⟩

def retry (q : Request) (next : Position) : Request :=
  ⟨next, q.depth - 1, false, !q.cut, -(q.alpha + 1), -q.alpha⟩

def wide (q : Request) (next : Position) : Request :=
  ⟨next, q.depth - 1, true, false, -q.beta, -q.alpha⟩

def verified (q : Request) (pcDepth : ExactDepth.Depth) (threshold : Int) : Request :=
  ⟨q.board, pcDepth, false, true, threshold - 1, threshold⟩

/-- The shallow evaluator takes no cut-node argument. -/
def shallowMove (q : Request) (next : Position) : Request :=
  ⟨next, q.depth - 1, false, false, -q.beta, -q.alpha⟩

end NeuralReversi.PvsParallel.SerialTable.MidgameSites

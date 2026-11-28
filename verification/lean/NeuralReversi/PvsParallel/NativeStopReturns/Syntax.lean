module
public import NeuralReversi.ScoreRefresh.BoardBridge

@[expose] public section

namespace NeuralReversi.PvsParallel.NativeStopReturns

inductive Mode | cached | postJoin | passChild | terminal deriving DecidableEq, Repr

inductive Op
  | poll | taken | zero | releaseClaim | ret | child | negate | undoPass | forward | solve
  | tableStore | rootScore
  deriving DecidableEq, Repr

structure Route where
  mode : Mode
  continuePc : Nat
  blocks : List (Nat × List Op)
  deriving DecidableEq, Repr

def Route.pcs (r : Route) := r.blocks.map Prod.fst
def Route.ops (r : Route) := r.blocks.flatMap Prod.snd
def Route.publications (r : Route) := r.ops.filter (fun op => op == .tableStore || op == .rootScore)

end NeuralReversi.PvsParallel.NativeStopReturns

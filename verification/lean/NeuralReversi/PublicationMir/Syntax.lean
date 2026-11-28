module
public import NeuralReversi.SearchFallback
public import NeuralReversi.RootMoveStorage

@[expose] public section

namespace NeuralReversi.PublicationMir

open RootMoveStorage (Row)

/-- The two properties of Rust's slice sort needed to compare the reported scores.
Equal keys may be permuted arbitrarily. -/
structure SliceSort where
  byReverseKey : (Row → Int) → List Row → List Row
  permutation : ∀ k rows, (byReverseKey k rows).Perm rows
  ordered : ∀ k rows, (byReverseKey k rows).Pairwise (fun a b => k b ≤ k a)

structure Iteration where
  rows : List Row
  depth : Nat
  selectivity : Nat
  endgame : Bool

inductive SearchResult where
  | bestMove (report : SearchFallback.Report)
  | noLegalMove

end NeuralReversi.PublicationMir

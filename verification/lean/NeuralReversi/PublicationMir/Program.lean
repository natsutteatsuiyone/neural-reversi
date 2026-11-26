import NeuralReversi.PublicationMir.Syntax

namespace NeuralReversi.PublicationMir

open RootMoveStorage (Row)

def snapshot (rows : List Row) : List Row :=
  rows

def key (r : Row) : Int :=
  r.score

def capture (sort : SliceSort) (rows : List Row) (depth selectivity : Nat) (endgame : Bool) : Iteration :=
  ⟨(sort.byReverseKey key (snapshot rows)), depth, selectivity, endgame⟩

def first (saved : Iteration) : Option Row := do
  let chosen ← saved.rows.head?
  return chosen

def record (sort : SliceSort) (rows : List Row) (depth selectivity : Nat) (endgame : Bool) : Option (Option Iteration × Row) := do
  let chosen ← first (capture sort rows depth selectivity endgame)
  return ((some (capture sort rows depth selectivity endgame)), chosen)

def project (r : Row) : SearchPublication.Row :=
  ⟨r.square, r.score, r.pv⟩

def build (rows : List Row) (r : Row) (depth selectivity : Nat) (endgame : Bool) (nodes : Nat) : SearchResult :=
  .bestMove ⟨⟨r.square, r.score, r.pv⟩, (rows.map project), ⟨depth, selectivity⟩, endgame, nodes⟩

def intoResult (last : Option Iteration) (nodes : Nat) (fallback : Unit → SearchResult) :
    Option SearchResult :=
  match last with
  | none => do
    return (fallback ())
  | some saved => do
    let chosen ← first saved
    return (build saved.rows chosen saved.depth saved.selectivity saved.endgame nodes)

def score : SearchResult → Option Int
  | .bestMove report => some report.best.score
  | .noLegalMove => none

end NeuralReversi.PublicationMir

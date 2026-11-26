import NeuralReversi.PublicationMir.Correct

namespace NeuralReversi.OuterMir

def ladder : List Nat := [1, 2, 3, 4, 5]

def directEntry (count : Nat) (multi : Bool) : Option Nat :=
  if count = 0 then none else some (if multi then count else 1)

def lineStamp (lines : List (Option (Nat × Nat))) (idx depth level : Nat) :
    List (Option (Nat × Nat)) × Bool :=
  let mark := some (depth, level)
  let next := lines.set idx mark
  (next, next.all (· == mark))

inductive Step where
  | endLine
  | accepted (level : Nat) (publish mayContinue : Bool)
deriving DecidableEq, Repr

def step (next : Option Nat) (last minimum : Nat) (stop : Bool) : Step :=
  match next with
  | none => .endLine
  | some level =>
    if last < level then .endLine else .accepted level (decide (minimum ≤ level)) (!stop)

def finish (minimum level : Nat) (sort : PublicationMir.SliceSort) (rows : List RootMoveStorage.Row)
    (depth nodes : Nat) (prior : Option PublicationMir.Iteration) (fallback : Unit → PublicationMir.SearchResult) :
    Option PublicationMir.SearchResult :=
  if minimum ≤ level then PublicationMir.completed sort rows depth level true nodes fallback
  else PublicationMir.intoResult prior nodes fallback

end NeuralReversi.OuterMir

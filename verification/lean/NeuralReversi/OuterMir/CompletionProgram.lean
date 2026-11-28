module
public import NeuralReversi.OuterMir.Program

@[expose] public section

namespace NeuralReversi.OuterMir.Completion

open PublicationMir (Iteration SliceSort SearchResult)
open RootMoveStorage (Row)

structure State where
  marks : List (Option (Nat × Nat))
  last : Option Iteration

def new (count : Nat) : State :=
  ⟨List.replicate count none, none⟩

def beginPhase (state : State) : State :=
  ⟨List.replicate state.marks.length none, state.last⟩

def iterationCompleted (sort : SliceSort) (rows : List Row) (depth level : Nat)
    (endgame : Bool) (state : State) : Option (State × Row) := do
  let (last, best) ← PublicationMir.record sort rows depth level endgame
  return (⟨state.marks, last⟩, best)

def lineCompleted (sort : SliceSort) (rows : List Row) (idx depth level : Nat)
    (endgame : Bool) (state : State) : Option (State × Option Row) := do
  if idx < state.marks.length then
    let (marks, ready) := lineStamp state.marks idx depth level
    let marked : State := ⟨marks, state.last⟩
    if ready then
      let (saved, best) ← iterationCompleted sort rows depth level endgame marked
      return (saved, some best)
    else
      return (marked, none)
  else
    none

def intoResult (state : State) (nodes : Nat) (fallback : Unit → SearchResult) : Option SearchResult :=
  PublicationMir.intoResult state.last nodes fallback

end NeuralReversi.OuterMir.Completion

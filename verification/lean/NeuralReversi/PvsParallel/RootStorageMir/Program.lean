module
public import NeuralReversi.RootMoveStorage
public import NeuralReversi.MidgameAspiration.Arithmetic

@[expose] public section

namespace NeuralReversi.PvsParallel.RootStorageMir

open Bitboard (Square)
open RootMoveStorage (Row)

def capacity : Nat := 61
abbrev Slots := Vector (Option Square) capacity
def negInf : Int := -32767

def pvPrefix : List (Option Square) → List Square
  | some sq :: tail => sq :: pvPrefix tail
  | _ => []

def reset (idx : Nat) (rows : List Row) : List Row :=
  rows.take idx ++ (rows.drop idx).map (fun old => {old with score := negInf})

def sort (idx : Nat) (rows : List Row) : List Row :=
  if idx < rows.length then
    rows.take idx ++ (rows.drop idx).mergeSort (fun a b => decide (b.score ≤ a.score)) else rows

def changed (old : Row) (value : Int) (shown : Bool) (slots : Slots) : Row :=
  let seed := old.average = negInf
  let squared := value * (value.natAbs : Int)
  let averageSum := old.average + value
  let squareSum := old.meanSquared + squared
  ⟨old.square, (if shown then value else negInf), (if seed then value else averageSum.tdiv 2), (if seed then squared else squareSum.tdiv 2), (if shown then pvPrefix slots.toList else old.pv)⟩

instance (bits : Nat) (x : Int) : Decidable (MidgameAspiration.FitsSigned bits x) :=
  inferInstanceAs (Decidable (_ ≤ x ∧ x < _))

def numeric (old : Row) (value : Int) : Bool :=
  let squared := value * (value.natAbs : Int)
  decide (MidgameAspiration.FitsSigned 32 (value.natAbs : Int) ∧ MidgameAspiration.FitsSigned 64 squared ∧
    (old.average ≠ negInf → MidgameAspiration.FitsSigned 32 (old.average + value) ∧
      MidgameAspiration.FitsSigned 64 (old.meanSquared + squared)))

def update (square : Square) (value : Int) (shown : Bool) (slots : Slots) : List Row → Option (List Row)
  | [] => none
  | old :: tail => if old.square = square then
      if numeric old value then some (changed old value shown slots :: tail) else none
    else (update square value shown slots tail).map (old :: ·)

end NeuralReversi.PvsParallel.RootStorageMir

namespace NeuralReversi.PvsParallel.DeepLeafMir

def parallelPool (size : Nat) : Bool := decide (1 < size)

def boundaryStoreMinNodes : Nat := 256

def boundaryPvsSite : Nat := 1

inductive Arm where
  | probe | activeNws | ignoreNws
  deriving DecidableEq

def dispatch (empties : Nat) (parallel tableCut active : Bool) : Option Arm :=
  if 6 < empties ∧ empties ≤ 12 then
    if empties = 12 ∧ parallel ∧ tableCut then some .probe
    else some (if active then .activeNws else .ignoreNws)
  else none

end NeuralReversi.PvsParallel.DeepLeafMir

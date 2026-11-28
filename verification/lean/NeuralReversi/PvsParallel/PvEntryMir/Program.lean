module

@[expose] public section

namespace NeuralReversi.PvsParallel.PvEntryMir

inductive Action where
  | zero | terminal | pass | wipeout | continueSearch
  deriving DecidableEq

def dispatch (depth : Nat) (mobile opponent wipes : Bool) : Action :=
  if depth = 0 then .zero else if mobile then
    if wipes then .wipeout else .continueSearch
  else if opponent then .pass else .terminal

end NeuralReversi.PvsParallel.PvEntryMir

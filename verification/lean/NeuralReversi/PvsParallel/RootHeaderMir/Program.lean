namespace NeuralReversi.PvsParallel

def rootMinimumSplitDepth : Nat := 12

inductive HeaderAction where
  | finish
  | split
  | postpone (idx fresh : Nat)
  | child (next : Nat) (scoutUsed pvUsed : Bool) (score : Int)
deriving DecidableEq, Repr

def headerAction (idx count fresh depth minSplit : Nat) (canSplit busy : Bool) (alpha scout pv : Int) :
    HeaderAction :=
  let branch1 : HeaderAction := if alpha < -scout then .child (idx + 1) true true (-pv) else .child (idx + 1) true false (-scout)
  let branch2 : HeaderAction := if idx + 1 < fresh then .postpone idx (fresh - 1) else branch1
  let branch3 : HeaderAction := if busy then branch2 else branch1
  let branch4 : HeaderAction := if canSplit then .split else branch3
  let branch5 : HeaderAction := if 2 ≤ count - idx then branch4 else branch3
  let branch6 : HeaderAction := if minSplit ≤ depth then branch5 else branch3
  let branch7 : HeaderAction := if idx = 0 then .child (idx + 1) false true (-pv) else branch6
  let branch8 : HeaderAction := if idx < count then branch7 else .finish
  branch8

end NeuralReversi.PvsParallel

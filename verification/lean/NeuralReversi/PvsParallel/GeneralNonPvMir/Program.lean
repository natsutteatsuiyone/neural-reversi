module

@[expose] public section

namespace NeuralReversi.PvsParallel.GeneralNonPvMir

inductive Action where
  | shallow | stability | terminal | pass | wipeout | probe | enhanced | cache | rest
  deriving DecidableEq

def dispatch (depth : Nat) (stable mobile other wipes parentCut childCut : Bool)
    (cacheCut : Bool := false) : Action :=
  if depth ≤ 12 then .shallow else if stable then .stability else if !mobile then
    if other then .pass else .terminal
  else if wipes then .wipeout else if parentCut then .probe
  else if 14 ≤ depth then
    if childCut then .enhanced else .rest
  else if depth = 13 && cacheCut then .cache else .rest

inductive Presearch (Move : Type) where
  | owner (idx : Nat) (best : Int)
  | storeLower (move : Move) (score : Int)

def presearch (count : Nat) (hint : Option Move) (first : Move) (alpha beta : Int)
    (scout : Move → Int → Int → Int) : Presearch Move :=
  if count = 1 || hint.isSome then
    let move := hint.getD first
    let score := -(scout move (-(alpha + 1)) (-alpha))
    if alpha < score && beta ≤ score then .storeLower move score else .owner 1 score
  else .owner 0 (-32767)

end NeuralReversi.PvsParallel.GeneralNonPvMir

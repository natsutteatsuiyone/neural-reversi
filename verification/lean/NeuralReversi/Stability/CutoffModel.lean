module
public import NeuralReversi.Stability.EdgeAssembly
public import NeuralReversi.Stability.Propagation

@[expose] public section

namespace NeuralReversi.Stability.Cutoff

open NeuralReversi.Stability.Semantics

structure DiscScore where
  value : Int
  deriving DecidableEq, Repr

def discScoreMax : DiscScore := ⟨64⟩

def stabilityCutoffDiscMargin : Int := 8

def nwsStabilityThreshold (nEmpties : Fin 64) : Int :=
  match nEmpties.val with
  | 0  => 99 | 1  => 99 | 2  => 99 | 3  => 99 | 4  => 6  | 5  => 8  | 6  => 10 | 7  => 12
  | 8  => 14 | 9  => 16 | 10 => 20 | 11 => 22 | 12 => 24 | 13 => 26 | 14 => 28 | 15 => 30
  | 16 => 32 | 17 => 34 | 18 => 36 | 19 => 38 | 20 => 40 | 21 => 42 | 22 => 44 | 23 => 46
  | 24 => 48 | 25 => 48 | 26 => 50 | 27 => 50 | 28 => 52 | 29 => 52 | 30 => 54 | 31 => 54
  | 32 => 56 | 33 => 56 | 34 => 58 | 35 => 58 | 36 => 60 | 37 => 60 | 38 => 62 | 39 => 62
  | 40 => 64 | 41 => 64 | 42 => 64 | 43 => 64 | 44 => 64 | 45 => 64 | 46 => 64 | 47 => 64
  | _  => 99

def popcount (w : Word) : Nat :=
  (List.finRange 64).countP (fun sq => w.getLsbD sq.val)

def hasEnoughOpponentDiscsForCutoff (opponentCount : Nat) (alpha : DiscScore) : Bool :=
  decide (2 * (opponentCount : Int) ≥
    discScoreMax.value - alpha.value + 2 * stabilityCutoffDiscMargin)

def singleEmptyMaxShortfall : Int := 4

def byAxisExec (f : Fin 4 → Word) : FullLines := ⟨f 0, f 1, f 2, f 3⟩

def noWrapL1Exec (axis : Fin 4) : Word :=
  match axis.val with
  | 0 => 0xfefefefefefefefe
  | 1 => 0xffffffffffffffff
  | 2 => 0x7f7f7f7f7f7f7f7f
  | _ => 0xfefefefefefefefe

def noWrapR1Exec (axis : Fin 4) : Word :=
  match axis.val with
  | 0 => 0x7f7f7f7f7f7f7f7f
  | 1 => 0xffffffffffffffff
  | 2 => 0xfefefefefefefefe
  | _ => 0x7f7f7f7f7f7f7f7f

def contactLinesExec (known : Word) (lines : FullLines) : FullLines :=
  byAxisExec fun axis =>
    ((known <<< stride axis) &&& noWrapL1Exec axis) |||
    ((known >>> stride axis) &&& noWrapR1Exec axis) |||
    field lines axis

def oneEmptyShieldExec (disc : Word) (axis : Fin 4) (x : Fin 64) (blockers : Word) : Bool :=
  let cells := (List.finRange 64).filter fun z => decide (sameLine axis x.val z.val)
  match cells.filter (fun z => !disc.getLsbD z.val) with
  | [hole] => cells.any fun y =>
      blockers.getLsbD y.val &&
        decide ((hole.val < y.val ∧ y.val < x.val) ∨ (x.val < y.val ∧ y.val < hole.val))
  | _ => false

def keepersExec (discs : Word) (lines : FullLines) (axis : Fin 4) : Word :=
  encodeBits fun sq => discs.getLsbD sq.val && (List.finRange 4).all fun b =>
    decide (b = axis) || (field lines b).getLsbD sq.val

def shieldLinesExec (disc : Word) (base : FullLines) (blockers : Word)
    (blockerLines : FullLines) : FullLines := byAxisExec fun axis =>
  let keep := keepersExec blockers blockerLines axis
  field base axis ||| encodeBits (fun sq => oneEmptyShieldExec disc axis sq keep)

def getStableDiscsExec (player opponent : Word) : Word :=
  getStableByContact (player &&& center)
    (getStableEdge Edge.initEdgeStability player opponent)
    (getFullLines (player ||| opponent))

def singleEmptyStableExec (player opponent stable : Word) : Word :=
  let disc := player ||| opponent
  let full := getFullLines disc
  let playerContact := contactLinesExec stable full
  let opponentEdge := getStableEdge Edge.initEdgeStability opponent player
  let opponentSafe := shieldLinesExec disc (contactLinesExec opponentEdge full) player playerContact
  let playerSafe := shieldLinesExec disc playerContact opponent opponentSafe
  getStableByContact (player &&& center) stable playerSafe

def cutoffStableExec (discs other : Word) (alpha : DiscScore) : Word :=
  let stable := getStableDiscsExec discs other
  let shortfall := discScoreMax.value - 2 * (popcount stable : Int) - alpha.value
  if 0 < shortfall ∧ shortfall ≤ 2 * singleEmptyMaxShortfall then
    singleEmptyStableExec discs other stable
  else stable

def stabilityCutoffExec (player opponent : Word) (nEmpties : Fin 64) (alpha : DiscScore) :
    Option DiscScore :=
  if alpha.value ≥ nwsStabilityThreshold nEmpties then
    if !hasEnoughOpponentDiscsForCutoff (popcount opponent) alpha then
      none
    else
      let score : DiscScore :=
        ⟨discScoreMax.value - 2 * (popcount (cutoffStableExec opponent player alpha) : Int)⟩
      if score.value ≤ alpha.value then
        some score
      else
        none
  else
    none

def cutoffAtEmptyCountExec (player opponent : Word) (nEmpties : Nat) (alpha : DiscScore) :
    Option DiscScore :=
  if inTable : nEmpties < 64 then
    stabilityCutoffExec player opponent ⟨nEmpties, inTable⟩ alpha
  else none

end NeuralReversi.Stability.Cutoff

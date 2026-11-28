module

@[expose] public section

namespace NeuralReversi.Stability.Edge

abbrev Byte := BitVec 8
abbrev EdgeTable := Array Byte
abbrev State := Byte × Byte

def bit8 (x : Nat) : Byte := (1 : Byte) <<< x

def scanLeft (o : Byte) : Nat → Nat
  | 0 => 0
  | y + 1 => if o &&& bit8 (y + 1) != 0 then scanLeft o y else y + 1

def flipLeft (p o : Byte) : Nat → State
  | 0 => (p, o)
  | y + 1 => if o &&& bit8 (y + 1) != 0 then
      flipLeft (p ^^^ bit8 (y + 1)) (o ^^^ bit8 (y + 1)) y else (p, o)

def scanRight (o : Byte) (y : Nat) : Nat → Nat
  | 0 => y
  | fuel + 1 => if o &&& bit8 y != 0 then scanRight o (y + 1) fuel else y

def flipRight (p o : Byte) (y : Nat) : Nat → State
  | 0 => (p, o)
  | fuel + 1 => if o &&& bit8 y != 0 then
      flipRight (p ^^^ bit8 y) (o ^^^ bit8 y) (y + 1) fuel else (p, o)

def applyEdgeMove (x : Fin 8) (player opponent : Byte) : State :=
  let p := player ||| bit8 x.val
  let left := if x.val > 1 then
    let y := scanLeft opponent (x.val - 1)
    if p &&& bit8 y != 0 then flipLeft p opponent (x.val - 1) else (p, opponent)
    else (p, opponent)
  if x.val < 6 then
    let y := scanRight left.2 (x.val + 1) (8 - (x.val + 1))
    if y < 8 && left.1 &&& bit8 y != 0 then
      flipRight left.1 left.2 (x.val + 1) (8 - (x.val + 1)) else left
  else left

def edgeLookupIndex (p occ : Byte) : Nat := occ.toNat * 256 + p.toNat

def edgeIndex (p o : Byte) : Nat := edgeLookupIndex p (p ||| o)

def edgeStabilityOcc (table : EdgeTable) (p occ : Byte) : Byte :=
  table[edgeLookupIndex p occ]?.getD 0

def edgeStability (table : EdgeTable) (p o : Byte) : Byte :=
  edgeStabilityOcc table p (p ||| o)

def emptyAt (p o : Byte) (x : Fin 8) : Bool := (p ||| o) &&& bit8 x.val == 0

def firstChild (p o : Byte) (x : Fin 8) : State := applyEdgeMove x p o

def secondChild (p o : Byte) (x : Fin 8) : State :=
  let q := applyEdgeMove x o p
  (q.2, q.1)

def findAux (lookup : Byte → Byte → Byte) (p o stable : Byte) : List (Fin 8) → Byte
  | [] => stable
  | x :: xs =>
    if stable == 0 then stable else
    if emptyAt p o x then
      let a := firstChild p o x
      let stable := stable &&& lookup a.1 a.2
      if stable == 0 then stable else
      let b := secondChild p o x
      let stable := stable &&& lookup b.1 b.2
      if stable == 0 then stable else findAux lookup p o stable xs
    else findAux lookup p o stable xs

def findWith (lookup : Byte → Byte → Byte) (p o : Byte) : Byte :=
  if p == 0 || ~~~(p ||| o) == 0 then p
  else findAux lookup p o p (List.finRange 8)

def findEdgeStable (p o : Byte) (table : EdgeTable) : Byte :=
  findWith (edgeStability table) p o

def fillSubsets (m p : Byte) (table : EdgeTable) : Nat → EdgeTable
  | 0 => table
  | fuel + 1 =>
      let o := m &&& ~~~p
      let table := table.set! (edgeIndex p o) (findEdgeStable p o table)
      let next := (p - m) &&& m
      if next == 0 then table else fillSubsets m next table fuel

def fillOccupancy (occupied : Nat) (table : EdgeTable) : EdgeTable :=
  (List.range 256).foldl (fun table mNat =>
    let m := BitVec.ofNat 8 mNat
    if m.cpop.toNat == occupied then fillSubsets m 0 table 256 else table) table

def initEdgeStability : EdgeTable :=
  ((List.range 9).reverse).foldl (fun table occupied => fillOccupancy occupied table)
    (Array.replicate 65536 0)

def stableFuel : Nat → Byte → Byte → Byte
  | 0, p, _ => p
  | fuel + 1, p, o => findWith (stableFuel fuel) p o

def recursiveStable (p o : Byte) : Byte := stableFuel 8 p o

end NeuralReversi.Stability.Edge

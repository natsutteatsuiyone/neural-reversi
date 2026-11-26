import NeuralReversi.EndgameSearch.MoveOrder
import NeuralReversi.MoveList.Ordering
import NeuralReversi.MoveList.OrderWeights

namespace NeuralReversi.PvsParallel.EndgameOrderMir

open Bitboard.Moves (X86Config)
open PvsBoardTree (Position)
open MoveList.Ordering (sortByKeyDesc swapList)
open MoveList.OrderWeights

/-- Numeric result origins for search ordering; several control paths select or negate another origin. -/
inductive Estimate where
  | network (raw : Int)
  | cacheHit (packed : BitVec 16)
  | terminal (b : Position)
  | finalCount (board : Board.Board)
  | maximum
  | infinity
  | neg (child : Estimate)
  | higher (a b : Estimate)

def Estimate.read : Estimate → Int
  | .network raw => min 16383 (max (-16383) raw)
  | .cacheHit packed => packed.toInt
  | .terminal b => (PvsBoardTree.terminalScore b).val
  | .finalCount b => 256 * (2 * ((Bitboard.count b.player).toNat : Int) - 64)
  | .maximum => 16384
  | .infinity => 32767
  | .neg x => -x.read
  | .higher x y => max x.read y.read

inductive Heuristic where
  | fast
  | skip
  | search (estimate : Estimate)

def minSortDepth : Vector Nat 64 := #v[19, 18, 18, 18, 17, 17, 17, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

def useFast (depth : BitVec 32) (empty : Fin 64) : Bool := decide (depth.toNat < minSortDepth[empty])

def raw (method : Heuristic) (mc : X86Config) (next : Board.Board) (sq : Bitboard.Square) : Int :=
  let st : Int := (Bitboard.cornerStability next.opponent).toNat
  match method with
  | .fast => squareValue[sq] * squareWeight + st * stabilityWeight + (36 - ((Bitboard.cornerWeightedCount (Board.getMoves mc next)).toNat : Int)) * mobilityWeight
  | .skip => -((Bitboard.cornerWeightedCount (Board.getMoves mc next)).toNat : Int)
  | .search score =>
    let both := Board.getMovesAndPotential mc next
    score.read + st * searchStabilityWeight - ((Bitboard.cornerWeightedCount both.1).toNat : Int) * searchMobilityWeight -
      ((Bitboard.cornerWeightedCount both.2).toNat : Int) * searchPotentialWeight

def tt : Int := 1073741824

def actualKey [DecidableEq Move] (priority : Move → Int) (hint : Option Move) (m : Move) : Int :=
  if some m = hint then tt else priority m

/-- A witness for the enumeration of ties by the standard library's unstable sort. -/
def sortMoves {Move : Type} (priority : Move → Int) (tie : Move → Fin 64) (moves : List Move) : List Move :=
  if moves.length ≤ 3 then sortByKeyDesc priority moves
  else sortByKeyDesc (fun m => 64 * priority m + (tie m).val) moves

def ranked (priority : Move → Int) (tie : Move → Fin 64) (length : Nat) (m : Move) : Int :=
  if length ≤ 3 then priority m else 64 * priority m + (tie m).val

def ordinary {Move : Type} [DecidableEq Move] (priority : Move → Int) (tie : Move → Fin 64)
    (presearched : Bool) (hint : Option Move) (moves : List Move) : List Move :=
  if 1 < moves.length - (if presearched then 1 else 0) then sortByKeyDesc (actualKey (ranked priority tie moves.length) hint) moves
  else match moves with
    | [a,b] => if some a = hint then [a,b] else swapList [a,b] 0 1
    | _ => moves

end NeuralReversi.PvsParallel.EndgameOrderMir

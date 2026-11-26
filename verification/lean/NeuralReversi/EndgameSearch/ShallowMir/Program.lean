import NeuralReversi.EndgameSearch.FourOrder

namespace NeuralReversi.EndgameSearch.ShallowMir

open EndgameTreeCache (Store)
open PvsBoardTree (Position)

def masks (moves : LegalMoves.Raw) (empties oddQuadrants : Bitboard.Bits) : List LegalMoves.Raw :=
  let neighbors := ShallowMoveOrder.neighbors empties
  let isolated := LegalMoves.bitand moves (~~~neighbors)
  let rest := LegalMoves.andNot moves isolated.bits
  let odd := LegalMoves.bitand rest oddQuadrants
  let even := LegalMoves.bitand rest (~~~oddQuadrants)
  [isolated, LegalMoves.corners odd, LegalMoves.nonCorners odd, LegalMoves.corners even, LegalMoves.nonCorners even]

def write (b : Position) (alpha : Int) (r : Result) : Result :=
  ⟨r.score, r.history ++ [⟨.shallow, ⟨b.val, alpha, r.score⟩⟩]⟩

inductive Stage where
  | cutoff (result : Result)
  | next (best : Int) (history : List Store)

def stage (b : Position) (beta : Int) (child : Move → List Store → Result) :
    List Move → Int → List Store → Stage
  | [], best, h => .next best h
  | m :: rest, best, h =>
    let r := child m h
    if beta ≤ r.score then .cutoff (write b (beta - 1) r)
    else stage b beta child rest (max best r.score) r.history

def groups (b : Position) (alpha : Int) (child : Move → List Store → Result) :
    List (List Move) → Int → List Store → Result
  | [], best, h => write b alpha ⟨best, h⟩
  | stageMoves :: later, best, h =>
    match stage b (alpha + 1) child stageMoves best h with
    | .cutoff r => r
    | .next best' h' => groups b alpha child later best' h'

def node (b : Position) (moves : List Move) (alpha : Int) (h : List Store)
    (other : Bool) (terminal : Int) (passed : Unit → Result) (probe stable : Option Int)
    (stages : List (List Move)) (child : Move → List Store → Result) : Result :=
  match moves with
  | [] => if other then let r := passed (); ⟨-r.score, r.history⟩ else ⟨terminal, h⟩
  | m :: rest =>
    match probe with
    | some score => ⟨score, h⟩
    | none =>
      match stable with
      | some score => ⟨score, h⟩
      | none =>
        if rest.isEmpty then write b alpha (child m h)
        else groups b alpha child stages (-127) h

def child (b : Position) (alpha : Int) (h : List Store) (count : Nat) (probe stable : Option Int)
    (solveFour : Unit → Int) (recurse : Unit → Result) : Result :=
  if count = 4 then
    match probe with
    | some score => ⟨-score, h⟩
    | none =>
      match stable with
      | some score => ⟨-score, h⟩
      | none =>
        let v := solveFour ()
        ⟨-v, (write b alpha ⟨v, h⟩).history⟩
  else let r := recurse (); ⟨-r.score, r.history⟩

end NeuralReversi.EndgameSearch.ShallowMir

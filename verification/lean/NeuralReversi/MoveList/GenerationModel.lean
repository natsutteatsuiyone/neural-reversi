module
public import NeuralReversi.MoveList.Wipeout
public import NeuralReversi.Move.Model
public import NeuralReversi.Board.Model
public import NeuralReversi.Flip.Dispatch.Model

@[expose] public section

namespace NeuralReversi.MoveList.Generation
open Capacity Initialization
open Bitboard (Bits)
open Board (Board)

def squares (selected : Bits) : List (Fin 64) :=
  (List.finRange 64).filter (selected.getLsbD ∘ Fin.val)

def entry (fc : Flip.Dispatch.X86Config) (board : Board) (sq : Fin 64) : Move.Move :=
  ⟨Flip.Dispatch.resultValue fc sq board.player board.opponent, 0, sq⟩

def records (fc : Flip.Dispatch.X86Config) (board : Board) (selected : Bits) : List Move.Move :=
  (squares selected).map (entry fc board)

def candidatesForCache (moves : List Move.Move) : List Wipeout.Move :=
  moves.map fun mv => ⟨mv.sq, mv.flipped⟩

def fill (skipEmpty paired : Bool) (fc : Flip.Dispatch.X86Config) (board : Board)
    (selected : Bits) (mem : Memory Move.Move) : Result Move.Move (Fin 64) :=
  let moves := records fc board selected
  let values := fun i => moves[i]?.getD ⟨0, 0, 0⟩
  let cache := Wipeout.generationCache board.opponent (candidatesForCache moves) none
  match dispatch skipEmpty selected with
  | .empty => finish mem 0 none
  | .single => finish (scalar values 1 0 mem) 1 cache
  | .bulk =>
    if paired then finish (sourcePairRun values selected mem) (pairFinalLen selected) cache
    else finish (scalar values (population selected) 0 mem) (population selected) cache

end NeuralReversi.MoveList.Generation

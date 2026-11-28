module
public import NeuralReversi.EndgameCache
public import NeuralReversi.MulHi

@[expose] public section

namespace NeuralReversi.EndgameSearch.CacheMir

open EndgameCache (Entry)
open Board (Board)

def lowerUnset : Int8 := -65
def upperUnset : Int8 := 65
def initialCell : Entry := ⟨⟨0, 0⟩, lowerUnset, upperUnset⟩

def newCache (bytes : Nat) : Option (Array Entry) :=
  if 0 < bytes / 18 then some (Array.replicate (bytes / 18) initialCell) else none

def highWord (a b : BitVec 64) : BitVec 64 :=
  ((((a.setWidth 128) * (b.setWidth 128)) >>> (64 : Nat)).setWidth 64)

def slot (width length : Nat) (key : BitVec 64) : Nat :=
  ((highWord key (BitVec.ofNat 64 length)).setWidth width).toNat

def scoreForAlpha (e : Entry) (alpha : Int) : Option Int :=
  if e.lower.toInt > alpha then
    some e.lower.toInt
  else
    if e.upper.toInt ≤ alpha then
      some e.upper.toInt
    else
      none

def merge (e : Entry) (alpha score : Int) : Entry :=
  if score > alpha then
    ⟨⟨e.board.player, e.board.opponent⟩, (max e.lower (Int8.ofInt score)), e.upper⟩
  else
    ⟨⟨e.board.player, e.board.opponent⟩, e.lower, (min e.upper (Int8.ofInt score))⟩

def overwrite (_prior : Entry) (board : Board) (alpha score : Int) : Entry :=
  merge ⟨⟨board.player, board.opponent⟩, lowerUnset, upperUnset⟩ alpha score

def probeCell (e : Entry) (board : Board) (alpha : Int) : Option Int :=
  if e.board.player ≠ board.player then
    none
  else
    if e.board.opponent ≠ board.opponent then
      none
    else
      scoreForAlpha e alpha

def storeCell (e : Entry) (board : Board) (alpha score : Int) : Entry :=
  if e.board.player = board.player then
    if e.board.opponent = board.opponent then
      merge ⟨⟨e.board.player, e.board.opponent⟩, e.lower, e.upper⟩ alpha score
    else
      overwrite e board alpha score
  else
    overwrite e board alpha score

def probe (cells : Array Entry) (idx : Nat) (board : Board) (alpha : Int) : Option (Option Int) :=
  (cells[idx]?).map fun e => probeCell e board alpha

def store (cells : Array Entry) (idx : Nat) (board : Board) (alpha score : Int) : Option (Array Entry) :=
  if h : idx < cells.size then some (cells.set idx (storeCell cells[idx] board alpha score)) else none

end NeuralReversi.EndgameSearch.CacheMir
